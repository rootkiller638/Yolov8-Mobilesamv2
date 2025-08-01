import os
import cv2
import time
import torch
import numpy as np
import threading
from datetime import datetime
from queue import Queue, Empty
from ultralytics import YOLO
from mobile_sam import sam_model_registry, SamPredictor
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=bias,
        )

    def forward(self, x, state):
        h_cur, c_cur = state
        combined = torch.cat([x, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size, device=None):
        height, width = image_size
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
        )


class ConvLSTM(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        kernel_size,
        num_layers,
        batch_first=False,
        bias=True,
        return_all_layers=False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim if isinstance(hidden_dim, list) else [hidden_dim] * num_layers
        self.kernel_size = kernel_size if isinstance(kernel_size, list) else [kernel_size] * num_layers
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        self.cell_list = nn.ModuleList()
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            self.cell_list.append(
                ConvLSTMCell(cur_input_dim, self.hidden_dim[i], self.kernel_size[i], bias=self.bias)
            )

    def forward(self, input_tensor):
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        b, seq_len, _, h, w = input_tensor.size()

        hidden_state = [cell.init_hidden(b, (h, w), device=input_tensor.device) for cell in self.cell_list]

        layer_output_list = []
        last_state_list = []
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](cur_layer_input[:, t], [h, c])
                output_inner.append(h)
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]
        return layer_output_list, last_state_list


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

rtsp_url = "rtsp://用户名:密码@ip地址/live/ch00_0"
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

model_yolo = "./weights/yolov8n.pt"
model_yolo_pose = "./weights/yolov8n-pose.pt"
model_sam_ckpt = "./weights/mobile_sam.pt"
model_type = "vit_t"
conf_threshold = 0.5
mask_alpha = 0.4
sam_interval = 1
resize_ratio = 0.75
retry_delay = 2
frame_queue_size = 2

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device.upper()}")

yolo = YOLO(model_yolo).to(device)
yolo_pose = YOLO(model_yolo_pose).to(device)
sam = sam_model_registry[model_type](checkpoint=model_sam_ckpt).to(device).eval()
predictor = SamPredictor(sam)

if device == "cuda":
    yolo.model.half()
    yolo_pose.model.half()
    torch.set_float32_matmul_precision("high")

YOLO_CLASSES = yolo.names
np.random.seed(42)
CLASS_COLORS = {i: tuple(np.random.randint(0, 256, 3).tolist()) for i in range(len(YOLO_CLASSES))}

conv_lstm = ConvLSTM(input_dim=3, hidden_dim=16, kernel_size=(3, 3), num_layers=1, batch_first=True).to(device)

SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6),
    (5, 11), (6, 12), (11, 12), (11, 13),
    (13, 15), (12, 14), (14, 16),
]


def draw_skeleton(frame, keypoints, color=(0, 255, 0)):
    for x, y, conf in keypoints:
        if conf > 0.3:
            cv2.circle(frame, (int(x), int(y)), 3, color, -1)
    for i, j in SKELETON:
        if keypoints[i][2] > 0.3 and keypoints[j][2] > 0.3:
            cv2.line(
                frame,
                (int(keypoints[i][0]), int(keypoints[i][1])),
                (int(keypoints[j][0]), int(keypoints[j][1])),
                color,
                2,
            )
    return frame


def apply_mask_fast(image, mask, color, alpha=0.4):
    color_img = np.zeros_like(image)
    color_img[:] = color
    blended = cv2.addWeighted(image, 1 - alpha, color_img, alpha, 0)
    return np.where(mask[..., None], blended, image).astype(np.uint8)


class RTSPPipeline:
    def __init__(self, url):
        self.url = url
        self.cap = None
        self.frame_queue = Queue(maxsize=frame_queue_size)
        self.result_queue = Queue(maxsize=frame_queue_size)
        self.running = True
        self.frame_count = 0
        self.last_sam_frame = -1
        self.retry_count = 0

        self.orig_width = 1280
        self.orig_height = 720
        self.proc_width = int(self.orig_width * resize_ratio)
        self.proc_height = int(self.orig_height * resize_ratio)

        self.capture_thread = threading.Thread(target=self._capture, daemon=True)
        self.process_thread = threading.Thread(target=self._process, daemon=True)
        self.capture_thread.start()
        self.process_thread.start()

        self.seq_len = 6
        self.frame_buffer = []

    def _open_capture(self):
        if self.cap:
            self.cap.release()
        print(f"Connecting to RTSP: {self.url}")
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print("RTSP connected.")
            return True
        print("RTSP connection failed.")
        return False

    def _capture(self):
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                if not self._open_capture():
                    time.sleep(retry_delay)
                    continue

            ret, frame = self.cap.read()
            if not ret:
                print(f"Frame read failed, retrying ({self.retry_count})")
                self.retry_count += 1
                time.sleep(retry_delay)
                self.cap.release()
                self.cap = None
                continue

            self.retry_count = 0
            small_frame = cv2.resize(frame, (self.proc_width, self.proc_height))
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except Empty:
                    pass
            self.frame_queue.put((small_frame, frame.copy()))

    def _process(self):
        while self.running:
            try:
                small_frame, orig_frame = self.frame_queue.get(timeout=0.1)
                self.frame_count += 1

                frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                overlay = frame_rgb.copy()

                with torch.inference_mode():
                    results = yolo.track(
                        frame_rgb,
                        conf=conf_threshold,
                        persist=True,
                        verbose=False,
                        half=True,
                    )[0]

                frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                self.frame_buffer.append(frame_tensor)
                if len(self.frame_buffer) >= self.seq_len:
                    seq_tensor = torch.stack(self.frame_buffer[-self.seq_len:], dim=1).to(device)
                    _, last_state = conv_lstm(seq_tensor)

                if results.boxes is not None and len(results.boxes) > 0:
                    boxes = results.boxes.xyxy.cpu().numpy()
                    classes = results.boxes.cls.cpu().numpy().astype(int)
                    scores = results.boxes.conf.cpu().numpy()
                    ids = (
                        results.boxes.id.cpu().numpy().astype(int)
                        if results.boxes.id is not None
                        else np.arange(len(boxes))
                    )

                    run_sam = (self.frame_count - self.last_sam_frame) >= sam_interval
                    if run_sam:
                        predictor.set_image(frame_rgb)
                        self.last_sam_frame = self.frame_count

                    for i, box in enumerate(boxes):
                        xmin, ymin, xmax, ymax = map(int, box)
                        cls_id = classes[i]
                        track_id = ids[i]
                        color = CLASS_COLORS.get(cls_id, (0, 255, 255))

                        cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), color, 2)
                        name = YOLO_CLASSES.get(cls_id, f"class_{cls_id}")
                        label = f"ID {track_id} | {name} {scores[i]:.2f}"
                        cv2.putText(
                            overlay,
                            label,
                            (xmin, max(ymin - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2,
                        )

                        if run_sam:
                            input_box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
                            mask, _, _ = predictor.predict(
                                point_coords=None,
                                point_labels=None,
                                box=input_box[None, :],
                                multimask_output=False,
                            )
                            overlay = apply_mask_fast(overlay, mask[0], color, mask_alpha)

                        if cls_id == 0:
                            crop = frame_rgb[ymin:ymax, xmin:xmax]
                            if crop.size > 0:
                                pose_results = yolo_pose(crop, conf=0.3, verbose=False)[0]
                                if pose_results.keypoints is not None:
                                    for kp in pose_results.keypoints.xy.cpu().numpy():
                                        kp[:, 0] += xmin
                                        kp[:, 1] += ymin
                                        overlay = draw_skeleton(
                                            overlay,
                                            np.hstack(
                                                [kp, pose_results.keypoints.conf.cpu().numpy()[0, :, None]]
                                            ),
                                        )

                result_frame = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                if self.result_queue.full():
                    try:
                        self.result_queue.get_nowait()
                    except Empty:
                        pass
                self.result_queue.put((result_frame, orig_frame))

            except Empty:
                pass
            except Exception as e:
                print(f"Processing error: {str(e)}")
                time.sleep(0.1)

    def get_result(self):
        try:
            return self.result_queue.get_nowait()
        except Empty:
            return None

    def stop(self):
        self.running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
        if self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        if self.process_thread.is_alive():
            self.process_thread.join(timeout=1.0)


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_processed_path = os.path.join(output_dir, f"output_{timestamp}_processed.mp4")

    print("Starting RTSP pipeline...")
    pipeline = RTSPPipeline(rtsp_url)

    out_processed = None
    fps_counter, last_time = 0, time.time()
    fps = 0
    last_frame_time = time.time()

    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)

    while True:
        result = pipeline.get_result()
        if result:
            result_frame, orig_frame = result
            fps_counter += 1

            current_time = time.time()
            if current_time - last_time >= 1.0:
                fps = fps_counter / (current_time - last_time)
                fps_counter = 0
                last_time = current_time
                print(f"FPS: {fps:.1f}")

            full_size_frame = cv2.resize(result_frame, (pipeline.orig_width, pipeline.orig_height))

            cv2.putText(full_size_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if out_processed is None:
                h, w = full_size_frame.shape[:2]
                out_processed = cv2.VideoWriter(
                    output_processed_path, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h)
                )
                print(f"Saving processed video to: {output_processed_path}")

            cv2.imshow("Result", full_size_frame)
            out_processed.write(full_size_frame)

            last_frame_time = current_time
        else:
            if time.time() - last_frame_time > 5.0:
                print("No frame received for 5 seconds, checking connection...")
                last_frame_time = time.time()

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord("p"):
            cv2.waitKey(0)

    pipeline.stop()
    if out_processed:
        out_processed.release()
    cv2.destroyAllWindows()
    print(f"Processing finished, video saved to: {output_processed_path}")


if __name__ == "__main__":
    main()

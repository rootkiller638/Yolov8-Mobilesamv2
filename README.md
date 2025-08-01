# 🎥 YOLOv8 + MobileSAM + Pose + ConvLSTM 实时视频分析

基于 YOLOv8 的目标检测、MobileSAM 分割、人体姿态识别与时序特征提取的实时视频分析系统
支持 RTSP 流实时处理、自动重连、对象跟踪与可视化。

## 📖 项目简介

本项目实现了一个基于 YOLOv8 + MobileSAM + ConvLSTM 的实时视频处理管道，支持从 RTSP 流 获取视频，并完成以下功能：

目标检测与跟踪

使用 YOLOv8 进行实时检测与多目标跟踪，支持 ID 追踪与置信度显示。

目标分割（Instance Segmentation）

集成 MobileSAM，通过 YOLO 检测框触发分割，实现对象级别分割掩码与半透明可视化。

人体姿态识别（Pose Estimation）

对检测到的 person 类目标，使用 YOLOv8-Pose 进行人体关键点检测与骨架绘制。

视频流缓存与时序建模

使用轻量级 ConvLSTM 模块对连续帧进行编码，可为后续行为识别或时序分析提供特征支持。

实时处理与显示

支持实时 FPS 计算、处理结果可视化，并可保存处理后的视频文件。

## ⚙️ 系统功能流程
mermaid
复制
编辑

flowchart LR
    A[RTSP 视频流] -->|读取帧| B[YOLOv8 检测与跟踪]
    B -->|检测框| C[MobileSAM 分割]
    B -->|person类别| D[YOLOv8-Pose 关键点检测]
    C --> E[可视化融合显示]
    D --> E[可视化融合显示]
    E --> F[ConvLSTM 时序编码]
    F --> G[处理后视频显示与保存]
    
📦 环境依赖

bash
复制
编辑

# 1️⃣ 创建虚拟环境
conda create -n yolosam python=3.10 -y

conda activate yolosam


# 2️⃣ 安装依赖

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install ultralytics opencv-python pycocotools onnxruntime

pip install supervision lap

pip install git+https://github.com/ChaoningZhang/MobileSAM.git

⚡ 可选加速：若使用 RTX GPU，推荐开启 --half 模式，YOLOv8 推理可使用 FP16 加速。

## 🚀 使用方法

## 1️⃣ 克隆项目

bash

复制

编辑

git clone https://github.com/yourusername/YOLOv8-MobileSAM-Pose-RTSP.git

cd YOLOv8-MobileSAM-Pose-RTSP

## 2️⃣ 下载模型

YOLOv8 模型（目标检测 & 姿态识别）：

bash

复制

编辑

# 例：下载官方预训练模型

wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt

MobileSAM 模型：

bash
复制
编辑
wget https://github.com/ChaoningZhang/MobileSAM/releases/download/model/mobile_sam.pt
将模型放置在 weights/ 目录。

3️⃣ 运行示例
bash

复制

编辑

python app.py \
    --rtsp "rtsp://username:password@192.168.1.100:554/stream" \
    --yolo-weights weights/yolov8n.pt \
    --pose-weights weights/yolov8n-pose.pt \
    --sam-weights weights/mobile_sam.pt \
    --save-output
--rtsp：RTSP 视频流地址

--save-output：是否保存处理后的视频（默认保存在 output/）

## 🖼️ 可视化效果

目标检测 & 分割：

检测框、类别与置信度显示

SAM 分割掩码半透明融合

人体姿态识别：

仅对 person 类绘制骨架关键点

## 🔮 后续可扩展方向

✅ 行为识别：结合 ConvLSTM 输出特征进行动作分类

✅ 多路 RTSP 流并行处理

✅ Web Dashboard 实时显示分析结果

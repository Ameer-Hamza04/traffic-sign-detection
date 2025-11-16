# 🚦 Traffic Sign Detection Using YOLO (YOLOv8, YOLO11, YOLO12)

This repository contains the implementation of my project **“Traffic Sign Detection Using YOLO Architectures”**, which develops, trains, evaluates, and compares multiple YOLO models for robust real-world traffic sign detection.

---

## 📌 Project Overview

Traffic sign detection is a crucial component of autonomous driving and ADAS systems. This project:

- Builds a complete end-to-end detection pipeline  
- Trains YOLOv8, YOLO11, and YOLO12 models  
- Evaluates multiple model scales (n/s/m/l)  
- Tests robustness to blur, lighting variation, and scale changes  
- Fine-tunes top models for enhanced accuracy  

The best-performing model achieves **0.850 mAP@50-95 (YOLO11-l)**.

---

## 🧠 Model Architectures Evaluated

| Architecture | Highlights |
|-------------|------------|
| **YOLOv8** | Fast, anchor-free, modern baseline |
| **YOLO11** | Improved backbone & IoU accuracy |
| **YOLO12** | Hybrid CNN-Transformer + attention neck |

---

## 📂 Dataset & Preprocessing

### Dataset

- Multi-source dataset from traffic-scene collections  
- Includes varied environments: urban, rural, highways  
- Challenging conditions included:
  - Motion blur  
  - Different lighting  
  - Occlusions  
  - Distant/small signs  

### Annotation & Splits

- High-quality bounding-box annotations  
- Stratified split: **70% train / 15% val / 15% test**

### Data Augmentation

- Rotations (±15°)  
- Scaling (0.8×–1.2×)  
- Horizontal flips  
- Brightness/contrast/saturation jitter  
- Motion blur  
- Random occlusions and cutout  

---

## ⚙️ Training Setup

- **Framework:** Ultralytics YOLO  
- **Language:** Python 3.8+  
- **Hardware (used in this project):** 4× GTX 1080 Ti (CUDA)  

### Common Training Settings

- Image size: `640×640`  
- Epochs: `70`  
- Optimizer: `auto` (AdamW / SGD chosen by Ultralytics)  
- Learning rate schedule: cosine annealing  
- Early stopping enabled  

### Loss Function

YOLO minimizes three components:

```text
L_total = L_obj + L_cls + L_bbox
```

- Objectness loss: Binary Cross-Entropy (BCE)  
- Classification loss: BCE  
- Bounding box loss: CIoU  

---

## 📊 Results

### Main Performance Table

| Model     | mAP@50 | mAP@50-95 |
|----------|--------|-----------|
| YOLOv8-s | 0.974  | 0.847     |
| YOLO11-l | 0.973  | **0.848** |
| YOLO12-m | 0.974  | 0.847     |

### After Fine-Tuning

| Model     | mAP@50-95          |
|----------|---------------------|
| **YOLO11-l** | **0.850 (Best)** |
| YOLO12-m | 0.848              |
| YOLOv8-m | 0.849              |

### Robustness

The models show strong performance under:

- Motion blur  
- Low-light images  
- Long-distance small signs  
- Unseen test data  

---

## 🏆 Model Recommendations

- **Highest Accuracy:** YOLO11-l  
- **Best for Embedded Hardware:** YOLOv8-s  
- **General Automotive Use:** YOLOv8-m  
- **Research Exploration:** YOLO12-m  

---

## 📁 Repository Structure

You can organize this repository as follows (example):

```text
📦 Traffic-Sign-Detection
 ┣ 📁 data/                # Dataset (not included in this repo)
 ┣ 📁 notebooks/           # Jupyter notebooks used in the project
 ┣ 📁 models/              # Model configs/weights (if shared)
 ┣ 📁 results/             # Charts, metrics, sample predictions
 ┣ 📄 requirements.txt
 ┣ 📄 README.md
 ┗ 📄 train.py             # Example training script (optional)
```

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train a YOLO model

Example with YOLOv8:

```bash
yolo detect train model=yolov8s.pt data=traffic_sign.yaml imgsz=640 epochs=70
```

### 3. Validate

```bash
yolo detect val model=best.pt data=traffic_sign.yaml
```

### 4. Run Inference

```bash
yolo detect predict model=best.pt source=images/test/
```

Make sure `traffic_sign.yaml` points to your dataset paths and class names.

---

## 🔮 Future Work

Potential extensions of this project:

- Cross-country / cross-domain evaluation on other traffic sign datasets  
- Temporal (video-based) consistency analysis for smoother predictions  
- Edge deployment on devices like NVIDIA Jetson  
- Advanced augmentations for heavy rain, snow, fog, and night conditions  
- Exploring transformer-heavy architectures beyond YOLO12  

---

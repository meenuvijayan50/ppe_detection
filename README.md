# PPE Detection for Edge AI

Real-time Personal Protective Equipment (PPE) detection using YOLOv8 with ONNX conversion and FP16 quantization for edge deployment.

## Problem Statement

Industrial worksites require workers to wear safety equipment such as helmets, gloves, jackets, and safety shoes.

Manual monitoring is difficult and error-prone.

This project detects:

- PPE being worn
- PPE requirement signs


The goal is to build a lightweight object detection model and optimize it for edge inference while maintaining usable accuracy.

---

# Dataset

Dataset contains **8 classes**:

```yaml
names:
  - Jacket
  - hard_hat
  - Gloves
  - Shoes
  - Hard_hat_required
  - Shoes_required
  - Jacket_required
  - Gloves_required

```

Dataset split:

```yaml
train: /content/drive/MyDrive/Colab Notebooks/PPE/dataset/train
val: /content/drive/MyDrive/Colab Notebooks/PPE/dataset/valid/images
test: /content/drive/MyDrive/Colab Notebooks/PPE/dataset/test/images
```

Classes:

| Class | Description |
|---|---|
| Jacket | Safety jacket worn |
| hard_hat | Helmet worn |
| Gloves | Gloves worn |
| Shoes | Safety shoes worn |
| Hard_hat_required | Helmet sign |
| Shoes_required | Shoes sign |
| Jacket_required | Jacket sign |
| Gloves_required | Gloves sign |


---

# Model

Baseline model:

- **YOLOv8n**
- Precision: **FP32**

Edge optimized model:

- **ONNX**
- Quantization: **FP16**

Reason:

- Smaller model size
- Faster inference
- Good accuracy retention
- Easy deployment on edge hardware

---

# Project Structure

```bash
ppe_detection/
│
├── README.md
├── dataset.yaml
├── train.ipynb
├── export_onnx.py
├── live_inference.py
│
├── weights/
│   ├── best.pt
│   └── best.onnx
│
└── ppe_input.mp4
```

---

# Training

Install:

```bash
pip install ultralytics opencv-python onnxruntime
```

Train:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="dataset.yaml",
    epochs=50,
    imgsz=640
)
```

Output:

```bash
runs/detect/train/weights/best.pt
```

---

# ONNX Export + FP16 Quantization

Export:

```python
from ultralytics import YOLO

model = YOLO("best.pt")

model.export(
    format="onnx",
    half=True
)
```

Output:

```bash
best.onnx
```

---

# Live Inference

Run:

```bash
python live_inference.py
```

Features:

- Bounding boxes
- class labels
- confidence scores
- FPS
- preprocessing latency
- postprocessing latency

Input:

```bash
ppe_input.mp4
```

---

# Performance Benchmark

| Metric | FP32 (.pt) | ONNX FP16 |
|---|---:|---:|
| Model Size | XX MB | XX MB |
| mAP50-95 | XX | XX |
| FPS | XX | XX |
| Preprocess Latency | XX ms | XX ms |
| Postprocess Latency | XX ms | XX ms |

> Replace XX with measured values.

---

# Model Weights

FP32:

**best.pt**

ONNX FP16:

**best.onnx**

Upload to:

- Google Drive
- Hugging Face

Add links here.

Example:

```md
FP32 weights: [https://drive.google.com/...](https://drive.google.com/file/d/1zMDOwhExV5ppZp7AmPgycxW4FDMJ5qM8/view?usp=sharing)

ONNX weights: [https://drive.google.com/...](https://drive.google.com/file/d/1JumeHPj8aUlAEbnQNKY4Soqyg6Zajy0g/view?usp=sharing)
```

---




# Summary

This project demonstrates:

✅ PPE detection using YOLOv8n  
✅ Edge deployment with ONNX  
✅ FP16 quantization  
✅ Real-time inference  
✅ Performance benchmarking

Designed for industrial safety monitoring on edge devices.

# 🖼️ Production-Ready Image Classification System

> CNN-based image classifier built with PyTorch, served via FastAPI, and exported to ONNX for optimized deployment.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?style=flat-square&logo=pytorch)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi)
![ONNX](https://img.shields.io/badge/ONNX-Export-gray?style=flat-square&logo=onnx)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker)
![License](https://img.shields.io/badge/License-Educational-green?style=flat-square)

---

## 📌 Overview

This project implements an end-to-end **production-ready image classification system** trained on the CIFAR-10 dataset. Users can upload an image via a REST API and receive real-time predictions. The model is exported to **ONNX format** for optimized inference and deployment readiness.

---

## ✨ Features

- ✅ CNN-based image classification using **PyTorch**
- ✅ Real-time prediction via **FastAPI** REST API
- ✅ **ONNX** model export for optimized inference
- ✅ Modular and production-ready code structure
- ✅ Automated dataset download and training pipeline
- ✅ REST API testing using **Swagger UI**
- ✅ **Docker-ready** deployment support

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| **Machine Learning** | PyTorch, Torchvision, NumPy |
| **Backend / API** | FastAPI, Uvicorn |
| **Image Processing** | Pillow, OpenCV |
| **Deployment** | ONNX, Docker |
| **Language** | Python 3.10+ |

---

## 📂 Project Structure

```
image-classification-system/
│
├── api/
│   └── main.py              # FastAPI inference server
│
├── src/
│   ├── model.py             # CNN architecture
│   ├── train.py             # Model training script
│   ├── predict.py           # Prediction logic
│   └── export_onnx.py       # ONNX export script
│
├── models/
│   └── cnn_model.pth        # Trained PyTorch model
│
├── onnx/
│   └── model.onnx           # Exported ONNX model
│
├── data/                    # Dataset folder (auto-downloaded)
│
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🧠 Model Architecture

The CNN consists of **3 convolutional blocks** followed by fully connected layers with dropout regularization.

```
Input Image (32x32x3)
       │
  ┌────▼────┐
  │  Conv2D  │ → ReLU → MaxPool
  └────┬────┘
       │
  ┌────▼────┐
  │  Conv2D  │ → ReLU → MaxPool
  └────┬────┘
       │
  ┌────▼────┐
  │  Conv2D  │ → ReLU → MaxPool
  └────┬────┘
       │
  ┌────▼────┐
  │    FC    │ → Dropout → Output (10 classes)
  └─────────┘
```

---

## 🗂️ Dataset — CIFAR-10

Auto-downloaded via Torchvision. Contains **60,000 images** across 10 classes:

`✈️ Airplane` · `🚗 Automobile` · `🐦 Bird` · `🐱 Cat` · `🦌 Deer`  
`🐶 Dog` · `🐸 Frog` · `🐴 Horse` · `🚢 Ship` · `🚚 Truck`

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/production-image-classification-system.git
cd production-image-classification-system
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🏋️ Train the Model

```bash
cd src
python train.py
```

Model will be saved to:
```
models/cnn_model.pth
```

---

## 🌐 Run the API Server

```bash
cd api
python -m uvicorn main:app --reload
```

| URL | Purpose |
|---|---|
| `http://127.0.0.1:8000` | Base server |
| `http://127.0.0.1:8000/docs` | Swagger UI |

---

## 📡 API Usage

### `POST /predict`

Upload any `.jpg` or `.png` image to get a prediction.

**Example Response:**
```json
{
  "prediction": "cat"
}
```

You can test this directly through the **Swagger UI** at `/docs`.

---

## 📦 Export to ONNX

```bash
cd src
python export_onnx.py
```

Exported model saved at:
```
onnx/model.onnx
```

---

## 🐳 Docker Deployment

```bash
# Build the image
docker build -t image-classifier .

# Run the container
docker run -p 8000:8000 image-classifier
```

---

## 📊 Results

- ✔️ CNN model successfully trained on CIFAR-10
- ✔️ Real-time image classification via REST API
- ✔️ ONNX model exported for optimized deployment
- ✔️ Modular, production-ready architecture

---

## 🔮 Future Improvements

- [ ] Upgrade to **ResNet50** for higher accuracy
- [ ] Deploy on cloud (**AWS / Render / Railway**)
- [ ] Add a **frontend UI** for easy image uploads
- [ ] Support **batch prediction**

---

# Production-Ready Image Classification System using CNN, PyTorch, FastAPI, and ONNX

## Overview

This project implements a production-ready image classification system built with PyTorch and deployed using FastAPI. The system allows users to upload an image via a REST API and receive real-time predictions. The model is trained on the CIFAR-10 dataset and exported to ONNX format for optimized inference and deployment readiness.

This project demonstrates end-to-end machine learning pipeline development, including model training, inference API creation, and deployment preparation.

---

## Features

* CNN-based image classification using PyTorch
* Real-time prediction using FastAPI REST API
* ONNX model export for optimized inference
* Modular and production-ready code structure
* Automated dataset download and training pipeline
* REST API testing using Swagger UI
* Docker-ready deployment support

---

## Tech Stack

**Machine Learning**

* PyTorch
* Torchvision
* NumPy

**Backend / API**

* FastAPI
* Uvicorn

**Image Processing**

* Pillow
* OpenCV

**Deployment / Optimization**

* ONNX
* Docker

**Programming Language**

* Python 3.10+

---

## Dataset

Dataset used: **CIFAR-10**

Classes:

* Airplane
* Automobile
* Bird
* Cat
* Deer
* Dog
* Frog
* Horse
* Ship
* Truck

Dataset is automatically downloaded using Torchvision.

---

## Project Structure

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
│   └── export_onnx.py      # ONNX export script
│
├── models/
│   └── cnn_model.pth       # Trained PyTorch model
│
├── onnx/
│   └── model.onnx          # Exported ONNX model
│
├── data/                   # Dataset folder
│
├── requirements.txt       # Dependencies
├── Dockerfile             # Docker deployment config
└── README.md
```

---

## Model Architecture

The CNN model consists of:

* 3 Convolutional layers
* ReLU activation
* MaxPooling layers
* Fully connected layers
* Dropout regularization

Architecture flow:

```
Input Image → Conv → ReLU → Pool → Conv → ReLU → Pool → Conv → ReLU → Pool → FC → Output
```

---

## Installation

### Step 1: Clone repository

```
git clone https://github.com/YOUR_USERNAME/production-image-classification-system.git
cd production-image-classification-system
```

---

### Step 2: Create virtual environment

```
python -m venv venv
venv\Scripts\activate
```

---

### Step 3: Install dependencies

```
pip install -r requirements.txt
```

---

## Training the Model

Run:

```
cd src
python train.py
```

Output:

```
Model saved at models/cnn_model.pth
```

---

## Run the API Server

```
cd api
python -m uvicorn main:app --reload
```

Server will start at:

```
http://127.0.0.1:8000
```

Swagger UI available at:

```
http://127.0.0.1:8000/docs
```

---

## API Usage

### Endpoint

```
POST /predict
```

### Input

* Image file (.jpg, .png)

### Output Example

```
{
  "prediction": "cat"
}
```

---

## ONNX Export

To export model to ONNX format:

```
cd src
python export_onnx.py
```

Output:

```
onnx/model.onnx
```

---

## Docker Deployment (Optional)

Build Docker image:

```
docker build -t image-classifier .
```

Run container:

```
docker run -p 8000:8000 image-classifier
```

---

## Results

* Successfully trained CNN model on CIFAR-10 dataset
* Real-time image classification using REST API
* Production-ready modular architecture
* ONNX model exported for optimized deployment

---

## Skills Demonstrated

* Deep Learning
* Computer Vision
* CNN Architecture
* Model Deployment
* FastAPI
* REST API Development
* PyTorch
* ONNX
* Docker
* Backend Integration

---

## Future Improvements

* Use ResNet50 for higher accuracy
* Deploy on cloud (AWS / Render / Railway)
* Add frontend UI
* Add batch prediction support

---

## Author

Nishchal Dagar
B.Tech Computer Science (Data Science)
Machine Learning & Data Science Enthusiast

---

## License

This project is for educational and portfolio purposes.

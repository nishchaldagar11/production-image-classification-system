import torch
import torchvision.transforms as transforms
from PIL import Image

from model import CNNModel


# CIFAR10 classes
classes = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]


# model load karo
model = CNNModel()

model.load_state_dict(
    torch.load("../models/cnn_model.pth", map_location=torch.device("cpu"))
)

model.eval()


# transform define karo
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])


# prediction function
def predict(image_path):

    image = Image.open(image_path).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    with torch.no_grad():
        output = model(image)

    _, predicted = torch.max(output, 1)

    return classes[predicted.item()]
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import os

from model import CNNModel


# ensure models folder exists
os.makedirs("../models", exist_ok=True)


# data transform
transform = transforms.Compose([
    transforms.ToTensor()
])


# load dataset
trainset = torchvision.datasets.CIFAR10(
    root="../data",
    train=True,
    download=True,
    transform=transform
)


trainloader = DataLoader(
    trainset,
    batch_size=32,
    shuffle=True
)


# model create
model = CNNModel()


# loss and optimizer
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# training
epochs = 5

print("Training started...")


for epoch in range(epochs):

    running_loss = 0.0

    for images, labels in trainloader:

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1} completed, Loss: {running_loss}")


# save model
torch.save(
    model.state_dict(),
    "../models/cnn_model.pth"
)

print("Model saved successfully at ../models/cnn_model.pth")
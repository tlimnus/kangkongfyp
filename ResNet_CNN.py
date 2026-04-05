import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import xml.etree.ElementTree as ET
from PIL import Image
import os


# ------------------------
# CONFIG
# ------------------------
TRAIN_DIR = r"C:\FYPData\datasetkk_train"
VAL_DIR   = r"C:\FYPData\datasetkk_val"
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-4

# ------------------------
# DEVICE
# ------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ------------------------
# TRANSFORMS
# ------------------------
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.05
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ------------------------
# DATASETS
# ------------------------
train_data = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
val_data   = datasets.ImageFolder(VAL_DIR, transform=val_transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=BATCH_SIZE)

num_classes = len(train_data.classes)
print("Classes:", train_data.classes)

# ------------------------
# MODEL
# ------------------------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# ------------------------
# LOSS & OPTIMIZER
# ------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ------------------------
# TRAINING LOOP
# ------------------------
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)

    # ------------------------
    # VALIDATION
    # ------------------------
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f}, Val Acc: {acc:.2f}%")

# ------------------------
# SAVE MODEL
# ------------------------
torch.save(model.state_dict(), "plant_disease_modelppy_c.pth")
print("Training complete. Model saved.")

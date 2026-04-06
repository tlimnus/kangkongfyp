import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import pathlib

# ------------------------
# CONFIG
# ------------------------
filePath = pathlib.Path(__file__).parent.resolve()
models_dir = filePath / "models"
TRAIN_DIR = filePath / "datasetkk_train"
VAL_DIR   = filePath / "datasetkk_val"
BATCH_SIZE = 20
EPOCHS = 50
WEIGHT_DECAY = 1e-4

# Differential learning rates:
# Low LR for pretrained backbone (avoid destroying learned features)
# Higher LR for the new classifier head
BACKBONE_LR = 1e-4
HEAD_LR = 1e-3

# ------------------------
# DEVICE
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ------------------------
# TRANSFORMS
# ------------------------
# Same transforms as DINOv2 script for fair comparison
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.03
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
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

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

num_classes = len(train_data.classes)
print(f"Classes ({num_classes}): {train_data.classes}")
print(f"Training samples: {len(train_data)}")
print(f"Validation samples: {len(val_data)}")

# ------------------------
# LOAD EFFICIENTNETV2-S
# ------------------------
print("\nLoading EfficientNetV2-S (pretrained on ImageNet)...")
model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)

# Replace the classifier head for our number of classes
# EfficientNetV2-S classifier: Sequential(Dropout, Linear(1280, 1000))
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(1280, num_classes)
)
model = model.to(device)

# Count parameters
backbone_params = sum(p.numel() for name, p in model.named_parameters() if "classifier" not in name)
head_params = sum(p.numel() for name, p in model.named_parameters() if "classifier" in name)
total_params = backbone_params + head_params
print(f"Backbone parameters (fine-tuned): {backbone_params:,}")
print(f"Classifier head parameters: {head_params:,}")
print(f"Total parameters: {total_params:,}")
print(f"NOTE: ALL parameters are trainable (full fine-tuning)\n")

# ------------------------
# LOSS & OPTIMIZER
# ------------------------
# Differential learning rates: backbone gets lower LR to preserve pretrained features
backbone_param_list = [p for name, p in model.named_parameters() if "classifier" not in name]
head_param_list = [p for name, p in model.named_parameters() if "classifier" in name]

optimizer = torch.optim.AdamW([
    {"params": backbone_param_list, "lr": BACKBONE_LR},
    {"params": head_param_list,     "lr": HEAD_LR},
], weight_decay=WEIGHT_DECAY)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# Class weights — use sqrt-softened version to avoid over-penalizing
class_counts = [75, 28, 54]  # UPDATE THESE to match your train split
weights = torch.tensor(
    [np.sqrt(sum(class_counts) / c) for c in class_counts],
    dtype=torch.float
).to(device)

criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

# ------------------------
# TRAINING LOOP
# ------------------------
best_val_loss = float("inf")
best_acc = 0.0
patience = 15
patience_counter = 0

for epoch in range(EPOCHS):
    # --- TRAIN ---
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Full forward pass — backbone is NOT frozen
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (preds == labels).sum().item()

    scheduler.step()
    avg_loss = running_loss / len(train_loader)
    train_acc = 100 * correct_train / total_train

    # --- VALIDATE ---
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    val_acc = 100 * correct / total
    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.1f}% | "
          f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.1f}% | "
          f"LR backbone: {optimizer.param_groups[0]['lr']:.6f} | "
          f"LR head: {optimizer.param_groups[1]['lr']:.6f}")

    # --- EARLY STOPPING + BEST MODEL ---
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_acc = val_acc
        patience_counter = 0

        torch.save({
            'model_state_dict': model.state_dict(),
            'num_classes': num_classes,
            'classes': train_data.classes,
            'best_val_loss': best_val_loss,
            'best_acc': best_acc,
        }, models_dir / "efficientnetv2s_best.pth")
        print(f"  → Saved best model (Val Loss: {best_val_loss:.4f}, Val Acc: {val_acc:.1f}%)")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

print(f"\n{'='*50}")
print(f"EfficientNetV2-S Training Complete")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Validation accuracy at best loss: {best_acc:.1f}%")
print(f"{'='*50}")

# ------------------------
# CONFUSION MATRIX
# ------------------------
print("\nGenerating confusion matrix on validation set...")

checkpoint = torch.load(models_dir / "efficientnetv2s_best.pth")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("\n" + classification_report(
    all_labels, all_preds,
    target_names=train_data.classes,
    digits=3
))

cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix (rows=actual, cols=predicted):")
print(f"{'':>15}", end="")
for name in train_data.classes:
    print(f"{name:>15}", end="")
print()
for i, row in enumerate(cm):
    print(f"{train_data.classes[i]:>15}", end="")
    for val in row:
        print(f"{val:>15}", end="")
    print()

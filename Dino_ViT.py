import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import os
import pathlib
#.venv\Scripts\activate # Activate virtual environment (Windows)
# ------------------------
# CONFIG
# ------------------------
filePath = pathlib.Path(__file__).parent.resolve()
models_dir = filePath / "models"
TRAIN_DIR = filePath / "datasetkk_train"
VAL_DIR   = filePath / "datasetkk_val"
BATCH_SIZE = 20
EPOCHS = 50          # More epochs is fine — we're only training a tiny head
LR = 8e-4            # Higher LR is fine for a linear probe
WEIGHT_DECAY = 1e-4  # Light regularization helps with small datasets

# DINOv2 variant — choose based on your GPU memory:
#   "dinov2_vits14"  — Small  (21M params, ~350MB) ← start here
#   "dinov2_vitb14"  — Base   (86M params, ~1.4GB)
#   "dinov2_vitl14"  — Large  (300M params, ~4.8GB) ← best if you have the VRAM
#   "dinov2_vitg14"  — Giant  (1.1B params, ~18GB)
DINO_MODEL = "dinov2_vits14"

# ------------------------
# DEVICE
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ------------------------
# TRANSFORMS
# ------------------------
# DINOv2 was trained on 518x518 with patch size 14 (= 37x37 patches)
# but 224x224 works fine (= 16x16 patches) and saves memory.
# Use the same ImageNet normalization DINOv2 was trained with.

# train_transform = transforms.Compose([
#     transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomVerticalFlip(p=0.3),
#     transforms.RandomRotation(30),
#     transforms.ColorJitter(
#         brightness=0.4,
#         contrast=0.4,
#         saturation=0.4,
#         hue=0.1
#     ),
#     transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
#     transforms.RandomErasing(p=0.2),                # Simulates occlusion
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225]
#     )
# ])

# Note: RandomErasing must come after ToTensor since it operates on tensors.
# Reorder so RandomErasing is after ToTensor + Normalize:

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
# LOAD DINOv2 BACKBONE
# ------------------------
print(f"\nLoading {DINO_MODEL}...")
backbone = torch.hub.load("facebookresearch/dinov2", DINO_MODEL)
backbone = backbone.to(device)
backbone.eval()  # FROZEN — we never train this

# Get the embedding dimension from the model
embed_dim = backbone.embed_dim  # 384 for vits14, 768 for vitb14, 1024 for vitl14
print(f"DINOv2 embedding dimension: {embed_dim}")

# Freeze ALL backbone parameters — this is the key difference from your ResNet approach
for param in backbone.parameters():
    param.requires_grad = False

# ------------------------
# CLASSIFICATION HEAD
# ------------------------
# Option A: Simple linear probe (start here — fewest trainable params)
# With 100~ images, this is the safest choice.
classifier = nn.Linear(embed_dim, num_classes).to(device)

# Option B: Small MLP head (try this if linear probe underfits)
# classifier = nn.Sequential(
#     nn.Linear(embed_dim, 256),
#     nn.ReLU(),
#     nn.Dropout(0.3),
#     nn.Linear(256, num_classes)
# ).to(device)

trainable_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in backbone.parameters()) + trainable_params
print(f"\nTrainable parameters: {trainable_params:,}")      # ~1,155 for linear probe with 3 classes
print(f"Frozen backbone parameters: {total_params - trainable_params:,}")
print(f"Ratio: training {trainable_params/total_params*100:.4f}% of total params\n")

# ------------------------
# LOSS & OPTIMIZER
# ------------------------

# Only optimize the classifier head
optimizer = torch.optim.AdamW(classifier.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# Cosine annealing helps squeeze out performance
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ------------------------
# CLASS WEIGHTS
# ------------------------
class_counts = [64, 23, 46]

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
    classifier.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Extract features with frozen DINOv2 — no gradients needed here
        with torch.no_grad():
            features = backbone(images)  # Shape: (batch_size, embed_dim)

        # Only the classifier head is trained
        outputs = classifier(features)
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
    classifier.eval()
    correct = 0
    total = 0
    val_loss = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            features = backbone(images)
            outputs = classifier(features)
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
          f"LR: {scheduler.get_last_lr()[0]:.6f}")

    # --- EARLY STOPPING + BEST MODEL ---
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_acc = val_acc
        patience_counter = 0

        torch.save({
            'classifier_state_dict': classifier.state_dict(),
            'backbone_name': DINO_MODEL,
            'embed_dim': embed_dim,
            'num_classes': num_classes,
            'classes': train_data.classes,
            'best_val_loss': best_val_loss,
            'best_acc': best_acc,
        }, models_dir / "jrfull_dino.pth")
        print(f"  → Saved best model (Val Loss: {best_val_loss:.4f}, Val Acc: {val_acc:.1f}%)")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

print(f"\n{'='*50}")
print(f"Training complete. Best validation loss: {best_val_loss:.4f}")
print(f"Validation accuracy at best loss: {best_acc:.1f}%")
print(f"Model saved to: jrfull_dino.pth")
print(f"{'='*50}")

# ------------------------
# CONFUSION MATRIX
# ------------------------
print("\nGenerating confusion matrix on validation set...")

# Load best model
checkpoint = torch.load(models_dir / "jrfull_dino.pth")
classifier.load_state_dict(checkpoint['classifier_state_dict'])
classifier.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        features = backbone(images)
        outputs = classifier(features)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Per-class breakdown
print("\n" + classification_report(
    all_labels, all_preds,
    target_names=train_data.classes,
    digits=3
))

# Raw confusion matrix
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
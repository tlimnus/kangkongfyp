import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pathlib

# ------------------------
# CONFIG
# ------------------------
filePath = pathlib.Path(__file__).parent.resolve()
models_dir = filePath / "models"
TEST_DIR = filePath / "datasetkk_test"
MODEL_PATH = models_dir / "jrfull_dino.pth"
BATCH_SIZE = 12

# ------------------------
# DEVICE
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ------------------------
# TEST TRANSFORM
# Must match your validation transform
# ------------------------
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ------------------------
# LOAD CHECKPOINT
# ------------------------
checkpoint = torch.load(MODEL_PATH, map_location=device)

backbone_name = checkpoint["backbone_name"]
embed_dim = checkpoint["embed_dim"]
num_classes = checkpoint["num_classes"]
class_names = checkpoint["classes"]

print("Checkpoint backbone:", backbone_name)
print("Checkpoint classes:", class_names)
print("Best validation accuracy during training:", checkpoint["best_acc"])

# ------------------------
# LOAD TEST DATA
# ------------------------
test_data = datasets.ImageFolder(TEST_DIR, transform=test_transform)
test_loader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

print("Test folder classes:", test_data.classes)

# Make sure class order matches training
if test_data.classes != class_names:
    raise ValueError(
        f"Class mismatch!\n"
        f"Checkpoint classes: {class_names}\n"
        f"Test folder classes: {test_data.classes}"
    )

idx_to_class = {i: name for i, name in enumerate(class_names)}

# ------------------------
# REBUILD MODEL
# ------------------------
print(f"\nLoading backbone: {backbone_name}")
backbone = torch.hub.load("facebookresearch/dinov2", backbone_name)
backbone = backbone.to(device)
backbone.eval()

for param in backbone.parameters():
    param.requires_grad = False

classifier = nn.Linear(embed_dim, num_classes).to(device)

# classifier = nn.Sequential(
#     nn.Linear(embed_dim, 256),
#     nn.ReLU(),
#     nn.Dropout(0.3),
#     nn.Linear(256, num_classes)
# ).to(device)
# classifier.load_state_dict(checkpoint["classifier_state_dict"])
# classifier.eval()

# ------------------------
# EVALUATION
# ------------------------
correct = 0
total = 0

class_correct = [0] * num_classes
class_total = [0] * num_classes

all_preds = []
all_labels = []

print("\nPer-image predictions:")
print("=" * 80)

with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)

        features = backbone(images)
        outputs = classifier(features)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)

        total += labels.size(0)
        correct += (preds == labels).sum().item()

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

        start_idx = batch_idx * test_loader.batch_size
        end_idx = start_idx + len(images)
        batch_paths = test_data.samples[start_idx:end_idx]

        for i, (path, _) in enumerate(batch_paths):
            true_idx = labels[i].item()
            pred_idx = preds[i].item()
            conf = probs[i][pred_idx].item()

            class_total[true_idx] += 1
            if pred_idx == true_idx:
                class_correct[true_idx] += 1

            image_name = os.path.basename(path)
            print(
                f"Image: {image_name} | "
                f"Pred: {idx_to_class[pred_idx]} ({conf*100:.1f}%) | "
                f"True: {idx_to_class[true_idx]}"
            )

# ------------------------
# RESULTS
# ------------------------
overall_acc = 100 * correct / total if total > 0 else 0.0

print("\n" + "=" * 80)
print(f"Overall Test Accuracy: {overall_acc:.2f}%")
print("=" * 80)

print("\nPer-class accuracy:")
for i, class_name in enumerate(class_names):
    if class_total[i] > 0:
        acc = 100 * class_correct[i] / class_total[i]
        print(f"{class_name}: {acc:.2f}% ({class_correct[i]}/{class_total[i]})")
    else:
        print(f"{class_name}: No test images")

# ------------------------
# CONFUSION MATRIX
# ------------------------
print("\nConfusion Matrix:")
conf_matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]

for true_label, pred_label in zip(all_labels, all_preds):
    conf_matrix[true_label][pred_label] += 1

header = "true\\pred".ljust(15) + "".join(name.ljust(15) for name in class_names)
print(header)

for i, row in enumerate(conf_matrix):
    row_str = class_names[i].ljust(15) + "".join(str(x).ljust(15) for x in row)
    print(row_str)
import shutil
import random
from pathlib import Path

filePath = Path(__file__).resolve().parent
projectRoot = filePath.parent

SOURCE_DIR = projectRoot / "datasetkk"
TRAIN_DIR = projectRoot / "datasetkk_train"
VAL_DIR = projectRoot / "datasetkk_val"

SPLIT_RATIO = 0.8
random.seed(42)

# Optional: clear old split folders before recreating
if TRAIN_DIR.exists():
    shutil.rmtree(TRAIN_DIR)
if VAL_DIR.exists():
    shutil.rmtree(VAL_DIR)

TRAIN_DIR.mkdir(exist_ok=True)
VAL_DIR.mkdir(exist_ok=True)

valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for cls_path in SOURCE_DIR.iterdir():
    if not cls_path.is_dir():
        continue

    images = [img for img in cls_path.iterdir() if img.is_file() and img.suffix.lower() in valid_exts]
    random.shuffle(images)

    split = int(len(images) * SPLIT_RATIO)

    train_cls_dir = TRAIN_DIR / cls_path.name
    val_cls_dir = VAL_DIR / cls_path.name
    train_cls_dir.mkdir(exist_ok=True)
    val_cls_dir.mkdir(exist_ok=True)

    for img in images[:split]:
        shutil.copy(img, train_cls_dir / img.name)

    for img in images[split:]:
        shutil.copy(img, val_cls_dir / img.name)

print("✅ Dataset split complete.")
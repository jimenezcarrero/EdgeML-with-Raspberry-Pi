import os
from pathlib import Path

import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image

# Where to store calibration images
OUT_ROOT = Path("calib_images")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# 1) Load a small, natural-image dataset (CIFAR-10)
transform = transforms.ToTensor()  # we will NOT normalize here
dataset = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=transform,
)

# 2) Map label index -> class name (CIFAR-10 has 10 classes)
classes = dataset.classes  # ['airplane', 'automobile', ..., 'truck']

# 3) Choose how many classes and images per class
num_classes = 10
images_per_class = 10   # 10 x 10 = 100 images

# 4) Collect and save images
counts = {cls: 0 for cls in classes[:num_classes]}

for img, label in dataset:
    cls_name = classes[label]
    if cls_name not in counts:
        continue
    if counts[cls_name] >= images_per_class:
        continue

    # Make class subdir
    class_dir = OUT_ROOT / cls_name
    class_dir.mkdir(parents=True, exist_ok=True)

    idx = counts[cls_name]
    out_path = class_dir / f"img_{idx:04d}.jpg"
    save_image(img, out_path)

    counts[cls_name] += 1

    # Stop when we have enough
    if all(counts[c] >= images_per_class for c in counts):
        break

print("Saved calibration images:")
for cls_name, n in counts.items():
    print(f"  {cls_name}: {n} images")
print(f"\nRoot folder: {OUT_ROOT.resolve()}")

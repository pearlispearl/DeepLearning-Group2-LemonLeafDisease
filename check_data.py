import os
from PIL import Image
import matplotlib.pyplot as plt
import random

dataset_path = r"C:\Users\ASUS\Desktop\DL_Project\Original Dataset"

# ── 1. หาไฟล์ corrupted ──────────────────────────────
print("Checking corrupted files...")
corrupted = []

for class_name in os.listdir(dataset_path):
    class_dir = os.path.join(dataset_path, class_name)
    if os.path.isdir(class_dir):
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            try:
                with Image.open(img_path) as img:
                    img.verify()
            except Exception as e:
                corrupted.append(img_path)
                print(f"  Corrupted: {img_path}")

print(f"Total corrupted: {len(corrupted)} files")

# ── 2. ตรวจ image sizes ──────────────────────────────
print("\nChecking image sizes...")
sizes = []
for class_name in os.listdir(dataset_path):
    class_dir = os.path.join(dataset_path, class_name)
    if os.path.isdir(class_dir):
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            try:
                with Image.open(img_path) as img:
                    sizes.append(img.size)
            except:
                pass

from collections import Counter
print("Top 5 sizes (W×H):")
for size, count in Counter(sizes).most_common(5):
    print(f"  {size[0]}×{size[1]} → {count} images")

# ── 3. แสดง sample รูปจากแต่ละ class ────────────────
print("\nGenerating sample grid...")
fig, axes = plt.subplots(3, 3, figsize=(12, 12))
axes = axes.flatten()

for i, class_name in enumerate(sorted(os.listdir(dataset_path))):
    class_dir = os.path.join(dataset_path, class_name)
    if os.path.isdir(class_dir):
        imgs = [f for f in os.listdir(class_dir)
                if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        sample = random.choice(imgs)
        img = Image.open(os.path.join(class_dir, sample))
        axes[i].imshow(img)
        axes[i].set_title(class_name, fontsize=11)
        axes[i].axis('off')

plt.suptitle('Sample Images — LLDD Dataset', fontsize=14)
plt.tight_layout()
plt.savefig('sample_grid.png', dpi=150)
plt.show()
print("Saved → sample_grid.png")
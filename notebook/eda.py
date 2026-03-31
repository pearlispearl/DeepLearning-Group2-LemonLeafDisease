import os
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter

dataset_path = r"C:\Users\ASUS\Desktop\DL_Project\Original Dataset"  # ปรับถ้า path ต่างกัน

# นับรูปต่อ class
class_counts = {}
for class_name in os.listdir(dataset_path):
    class_dir = os.path.join(dataset_path, class_name)
    if os.path.isdir(class_dir):
        count = len([f for f in os.listdir(class_dir)
                     if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        class_counts[class_name] = count

# Print summary
print("=" * 40)
for cls, cnt in sorted(class_counts.items()):
    print(f"{cls:<25} : {cnt} images")
print("=" * 40)
print(f"Total: {sum(class_counts.values())} images")
print(f"Classes: {len(class_counts)}")

# Plot
plt.figure(figsize=(12, 5))
plt.bar(class_counts.keys(), class_counts.values(), color='steelblue')
plt.xticks(rotation=45, ha='right')
plt.title('Class Distribution — LLDD Dataset')
plt.ylabel('Number of Images')
plt.tight_layout()
plt.savefig('class_distribution.png', dpi=150)
plt.show()
print("Saved → class_distribution.png")

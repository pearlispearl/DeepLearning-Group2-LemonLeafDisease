import torch
import torch.nn as nn
from torchvision import datasets, models
import torchvision.transforms.v2 as transforms_v2
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from model import CNNBaseline,CNNBaseline2, ResNetModel, MobileNetModel, CLASS_NAMES
import numpy as np
import os

# ==========================================
# 🎯 STEP 1: MODEL CONFIGURATION
# Change these values to switch between models
# ==========================================
ITERATION_NAME = "Iteration 2: cnn2.pth"
MODEL_FILENAME = "cnn2.pth"  # Options: cnn.pth, cnn2.pth, resnet.pth, mobilenet.pth
MODEL_TYPE = "SimpleCNN2"    # Options: SimpleCNN, SimpleCNN2, ResNet50, MobileNet
DATASET_PATH = "Lemon_Dataset/test"

print(f"🚀 Starting Evaluation for: {ITERATION_NAME}")
print(f"📦 Using Model File: {MODEL_FILENAME}")


# ==========================================
# ⚙️ STEP 2: INITIALIZATION & LOADING
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_selected_model():
    if MODEL_TYPE == "SimpleCNN":
        model = CNNBaseline()
    elif MODEL_TYPE == "SimpleCNN2":
        model = CNNBaseline2()
    elif MODEL_TYPE == "ResNet50":
        model = ResNetModel(num_classes=9) 
    elif MODEL_TYPE == "MobileNet":
        model = MobileNetModel(num_classes=9)
    
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")
    
    model = model.to(device)
    model_path = os.path.join('..', 'best_model', MODEL_FILENAME)
    
    # 2. Load and validate the checkpoint
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # Check whether the model architecture matches the loaded weights
        model_keys = set(model.state_dict().keys())
        checkpoint_keys = set(checkpoint.keys())
        
        missing_keys = model_keys - checkpoint_keys
        unexpected_keys = checkpoint_keys - model_keys
        
        if len(missing_keys) > 0 or len(unexpected_keys) > 0:
            print(f"⚠️ Warning: Model structure does not match the file {MODEL_FILENAME}")
            print(f"   - Missing keys: {len(missing_keys)}")
            print(f"   - Unexpected keys: {len(unexpected_keys)}")
            # If you want to see exact layers, you can print them
            # print("Missing:", missing_keys)
        else:
            print("✅ Model structure and weights match perfectly!")

        model.load_state_dict(checkpoint)
        model.eval()
        print(f"✅ {MODEL_FILENAME} loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        # Stop execution if loading fails to clearly indicate the issue
        raise e
    
    return model

model = load_selected_model()
    
# ==========================================
# 📊 STEP 3: EVALUATION & REPORTING
# ==========================================
# 1. Prepare Test Dataset
# Dataset
# Transform for CNN (uncomment line below to train CNN)
trans = transforms_v2.Compose([
    transforms_v2.ToImage(),  # Convert to tensor (C, H, W), only needed if you had a PIL image
    transforms_v2.Resize((224, 224)),
    transforms_v2.ToDtype(torch.float32, scale=True),    # Converts the input to a type float32, and rescale from [0, 255] to [0, 1]
    transforms_v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize a tensor with mean and standard deviation.
])
# transfrom for ResNet and MobileNet (uncomment line below to train ResNet and MobileNet)
# trans = transforms_v2.Compose([
#     transforms_v2.Resize((224, 224)),
#     transforms_v2.ToTensor(),
#     transforms_v2.Normalize([0.485, 0.456, 0.406],
#                          [0.229, 0.224, 0.225])
# ])

# Load dataset (ensure path points to the folder containing 9 disease subfolders)
# Example path: 'Lemon_Dataset/test'
data_path = os.path.join('..', 'Lemon_Dataset', 'test')
if os.path.exists(data_path):
    test_data = datasets.ImageFolder(root=data_path, transform=trans)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    print(f"✅ Detected {len(test_data.classes)} classes: {test_data.classes}")
else:
    print("❌ Path not found. Please check your dataset directory.")
    exit()
    
# 2. Evaluation Loop
all_preds = []
all_labels = []

print("🚀 Computing accuracy and performance metrics...")
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 3. Generate Classification Report
print(f"\n📊 Classification Report ({ITERATION_NAME}):")
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

# 4. Generate Confusion Matrix Visualization
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=test_data.classes, yticklabels=test_data.classes)
plt.title(f'Confusion Matrix - {ITERATION_NAME}')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

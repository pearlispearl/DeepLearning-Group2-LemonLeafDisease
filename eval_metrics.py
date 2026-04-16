import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os

# ==========================================
# 🎯 STEP 1: MODEL CONFIGURATION
# Change these values to switch between models
# ==========================================
ITERATION_NAME = "Iteration 4: MobileNetV2"
MODEL_FILENAME = "mobilenet.pth"  # Options: cnn.pth, cnn2.pth, resnet.pth, mobilenet.pth
MODEL_TYPE = "MobileNet"    # Options: SimpleCNN, ResNet50, MobileNet
DATASET_PATH = "Lemon_Dataset/test"

print(f"🚀 Starting Evaluation for: {ITERATION_NAME}")
print(f"📦 Using Model File: {MODEL_FILENAME}")

# ==========================================
# 🧠 STEP 2: MODEL ARCHITECTURE DEFINITION
# ==========================================
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)) 
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 9)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x
    
class SimpleCNN2(nn.Module):
    def __init__(self):
        super(SimpleCNN2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # Added to match Unexpected key conv.8
            nn.BatchNorm2d(128),                           # Added to match Unexpected key conv.9
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128), # Adjusted based on fc.1 error
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 9)    # Final output layer
        )
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# ==========================================
# ⚙️ STEP 3: INITIALIZATION & LOADING
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_selected_model():
    if MODEL_TYPE == "SimpleCNN":
        model = SimpleCNN()
    elif MODEL_TYPE == "SimpleCNN2":
        model = SimpleCNN2()
    elif MODEL_TYPE == "ResNet50":
        model = models.resnet50(weights=None)
        num_ftrs = model.fc.in_features
        # Updated from 128 to 256 as specified in the error message before
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256), # fc.3 maps 256 features to 9 classes
            nn.ReLU(),                # Layer fc.1
            nn.Dropout(0.2),          # Layer fc.2
            nn.Linear(256, 9)         # Layer fc.3: Maps 256 features to 9 classes
        )
    elif MODEL_TYPE == "MobileNet":
        model = models.mobilenet_v2(weights=None)
        # Standard architecture connects directly to the final class layer
        model.classifier[1] = nn.Linear(model.last_channel, 9)
    
    model = model.to(device)
    
    model_path = os.path.join('best_model', MODEL_FILENAME)
    try:
        # 1. Load the model file into the checkpoint variable
        checkpoint = torch.load(model_path, map_location=device)
        
        # 2. Create a new state_dict to remove "model." prefix
        # (to match local model structure)
        new_state_dict = {}
        for k, v in checkpoint.items():
            if k.startswith('model.'):
                # Strip 'model.' prefix to align with standard architecture
                new_state_dict[k.replace('model.', '')] = v
            else:
                new_state_dict[k] = v
        
        # 3. Load the cleaned state_dict into the model
        model.load_state_dict(new_state_dict)
        model.eval()
        print(f"✅ {MODEL_FILENAME} loaded successfully (with prefix stripping)!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    return model

model = load_selected_model()
    
# ==========================================
# 📊 STEP 4: EVALUATION & REPORTING
# ==========================================
# 1. Prepare Test Dataset
# Define image transformations (must match training phase)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load dataset (ensure path points to the folder containing 9 disease subfolders)
# Example path: 'Lemon_Dataset/test'
data_path = 'Lemon_Dataset/test'
if os.path.exists(data_path):
    test_data = datasets.ImageFolder(root=data_path, transform=transform)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    print(f"✅ Detected {len(test_data.classes)} classes: {test_data.classes}")
else:
    print("❌ Path not found. Please check your dataset directory.")

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
print(classification_report(all_labels, all_preds, target_names=test_data.classes))

# 4. Generate Confusion Matrix Visualization
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=test_data.classes, yticklabels=test_data.classes)
plt.title(f'Confusion Matrix - {ITERATION_NAME}')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

# Import Python Packages
import os
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from torchvision import datasets
import torchvision.transforms.v2 as transforms_v2

# Hyperparameters
learning_rate = 1e-4
batch_size = 16
epochs = 20

data_dir = 'Lemon_Dataset'

# Dataset
# Transform for CNN (uncomment line below to train CNN)
# trans = transforms_v2.Compose([
#     transforms_v2.ToImage(),  # Convert to tensor (C, H, W), only needed if you had a PIL image
#     transforms_v2.Resize((224, 224)),
#     transforms_v2.ToDtype(torch.float32, scale=True),    # Converts the input to a type float32, and rescale from [0, 255] to [0, 1]
#     transforms_v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize a tensor with mean and standard deviation.
# ])
# transfrom for ResNet and MobileNet (uncomment line below to train ResNet and MobileNet)
trans = transforms_v2.Compose([
    transforms_v2.Resize((224, 224)),
    transforms_v2.ToTensor(),
    transforms_v2.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Define DataSet for train, valid, and test
train_ds = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=trans)
valid_ds = datasets.ImageFolder(os.path.join(data_dir, 'valid'), transform=trans)
test_ds = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=trans)

# Define DataLoader for train, valid, and test
train_dl = DataLoader(train_ds, batch_size, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size, shuffle=False)
test_dl = DataLoader(test_ds, batch_size, shuffle=False)

# Detect computing device on your computer
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using {device} device")
    
# Import the model class from model.py
from model import  CNNBaseline
# Create a model using the model class from model.py
model =  CNNBaseline().to(device)

print(model)


# feed `batch_x` into the model to test the forward pass
batch_x, batch_y = next(iter(train_dl))
y_hat = model(batch_x.to(device))

from dl_utils import train_one_epoch, test
# Model Training
# Setup tensorboard
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torcheval.metrics.functional import (
    multiclass_accuracy,
    multiclass_f1_score,
    mean_squared_error
)

writer = SummaryWriter(f'./runs/train_dl_{model._get_name()}_{datetime.now().strftime("%Y%m%d-%H%M%S")}')
# Specify loss function
loss_fn = nn.CrossEntropyLoss()
# Specify optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# Training loop
best_vloss = 100000.
for epoch in range(epochs):
    print(f"Epoch {epoch+1} / {epochs}")
    
    # Run train_one_epoch
    train_one_epoch(train_dl, model, loss_fn, optimizer, epoch, device, writer)
    
    # Evaluate the model on the training set 
    train_loss, train_y_preds, train_y_trues = test(train_dl, model, loss_fn, device)
    
    # Evaluate the model on the validation set 
    val_loss, val_y_preds, val_y_trues = test(valid_dl, model, loss_fn, device)
    
    # Performance metrics for training set
    train_perf = {
        'accuracy': multiclass_accuracy(train_y_preds, train_y_trues).item(),
        'f1': multiclass_f1_score(train_y_preds, train_y_trues).item(),
    }
    
    # Performance metrics for validation set
    val_perf = {
        'accuracy': multiclass_accuracy(val_y_preds, val_y_trues).item(),
        'f1': multiclass_f1_score(val_y_preds, val_y_trues).item(),
    }
    
    # Log model training performance
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/valid', val_loss, epoch)
    writer.add_scalar('Accuracy/train', train_perf['accuracy'], epoch)
    writer.add_scalar('Accuracy/valid', val_perf['accuracy'], epoch)
    writer.add_scalar('F1-Score/train', train_perf['f1'], epoch)
    writer.add_scalar('F1-Score/valid', val_perf['f1'], epoch)

    # Track best performance, and save the model's state
    if val_loss < best_vloss:
        best_vloss = val_loss
        torch.save(model.state_dict(), 'cnn.pth')
        print('Saved best model to cnn.pth')
print("Done!")

# Evaluate on the Test Set
# Load the best model
model_best =  CNNBaseline().to(device)
model_best.load_state_dict(torch.load("cnn.pth"))
# Use the best model on the training set
train_loss, train_y_preds, train_y_trues= test(train_dl, model_best, loss_fn, device)
# Performance metrics on the training set
train_perf = {
    'accuracy':  multiclass_accuracy(train_y_preds, train_y_trues).item(),
    'f1': multiclass_f1_score(train_y_preds, train_y_trues).item(),
}
# Use the best model on the test set
test_loss, test_y_preds, test_y_trues = test(test_dl, model_best, loss_fn, device)
# Performance metrics
test_perf = {
    'accuracy': multiclass_accuracy(test_y_preds, test_y_trues).item(),
    'f1': multiclass_f1_score(test_y_preds, test_y_trues).item(),
}

train_mse = mean_squared_error(train_y_preds.float(), train_y_trues.float()).item()
test_mse  = mean_squared_error(test_y_preds.float(),  test_y_trues.float()).item()

print(f"Train: loss={train_loss:>8f}, acc={(100*train_perf['accuracy']):>0.1f}%, f1={(100*train_perf['f1']):>0.1f}%, mse={train_mse}")
print(f"Test: loss={test_loss:>8f}, acc={(100*test_perf['accuracy']):>0.1f}%, f1={(100*test_perf['f1']):>0.1f}%, mse={test_mse}")

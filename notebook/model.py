import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models

class CNNBaseline(nn.Module):

    def __init__(self, num_classes=9):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128), 
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

class CNNBaseline2(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32), #add batchnorm
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1), #add more layers
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1,1))
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3), #add dropout
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

class ResNetModel(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()

        # Load pretrained ResNet50
        self.model = models.resnet50(weights="DEFAULT")

        # Freeze early layers 
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze last layer block 
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # Replace final fully connected layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)

class MobileNetModel(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()

        # Load pretrained MobileNetV2
        self.model = models.mobilenet_v2(weights="DEFAULT")

        # Freeze all layers 
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze last feature block
        for param in self.model.features[-1].parameters():
            param.requires_grad = True

        # Replace classifier
        in_features = self.model.classifier[1].in_features

        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)
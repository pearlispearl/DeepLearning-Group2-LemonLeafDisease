import torch
from torchvision import transforms, datasets
from PIL import Image

from model import CNNBaseline, CNNBaseline2, ResNetModel, MobileNetModel

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

test_ds = datasets.ImageFolder("Lemon_Dataset/test")

class_names = test_ds.classes
print(class_names)

# Load image
img_path = "Lemon_Dataset/test/Deficiency Leaf/Deficiency_leaf00035_JPG.rf.698cf6ee15dbbfe259d5515bbada52e6.jpg"
image = Image.open(img_path).convert("RGB")

# Transforms
transform_cnn = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

transform_net = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])
def predict(model, transform, model_name, weight_path):
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()

    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(img)
        probs = torch.softmax(out, dim=1)
        conf, pred = torch.max(probs, 1)

    print(f"\n{model_name}")
    print(f"Prediction: {class_names[pred.item()]}")
    print(f"Confidence: {conf.item()*100:.2f}%")


# Test each model

predict(CNNBaseline(9), transform_cnn, "CNNBaseline", "best_model/cnn.pth")
predict(CNNBaseline2(9), transform_cnn, "CNNBaseline2", "best_model/cnn2.pth")
predict(ResNetModel(9), transform_net, "ResNet50", "best_model/Resnet.pth")
predict(MobileNetModel(9), transform_net,"MobileNetV2", "best_model/mobilenet.pth"
)
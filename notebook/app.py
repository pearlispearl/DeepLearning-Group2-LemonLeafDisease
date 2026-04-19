import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
# Import model classes defined in model.py
from model import ResNetModel, MobileNetModel, CLASS_NAMES

# 1. Device configuration (MPS for Mac, CUDA for Windows/Linux, or CPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 2. Function to load the model
@st.cache_resource
def load_prediction_model(model_type="ResNet50"):
    import os
    
    # 1. Get the directory of the current app.py file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Create the path to the best_model folder by moving up one level from current_dir
    root_dir = os.path.dirname(current_dir)
    
    if model_type == "ResNet50":
        model = ResNetModel(num_classes=9)
        # Use os.path.join to construct the path correctly for different OS (Windows/Linux)
        weight_path = os.path.join(root_dir, 'best_model', 'Resnet.pth')
    else:
        model = MobileNetModel(num_classes=9)
        weight_path = os.path.join(root_dir, 'best_model', 'mobilenet.pth')
        
    # 3. Load the file (Checking if the file exists before loading helps prevent errors)
    if not os.path.exists(weight_path):
        st.error(f"File not found at: {weight_path}")
        return None
        
    # 2. Load weights file
    checkpoint = torch.load(weight_path, map_location=device)
    
    # 3. Strip 'model.' prefix from state_dict keys
    new_state_dict = {}
    for k, v in checkpoint.items():
        if not k.startswith('model.'):
            new_state_dict[f'model.{k}'] = v
        else:
            new_state_dict[k] = v
    
    # 4. Load cleaned weights into the model
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model

# 3. Define preprocessing (must match training phase)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Streamlit UI Section ---
st.title("🍋Lemon Disease Detection System")
st.write("Upload a lemon leaf image for disease analysis")

# Select model for inference
selected_model_name = st.selectbox("Select Model Architecture", ["ResNet50", "MobileNetV2"])

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button("Start Analysis"):
        with st.spinner('AI is processing...'):
            # 1. Load the selected model
            model = load_prediction_model(selected_model_name)
            if model is None:
                st.stop()
                
            # 2. Prepare the image
            img_tensor = transform(image).unsqueeze(0).to(device)
            
            # 3. Perform prediction
            with torch.no_grad():
                outputs = model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                conf, pred = torch.max(probs, 1)
                
            # 4. Display results
            st.success("Analysis Complete!")
            result_class = CLASS_NAMES[pred.item()]
            confidence_score = conf.item() * 100
            
            col1, col2 = st.columns(2)
            col1.metric("Prediction", result_class)
            col2.metric("Confidence", f"{confidence_score:.2f}%")
            
            # Display confidence progress bar
            st.progress(conf.item())
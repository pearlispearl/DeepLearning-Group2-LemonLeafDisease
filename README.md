# 🍋 Lemon Leaf Disease Detection System
An end-to-end Deep Learning solution for classifying lemon leaf diseases using ResNet50 and MobileNetV2 architectures.

## 🚀 Quick Start
Follow these steps to set up the environment and launch the application.

### 1. Installation & Environment Setup
Install all required dependencies directly via your terminal:
```bash
pip install torch torchvision streamlit pillow numpy pandas matplotlib seaborn scikit-learn tensorboard
```

### 2. Project Structure
To ensure the system functions correctly, organize your files as follows:

* **`app.py`**: The main Streamlit web application.
* **`notebook/model.py`**: Neural Network architecture definitions (ResNet & MobileNet).
* **`best_model/`**: Folder containing pre-trained weights (`resnet.pth`, `mobilenet.pth`).
* **`runs/`**: Directory containing TensorBoard logs for training history.
* **`plot_curves.py`**: Utility script to generate Loss and Accuracy visualization graphs.
* **`eval_metrics.py`**: Script for detailed performance evaluation (Classification Report & Confusion Matrix).

---

## 3. Running the Demo App
Launch the web interface by executing the following command in your terminal:

```bash
streamlit run app.py
```

## 4. Generate Learning Curves
Execute this script to visualize the Training/Validation Loss and Accuracy from your TensorBoard logs:

```bash
python plot_curves.py
```

## 5. Run Performance Evaluation
Generate a detailed Classification Report and Confusion Matrix for the selected model:

```bash
python eval_metrics.py
```

### User Instructions:
1.  **Select Model**: Choose between **ResNet50** (Highest Accuracy) or **MobileNetV2** (High Efficiency).
2.  **Upload Image**: Upload a JPG, JPEG, or PNG image of a lemon leaf.
3.  **Analyze**: Click the **"Start Analysis"** button to view the prediction and confidence score.

---

## 📊 Model Evaluation Summary
We conducted four iterations to determine the optimal model for deployment:

| Iteration | Model Architecture | Accuracy | Macro F1-Score | Status |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Simple CNN (Baseline) | 62% | 0.49 | Underfitting |
| 2 | Improved CNN | 20% | 0.14 | Failed (Model Bias) |
| 3 | **ResNet50** | **90%** | **0.88** | **Best Accuracy** |
| 4 | **MobileNetV2** | **89%** | **0.87** | **Best for Deployment** |

> **Selection Logic:** While ResNet50 achieved the highest accuracy, **MobileNetV2** was selected for final deployment due to its superior inference speed and lightweight architecture, making it ideal for mobile applications.

---

## 🔍 Key Technical Responsibilities

* **Performance Metrics**: Calculated Accuracy, Precision, Recall, and F1-Score to ensure model stability across imbalanced classes.
* **Visual Analysis**: Generated Confusion Matrices and Learning Curves (Loss/Accuracy) to detect training behaviors like Overfitting.
* **Weight Optimization**: Implemented Prefix Stripping to resolve weight loading conflicts (e.g., removing `"model."` prefixes).
* **Error Handling**: Resolved Size Mismatch errors by standardizing Global Average Pooling across the architecture.
* **Real-world Testing**: Evaluated model robustness against "Out-of-Distribution" data, including leaves with physical damage and varying backgrounds.

## 🛠 Tech Stack
* **Framework**: PyTorch
* **Frontend**: Streamlit
* **Visualization**: TensorBoard, Matplotlib, Seaborn
* **Image Processing**: Pillow (PIL), Torchvision

## 🐳 (Bonus) Reproducibility with Docker
To run this project in a containerized environment (no local installation required):

### 1. Build the Docker Image
```bash
docker build -t lemon-disease-app .
import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def extract_scalars(log_dir):
    """Extracting Accuracy and Loss data from the 'runs' directory."""
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    tags = ea.Tags()['scalars']
    
    data = {}
    for tag in tags:
        events = ea.Scalars(tag)
        data[tag] = [e.value for e in events]
        data[f'{tag}_step'] = [e.step for e in events]
    return data

def plot_all_curves(runs_base_path):
    # Model Architectures Overview
    models = {
        "CNNBaseline": "Iteration 1: Baseline",
        "CNNBaseline2": "Iteration 2: Improved CNN",
        "ResNetModel": "Iteration 3: ResNet50",
        "MobileNetModel": "Iteration 4: MobileNetV2"
    }
    
    run_subdirs = [os.path.join(runs_base_path, d) for d in os.listdir(runs_base_path) 
                   if os.path.isdir(os.path.join(runs_base_path, d))]

    for model_key, display_name in models.items():
        # Find the latest run folder for the specific model
        model_runs = [d for d in run_subdirs if model_key in d]
        if not model_runs: continue
        latest_run = sorted(model_runs)[-1]
        
        data = extract_scalars(latest_run)
        if not data: continue

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 📉 Plotting Loss Curves
        if 'Loss/train' in data and 'Loss/valid' in data:
            ax1.plot(data['Loss/train_step'], data['Loss/train'], label='Train Loss', color='blue', marker='o')
            ax1.plot(data['Loss/valid_step'], data['Loss/valid'], label='Val Loss', color='orange', marker='s')
            ax1.set_title(f'Loss Curve - {display_name}')
            ax1.set_xlabel('Epochs')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)

        # 📈 Plotting Accuracy Curves
        if 'Accuracy/train' in data and 'Accuracy/valid' in data:
            ax2.plot(data['Accuracy/train_step'], data['Accuracy/train'], label='Train Acc', color='green', marker='o')
            ax2.plot(data['Accuracy/valid_step'], data['Accuracy/valid'], label='Val Acc', color='red', marker='s')
            ax2.set_title(f'Accuracy Curve - {display_name}')
            ax2.set_xlabel('Epochs')
            ax2.set_ylabel('Accuracy')
            ax2.legend()
            ax2.grid(True)

        plt.tight_layout()
        plt.savefig(f"learning_curve_{model_key}.png")
        print(f"✅ Generated: learning_curve_{model_key}.png")
        plt.show()

# Execute plotting (Point to the 'runs' folder in the project)
plot_all_curves('runs')
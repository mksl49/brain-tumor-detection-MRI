import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

def visualize_detections(image_path, boxes, labels, scores, class_names, 
                         threshold=0.3, save_path=None):
    """Visualize detections on image with confidence scores"""
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img_rgb)
    
    for box, label, score in zip(boxes, labels, scores):
        if score > threshold:
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            # Create rectangle
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2, edgecolor='lime', facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label
            label_text = f"{class_names[label]}: {score:.2f}"
            ax.text(
                x1, y1 - 5, label_text,
                fontsize=10, color='white',
                bbox=dict(facecolor='lime', alpha=0.8, edgecolor='none', pad=1)
            )
    
    ax.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_training_curves(log_dir, model_name, save_path=None):
    """Plot training curves from logs"""
    import pandas as pd
    import glob
    
    log_files = glob.glob(f"{log_dir}/*.csv")
    if not log_files:
        return
    
    df = pd.read_csv(log_files[0])
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot losses
    if 'train/loss' in df.columns:
        axes[0, 0].plot(df['epoch'], df['train/loss'], label='Train Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title(f'{model_name} - Training Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    if 'val/loss' in df.columns:
        axes[0, 1].plot(df['epoch'], df['val/loss'], label='Val Loss', color='orange')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title(f'{model_name} - Validation Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot metrics
    if 'metrics/mAP_0.5' in df.columns:
        axes[1, 0].plot(df['epoch'], df['metrics/mAP_0.5'], label='mAP@0.5', color='green')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('mAP')
        axes[1, 0].set_title(f'{model_name} - mAP@0.5')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    if 'metrics/mAP_0.5:0.95' in df.columns:
        axes[1, 1].plot(df['epoch'], df['metrics/mAP_0.5:0.95'], label='mAP@0.5:0.95', color='red')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('mAP')
        axes[1, 1].set_title(f'{model_name} - mAP@0.5:0.95')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def create_comparison_grid(image_path, predictions_dict, class_names, threshold=0.3):
    """Create grid comparing predictions from different models"""
    import cv2
    
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    n_models = len(predictions_dict)
    fig, axes = plt.subplots(1, n_models + 1, figsize=(5*(n_models + 1), 8))
    
    # Original image
    axes[0].imshow(img_rgb)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Each model's predictions
    for idx, (model_name, preds) in enumerate(predictions_dict.items(), 1):
        boxes, labels, scores = preds['boxes'], preds['labels'], preds['scores']
        
        axes[idx].imshow(img_rgb)
        for box, label, score in zip(boxes, labels, scores):
            if score > threshold:
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                
                rect = patches.Rectangle(
                    (x1, y1), width, height,
                    linewidth=2, edgecolor='lime', facecolor='none'
                )
                axes[idx].add_patch(rect)
                
                label_text = f"{class_names[label]}: {score:.2f}"
                axes[idx].text(
                    x1, y1 - 5, label_text,
                    fontsize=8, color='white',
                    bbox=dict(facecolor='lime', alpha=0.8, edgecolor='none', pad=1)
                )
        
        axes[idx].set_title(model_name)
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig
from ultralytics import YOLO
import wandb
import torch
import logging
import os
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_yolo():
    """
    Enhanced YOLOv8 training with medical-specific optimizations
    """
    # Initialize WandB
    run_name = f"yolov8m_{datetime.now().strftime('%Y%m%d_%H%M')}"
    
    try:
        wandb.init(
            project="mri-object-detection",
            name=run_name,
            config={
                "model": "yolov8m",
                "dataset": "mri_brain_tumor",
                "epochs": 70,
                "batch_size": 16,
                "image_size": 640
            },
            resume="allow"
        )
        
        # Save code artifacts
        try:
            wandb.save('*.py')
            wandb.save('configs/*.yaml')
        except:
            pass
            
    except Exception as e:
        logger.warning(f"WandB init failed: {e}. Continuing without WandB.")
        wandb.init(mode="disabled")
    
    logger.info("="*60)
    logger.info("YOLOv8 TRAINING - MEDICAL IMAGING CONFIG")
    logger.info("="*60)
    
    # Load pretrained model
    model = YOLO('yolov8m.pt')
    
    # Train with optimized hyperparameters
    results = model.train(
        # Data
        data='/home/likhon/zksl/lmn/mri/project/configs/yolo_data.yaml',
        
        # Training duration
        epochs=70,
        patience=20,  # Early stopping
        
        # Batch and image settings
        batch=16,
        imgsz=512,
        
        # Optimization
        optimizer='AdamW',
        lr0=0.001,      # Initial learning rate
        lrf=0.01,       # Final learning rate (1% of lr0)
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # Medical-specific augmentations
        # Geometric
        flipud=0.0,     # No vertical flip (brain anatomy)
        fliplr=0.5,     # Horizontal flip OK (brain symmetry)
        degrees=10,     # Small rotations (patient head tilt)
        translate=0.1,  # Small translations
        scale=0.2,      # Scale augmentation
        shear=5,        # Minimal shear
        perspective=0.0,  # No perspective (maintain anatomy)
        
        # Photometric (MRI-specific)
        hsv_h=0.01,     # Minimal hue (preserve anatomy)
        hsv_s=0.3,      # Moderate saturation
        hsv_v=0.2,      # Moderate brightness
        
        # Mosaic and mixup
        mosaic=1.0,     # Enable mosaic augmentation
        mixup=0.1,      # Light mixup
        copy_paste=0.1, # Copy-paste augmentation
        
        # Loss weights (adjusted for medical imaging)
        box=7.5,        # Box loss weight
        cls=0.5,        # Classification loss weight
        dfl=1.5,        # Distribution focal loss weight
        
        # Other settings
        cos_lr=True,    # Cosine learning rate scheduler
        close_mosaic=10,  # Disable mosaic in last 10 epochs
        
        # Validation
        val=True,
        save=True,
        save_period=5,  # Save checkpoint every 5 epochs
        
        # Output
        project='runs/detect',
        name='mri_yolo_v2',
        exist_ok=False,
        
        # Performance
        amp=True,       # Automatic Mixed Precision
        device=0 if torch.cuda.is_available() else 'cpu',
        workers=8,
        
        # Logging
        verbose=True,
        plots=True,
    )
    
    # Log best metrics to WandB
    logger.info("="*60)
    logger.info("TRAINING COMPLETE - FINAL METRICS")
    logger.info("="*60)
    
    try:
        # Extract metrics using the correct attribute access
        best_metrics = {
            'final/mAP50-95': float(results.box.map),
            'final/mAP50': float(results.box.map50),
            'final/mAP75': float(results.box.map75),
            'final/precision': float(results.box.mp),
            'final/recall': float(results.box.mr),
        }
        
        # Calculate F1
        p, r = best_metrics['final/precision'], best_metrics['final/recall']
        best_metrics['final/f1'] = 2 * (p * r) / (p + r + 1e-16)
        
        # Log to WandB (BEFORE wandb.finish())
        if wandb.run is not None:
            wandb.log(best_metrics)
        
        # Print metrics
        for metric, value in best_metrics.items():
            logger.info(f"{metric}: {value:.4f}")
            
    except Exception as e:
        logger.warning(f"Failed to log final metrics: {e}")
    
    logger.info("="*60)
    
    # Save best model to WandB
    try:
        best_model_path = Path('runs/detect/mri_yolo_v2/weights/best.pt')
        if best_model_path.exists() and wandb.run is not None:
            wandb.save(str(best_model_path))
            logger.info(f"✓ Model saved to WandB: {best_model_path}")
    except Exception as e:
        logger.warning(f"Failed to save model to WandB: {e}")
    
    # Finish WandB (AFTER all logging)
    if wandb.run is not None:
        wandb.finish()
    
    logger.info("✓ Training complete!")
    
    return results

if __name__ == '__main__':
    train_yolo()
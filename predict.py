#!/usr/bin/env python3
"""
Streamlined Prediction Script
Run inference on images using trained models
"""

import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import logging
from pathlib import Path
import sys

# Model imports
from transformers import DetrForObjectDetection, DetrImageProcessor
from ultralytics import YOLO
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==========================================
# Configuration
# ==========================================
CLASSES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
COLORS = ['red', 'blue', 'green', 'orange']

# Model paths - Scripts run from project root
PROJECT_ROOT = '/home/likhon/zksl/lmn/mri/project'

# YOLO: Check multiple possible locations
YOLO_PATHS = [
    f'{PROJECT_ROOT}/src/runs/detect/mri_yolo_v2/weights/best.pt',
    f'{PROJECT_ROOT}/runs/detect/mri_yolo_v2/weights/best.pt',
]

# DETR: Check multiple possible locations  
DETR_PATHS = [
    f'{PROJECT_ROOT}/src/checkpoints/detr/final_model.pth',
    f'{PROJECT_ROOT}/checkpoints/detr/final_model.pth',
]

# Faster R-CNN: Dynamic lookup
FRCNN_DIR = f'{PROJECT_ROOT}/output'

# ==========================================
# Helper Functions
# ==========================================
def find_model_path(paths_list):
    """Find first existing path from list"""
    for path in paths_list:
        if Path(path).exists():
            logger.info(f"Found model: {path}")
            return path
    return None

# ==========================================
# DETR Predictor
# ==========================================
class DETRPredictor:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info("Loading DETR model...")
        
        detr_model = find_model_path(DETR_PATHS)
        if not detr_model:
            raise FileNotFoundError(f"DETR model not found in any location")
        
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            num_labels=4,
            ignore_mismatched_sizes=True
        )
        self.model.load_state_dict(torch.load(detr_model, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        logger.info("✓ DETR loaded")
    
    def predict(self, image_path):
        """Run prediction on image"""
        image = Image.open(image_path).convert('RGB')
        
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs, threshold=self.threshold, target_sizes=target_sizes
        )[0]
        
        # Convert to numpy
        predictions = []
        for score, label, box in zip(results['scores'], results['labels'], results['boxes']):
            predictions.append({
                'bbox': box.cpu().numpy(),
                'label': int(label.cpu()),
                'score': float(score.cpu()),
                'class_name': CLASSES[int(label.cpu())]
            })
        
        return predictions, image

# ==========================================
# Faster R-CNN Predictor
# ==========================================
class FasterRCNNPredictor:
    def __init__(self, output_dir, threshold=0.5):
        self.threshold = threshold
        
        # Find latest model
        output_dirs = list(Path(output_dir).glob('faster_rcnn_*'))
        if not output_dirs:
            raise FileNotFoundError(f"No Faster R-CNN models found in {output_dir}")
        
        model_dir = sorted(output_dirs)[-1]
        logger.info(f"Loading Faster R-CNN from: {model_dir}")
        
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
        
        model_path = model_dir / "model_final.pth"
        if not model_path.exists():
            model_path = model_dir / "best_model.pth"
        
        cfg.MODEL.WEIGHTS = str(model_path)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
        
        # CRITICAL: Match training anchor configuration
        cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8, 16, 32, 64, 128, 256, 512]]
        cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]
        
        self.predictor = DefaultPredictor(cfg)
        logger.info("✓ Faster R-CNN loaded")
    
    def predict(self, image_path):
        """Run prediction on image"""
        import cv2
        image = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        outputs = self.predictor(image)
        instances = outputs["instances"].to("cpu")
        
        predictions = []
        for box, score, cls in zip(
            instances.pred_boxes.tensor.numpy(),
            instances.scores.numpy(),
            instances.pred_classes.numpy()
        ):
            predictions.append({
                'bbox': box,
                'label': int(cls),
                'score': float(score),
                'class_name': CLASSES[int(cls)]
            })
        
        return predictions, Image.fromarray(image_rgb)

# ==========================================
# YOLO Predictor
# ==========================================
class YOLOPredictor:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        logger.info("Loading YOLO model...")
        
        yolo_model = find_model_path(YOLO_PATHS)
        if not yolo_model:
            raise FileNotFoundError(f"YOLO model not found in any location")
        
        self.model = YOLO(yolo_model)
        logger.info("✓ YOLO loaded")
    
    def predict(self, image_path):
        """Run prediction on image"""
        results = self.model(image_path, conf=self.threshold, verbose=False)[0]
        
        image = Image.open(image_path).convert('RGB')
        
        predictions = []
        if results.boxes is not None:
            for box in results.boxes:
                bbox = box.xyxy[0].cpu().numpy()
                predictions.append({
                    'bbox': bbox,
                    'label': int(box.cls.cpu()),
                    'score': float(box.conf.cpu()),
                    'class_name': CLASSES[int(box.cls.cpu())]
                })
        
        return predictions, image

# ==========================================
# Visualization
# ==========================================
def visualize_predictions(image, predictions, model_name, save_path=None):
    """Visualize predictions on image"""
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    
    for pred in predictions:
        bbox = pred['bbox']
        x1, y1, x2, y2 = bbox
        
        color = COLORS[pred['label']]
        rect = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        
        label_text = f"{pred['class_name']}: {pred['score']:.2f}"
        ax.text(
            x1, y1-5, label_text,
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.7),
            fontsize=10, color='white', fontweight='bold'
        )
    
    ax.set_title(f'{model_name} Predictions ({len(predictions)} detections)', 
                 fontsize=14, fontweight='bold')
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"✓ Saved visualization: {save_path}")
    
    plt.close()

# ==========================================
# Main Prediction
# ==========================================
def predict(image_path, models=['yolo'], threshold=0.5, save_dir='predictions'):
    """Run prediction with selected models"""
    image_path = Path(image_path)
    if not image_path.exists():
        logger.error(f"Image not found: {image_path}")
        return
    
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    logger.info("="*60)
    logger.info(f"Running predictions on: {image_path.name}")
    logger.info(f"Models: {', '.join(models)}")
    logger.info(f"Threshold: {threshold}")
    logger.info("="*60)
    
    results = {}
    
    for model_name in models:
        logger.info(f"\n{'='*60}")
        logger.info(f"Model: {model_name.upper()}")
        logger.info("="*60)
        
        try:
            if model_name == 'yolo':
                predictor = YOLOPredictor(threshold)
            
            elif model_name == 'detr':
                predictor = DETRPredictor(threshold)
            
            elif model_name == 'faster_rcnn':
                if not Path(FRCNN_DIR).exists():
                    logger.error(f"Faster R-CNN dir not found: {FRCNN_DIR}")
                    continue
                predictor = FasterRCNNPredictor(FRCNN_DIR, threshold)
            
            else:
                logger.error(f"Unknown model: {model_name}")
                continue
            
            # Run prediction
            predictions, image = predictor.predict(image_path)
            
            # Log results
            logger.info(f"Detections: {len(predictions)}")
            for i, pred in enumerate(predictions, 1):
                logger.info(f"  {i}. {pred['class_name']}: {pred['score']:.3f}")
            
            # Visualize
            save_path = save_dir / f"{image_path.stem}_{model_name}.png"
            visualize_predictions(image, predictions, model_name.upper(), save_path)
            
            results[model_name] = predictions
            
        except Exception as e:
            logger.error(f"Error with {model_name}: {e}")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("PREDICTION SUMMARY")
    logger.info("="*60)
    for model, preds in results.items():
        logger.info(f"{model.upper()}: {len(preds)} detections")
    logger.info(f"\nVisualizations saved in: {save_dir}")
    logger.info("="*60)
    
    return results

# ==========================================
# CLI
# ==========================================

### suggested path
# gg="/home/likhon/zksl/lmn/mri/project/data/yolo_merged/val/images/gg (22).jpg"
# tr-pi="/home/likhon/zksl/lmn/mri/project/data/yolo_merged/val/images/Tr-pi_1318.jpg"
# tr-no="/home/likhon/zksl/lmn/mri/project/data/yolo_merged/val/images/Tr-no_0328.jpg"
# tr-me="/home/likhon/zksl/lmn/mri/project/data/yolo_merged/val/images/Tr-meTr_0004.jpg"
# tr-gl="/home/likhon/zksl/lmn/mri/project/data/yolo_merged/val/images/Tr-gl_1295.jpg"
# m1="/home/likhon/zksl/lmn/mri/project/data/yolo_merged/val/images/m1(134).jpg"
# neutral="/home/likhon/zksl/lmn/mri/project/data/yolo_merged/val/images/image(173).jpg"

def main():
    parser = argparse.ArgumentParser(description='Run inference on MRI images')
    parser.add_argument('image', type=str, help='Path to input image')
    parser.add_argument(
        '--models', 
        nargs='+', 
        choices=['yolo', 'detr', 'faster_rcnn', 'all'],
        default=['yolo'],
        help='Models to use (default: yolo)'
    )
    parser.add_argument(
        '--threshold', 
        type=float, 
        default=0.5,
        help='Confidence threshold (default: 0.5)'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default='/home/likhon/zksl/lmn/mri/project/predictions',
        help='Directory to save predictions (default: predictions)'
    )
    
    args = parser.parse_args()
    
    # Handle 'all' option
    if 'all' in args.models:
        models = ['yolo', 'detr', 'faster_rcnn']
    else:
        models = args.models
    
    # Run prediction
    predict(args.image, models, args.threshold, args.save_dir)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
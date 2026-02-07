#!/usr/bin/env python3
"""
Streamlined Model Evaluation Script
Evaluates trained models using COCO metrics
"""

import json
import torch
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image
import sys

# COCO evaluation
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

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
GT_PATH = '/home/likhon/zksl/lmn/mri/project/data/coco/val/annotations.json'
VAL_IMAGES = '/home/likhon/zksl/lmn/mri/project/data/coco/val/images'
RESULTS_DIR = Path('/home/likhon/zksl/lmn/mri/project/results')
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# Model paths - Update these based on where you ran training from
PROJECT_ROOT = '/home/likhon/zksl/lmn/mri/project'

# YOLO: Check both /src/runs and /project/runs
YOLO_PATHS = [
    f'{PROJECT_ROOT}/src/runs/detect/mri_yolo_v2/weights/best.pt',
    f'{PROJECT_ROOT}/runs/detect/mri_yolo_v2/weights/best.pt',
    f'{PROJECT_ROOT}/src/runs/detect/train/weights/best.pt',  # Fallback
]

# DETR: Check both /src/checkpoints and /project/checkpoints  
DETR_PATHS = [
    f'{PROJECT_ROOT}/checkpoints/detr/final_model.pth',
    f'{PROJECT_ROOT}/src/checkpoints/detr/final_model.pth',
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
# COCO Evaluation
# ==========================================
def evaluate_coco(gt_path, pred_path, model_name):
    """Run COCO evaluation and return metrics"""
    logger.info(f"Evaluating {model_name} with COCO metrics...")
    
    try:
        coco_gt = COCO(gt_path)
        coco_dt = coco_gt.loadRes(pred_path)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        metrics = {
            'model': model_name,
            'mAP_50-95': float(coco_eval.stats[0]),
            'mAP_50': float(coco_eval.stats[1]),
            'mAP_75': float(coco_eval.stats[2]),
            'mAP_small': float(coco_eval.stats[3]),
            'mAP_medium': float(coco_eval.stats[4]),
            'mAP_large': float(coco_eval.stats[5]),
        }
        
        logger.info(f"✓ {model_name} mAP@50-95: {metrics['mAP_50-95']:.4f}")
        return metrics
        
    except Exception as e:
        logger.error(f"COCO evaluation failed for {model_name}: {e}")
        return None

# ==========================================
# DETR Evaluator
# ==========================================
def evaluate_detr():
    """Evaluate DETR model"""
    detr_model = find_model_path(DETR_PATHS)
    if not detr_model:
        logger.warning(f"DETR model not found in any of these locations:")
        for p in DETR_PATHS:
            logger.warning(f"  - {p}")
        return None
    
    logger.info("Loading DETR model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=4,
        ignore_mismatched_sizes=True
    )
    model.load_state_dict(torch.load(detr_model, map_location=device))
    model.to(device)
    model.eval()
    
    # Load images
    with open(GT_PATH, 'r') as f:
        coco_data = json.load(f)
    
    predictions = []
    logger.info("Generating DETR predictions...")
    
    for img_info in tqdm(coco_data['images'], desc="DETR inference"):
        img_path = Path(VAL_IMAGES) / img_info['file_name']
        if not img_path.exists():
            continue
        
        image = Image.open(img_path).convert('RGB')
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        target_sizes = torch.tensor([image.size[::-1]]).to(device)
        results = processor.post_process_object_detection(
            outputs, threshold=0.05, target_sizes=target_sizes
        )[0]
        
        for score, label, box in zip(results['scores'], results['labels'], results['boxes']):
            x1, y1, x2, y2 = box.cpu().numpy()
            predictions.append({
                'image_id': img_info['id'],
                'category_id': int(label.cpu()) + 1,
                'bbox': [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                'score': float(score.cpu())
            })
    
    # Save predictions
    pred_path = RESULTS_DIR / 'detr_predictions.json'
    with open(pred_path, 'w') as f:
        json.dump(predictions, f)
    
    return evaluate_coco(GT_PATH, str(pred_path), 'DETR')

# ==========================================
# Faster R-CNN Evaluator
# ==========================================
def evaluate_faster_rcnn():
    """Evaluate Faster R-CNN model"""
    output_dirs = list(Path(FRCNN_DIR).glob('faster_rcnn_*'))
    if not output_dirs:
        logger.warning(f"No Faster R-CNN models found in {FRCNN_DIR}")
        return None
    
    model_dir = sorted(output_dirs)[-1]
    logger.info(f"Loading Faster R-CNN from: {model_dir}")
    
    # Setup config - MUST match training configuration
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
    
    model_path = model_dir / "model_final.pth"
    if not model_path.exists():
        model_path = model_dir / "best_model.pth"
    
    if not model_path.exists():
        logger.error(f"No model found in {model_dir}")
        return None
    
    cfg.MODEL.WEIGHTS = str(model_path)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    
    # CRITICAL: Match training anchor configuration from train_faster_rcnn.py
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8, 16, 32, 64, 128, 256, 512]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]
    
    # Match training RPN config
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 2000
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 1000
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 1000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 500
    
    predictor = DefaultPredictor(cfg)
    
    # Load images
    with open(GT_PATH, 'r') as f:
        coco_data = json.load(f)
    
    predictions = []
    logger.info("Generating Faster R-CNN predictions...")
    
    import cv2
    for img_info in tqdm(coco_data['images'], desc="Faster R-CNN inference"):
        img_path = Path(VAL_IMAGES) / img_info['file_name']
        if not img_path.exists():
            continue
        
        image = cv2.imread(str(img_path))
        outputs = predictor(image)
        
        instances = outputs["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        classes = instances.pred_classes.numpy()
        
        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = box
            predictions.append({
                'image_id': img_info['id'],
                'category_id': int(cls) + 1,
                'bbox': [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                'score': float(score)
            })
    
    # Save predictions
    pred_path = RESULTS_DIR / 'frcnn_predictions.json'
    with open(pred_path, 'w') as f:
        json.dump(predictions, f)
    
    return evaluate_coco(GT_PATH, str(pred_path), 'Faster R-CNN')

# ==========================================
# YOLO Evaluator
# ==========================================
def evaluate_yolo():
    """Evaluate YOLO model"""
    yolo_model = find_model_path(YOLO_PATHS)
    if not yolo_model:
        logger.warning(f"YOLO model not found in any of these locations:")
        for p in YOLO_PATHS:
            logger.warning(f"  - {p}")
        return None
    
    logger.info("Loading YOLO model...")
    model = YOLO(yolo_model)
    
    logger.info("Running YOLO validation...")
    results = model.val(
        data='/home/likhon/zksl/lmn/mri/project/configs/yolo_data.yaml',
        split='val',
        batch=8,
        verbose=False
    )
    
    metrics = {
        'model': 'YOLOv8',
        'mAP_50-95': float(results.box.map),
        'mAP_50': float(results.box.map50),
        'mAP_75': float(results.box.map75),
        'precision': float(results.box.mp),
        'recall': float(results.box.mr),
    }
    
    # Calculate F1
    p, r = metrics['precision'], metrics['recall']
    metrics['f1'] = 2 * (p * r) / (p + r + 1e-16)
    
    logger.info(f"✓ YOLOv8 mAP@50-95: {metrics['mAP_50-95']:.4f}")
    return metrics

# ==========================================
# Main
# ==========================================
def main():
    """Run evaluation for all models"""
    logger.info("="*60)
    logger.info("MODEL EVALUATION")
    logger.info("="*60)
    
    results = []
    
    # Evaluate each model
    for evaluator, name in [
        (evaluate_yolo, "YOLOv8"),
        (evaluate_faster_rcnn, "Faster R-CNN"),
        (evaluate_detr, "DETR")
    ]:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating {name}")
        logger.info("="*60)
        
        try:
            metrics = evaluator()
            if metrics:
                results.append(metrics)
        except Exception as e:
            logger.error(f"{name} evaluation failed: {e}")
    
    # Save results
    if results:
        import pandas as pd
        df = pd.DataFrame(results)
        
        output_path = RESULTS_DIR / 'evaluation_metrics.csv'
        df.to_csv(output_path, index=False)
        
        logger.info(f"\n{'='*60}")
        logger.info("EVALUATION COMPLETE")
        logger.info("="*60)
        logger.info(f"\n{df.to_string(index=False)}")
        logger.info(f"\n✓ Results saved to: {output_path}")
    else:
        logger.error("No models were successfully evaluated!")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())





import os
import sys
import torch
import numpy as np
import logging
from datetime import datetime
import warnings
import json

# Detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor, HookBase
from detectron2.config import get_cfg
from detectron2.data import (
    DatasetCatalog, 
    MetadataCatalog, 
    build_detection_train_loader,
    build_detection_test_loader
)
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.logger import setup_logger
from detectron2.data import transforms as T
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.samplers import RepeatFactorTrainingSampler
from detectron2.utils.events import get_event_storage

import wandb

warnings.filterwarnings("ignore", category=FutureWarning)
setup_logger()
logger = logging.getLogger(__name__)

# Initialize WandB
run_name = f"faster_rcnn_{datetime.now().strftime('%Y%m%d_%H%M')}"
wandb.init(project="mri-faster-rcnn", name=run_name)

# ==========================================
# Custom Trainer with Best Practices
# ==========================================
class MRITrainer(DefaultTrainer):
    """
    Enhanced Trainer for Medical MRI Detection
    """
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)
    
    @classmethod
    def build_train_loader(cls, cfg):
        """
        Custom data loader with:
        - Medical-specific augmentations
        - Class imbalance handling via RepeatFactorTrainingSampler
        """
        # Medical augmentations
        augs = [
            # Resize
            T.ResizeShortestEdge(
                short_edge_length=cfg.INPUT.MIN_SIZE_TRAIN,
                max_size=cfg.INPUT.MAX_SIZE_TRAIN,
                sample_style="choice"
            ),
            
            # Geometric
            T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
            T.RandomRotation(angle=[-15, 15], sample_style="range"),
            T.RandomCrop(crop_type="relative_range", crop_size=(0.8, 1.0)),
            
            # Photometric (MRI-specific)
            T.RandomBrightness(intensity_min=0.8, intensity_max=1.2),
            T.RandomContrast(intensity_min=0.8, intensity_max=1.2),
            T.RandomLighting(scale=0.5),
            T.RandomSaturation(intensity_min=0.8, intensity_max=1.2),
        ]
        
        mapper = DatasetMapper(cfg, is_train=True, augmentations=augs)
        
        # Handle class imbalance
        dataset_dicts = DatasetCatalog.get(cfg.DATASETS.TRAIN[0])
        
        # Calculate repeat factors based on category frequency
        repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
            dataset_dicts, 
            repeat_thresh=0.001  # Oversample rare classes
        )
        
        sampler = RepeatFactorTrainingSampler(repeat_factors)
        
        return build_detection_train_loader(
            cfg, 
            mapper=mapper, 
            sampler=sampler,
            total_batch_size=cfg.SOLVER.IMS_PER_BATCH
        )
    
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """Build test loader with minimal augmentation"""
        return build_detection_test_loader(cfg, dataset_name)
    
    def build_hooks(self):
        hooks = super().build_hooks()
        
        # Custom WandB logging hook
        class WandBLogHook(HookBase):
            def __init__(self, cfg):
                self.cfg = cfg
                
            def after_step(self):
                if self.trainer.iter % 100 == 0:
                    try:
                        storage = get_event_storage()
                        
                        # Get latest metrics
                        metrics = {}
                        for k in storage.histories().keys():
                            v = storage.history(k).latest()
                            metrics[k] = v
                        
                        # Log to WandB
                        wandb.log({
                            "iteration": self.trainer.iter,
                            **metrics
                        })
                        
                    except Exception as e:
                        logger.warning(f"WandB logging failed: {e}")
            
            def after_train(self):
                # Log final model
                try:
                    model_path = os.path.join(self.cfg.OUTPUT_DIR, "model_final.pth")
                    if os.path.exists(model_path):
                        wandb.save(model_path)
                except Exception as e:
                    logger.warning(f"Failed to save model to WandB: {e}")
        
        # Insert before the last hook (usually writer)
        hooks.insert(-1, WandBLogHook(self.cfg))
        
        # Evaluation hook for mAP logging
        class EvalLogHook(HookBase):
            def __init__(self, eval_period, cfg):
                self.eval_period = eval_period
                self.cfg = cfg
            
            def after_step(self):
                if (self.trainer.iter + 1) % self.eval_period == 0:
                    # Evaluation happens automatically via DefaultTrainer
                    # We just log the results to WandB
                    try:
                        storage = get_event_storage()
                        
                        # Check for validation metrics
                        eval_metrics = {}
                        for k in storage.histories().keys():
                            if 'bbox' in k or 'segm' in k:
                                eval_metrics[f"val/{k}"] = storage.history(k).latest()
                        
                        if eval_metrics:
                            wandb.log(eval_metrics)
                            
                    except Exception as e:
                        logger.debug(f"Eval logging: {e}")
        
        if self.cfg.TEST.EVAL_PERIOD > 0:
            hooks.insert(-1, EvalLogHook(self.cfg.TEST.EVAL_PERIOD, self.cfg))
        
        return hooks

# ==========================================
# Dataset Setup
# ==========================================
def setup_datasets():
    """Register COCO datasets"""
    # Train
    register_coco_instances(
        "mri_train",
        {},
        "/home/likhon/zksl/lmn/mri/project/data/coco/train/annotations.json",
        "/home/likhon/zksl/lmn/mri/project/data/coco/train/images"
    )
    
    # Validation
    register_coco_instances(
        "mri_val",
        {},
        "/home/likhon/zksl/lmn/mri/project/data/coco/val/annotations.json",
        "/home/likhon/zksl/lmn/mri/project/data/coco/val/images"
    )
    
    # Metadata
    classes = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
    MetadataCatalog.get("mri_train").set(thing_classes=classes)
    MetadataCatalog.get("mri_val").set(thing_classes=classes)
    
    logger.info("✓ Datasets registered")

# ==========================================
# Configuration
# ==========================================
def get_config():
    cfg = get_cfg()
    
    # Base model - ResNet-101-FPN (Strong backbone)
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
    )
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
    )
    
    # Datasets
    cfg.DATASETS.TRAIN = ("mri_train",)
    cfg.DATASETS.TEST = ("mri_val",)
    cfg.DATALOADER.NUM_WORKERS = 4
    
    # Model
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
    
    # Input preprocessing
    cfg.INPUT.MIN_SIZE_TRAIN = (512, 544, 576, 608, 640)
    cfg.INPUT.MAX_SIZE_TRAIN = 1024
    cfg.INPUT.MIN_SIZE_TEST = 512
    cfg.INPUT.MAX_SIZE_TEST = 1024
    cfg.INPUT.FORMAT = "BGR"
    
    # CRITICAL: Multi-scale anchors for small tumor detection
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8, 16, 32, 64, 128, 256, 512]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]
    
    # RPN configuration
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 2000
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 1000
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 1000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 500
    cfg.MODEL.RPN.NMS_THRESH = 0.7
    cfg.MODEL.RPN.POSITIVE_FRACTION = 0.5
    cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 256
    
    # ROI Head
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.25
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
    
    # Solver
    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"
    cfg.SOLVER.MAX_ITER = 24000
    cfg.SOLVER.WARMUP_ITERS = 800
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
    cfg.SOLVER.WARMUP_METHOD = "linear"
    cfg.SOLVER.WEIGHT_DECAY = 0.0001
    cfg.SOLVER.MOMENTUM = 0.9
    cfg.SOLVER.CHECKPOINT_PERIOD =3000
    cfg.SOLVER.AMP.ENABLED = True
    
    # Evaluation
    cfg.TEST.EVAL_PERIOD = 3000
    
    # Output
    output_dir = f"/home/likhon/zksl/lmn/mri/project/output/faster_rcnn_{datetime.now().strftime('%Y%m%d_%H%M')}"
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    return cfg

# ==========================================
# Post-training Evaluation
# ==========================================
def evaluate_model(cfg, trainer):
    """Comprehensive evaluation with COCO metrics"""
    logger.info("="*60)
    logger.info("RUNNING FINAL EVALUATION")
    logger.info("="*60)
    
    # Build evaluator
    evaluator = COCOEvaluator("mri_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "mri_val")
    
    # Run inference
    results = inference_on_dataset(trainer.model, val_loader, evaluator)
    
    # Extract metrics
    if 'bbox' in results:
        bbox_metrics = results['bbox']
        
        metrics = {
            'final/mAP': bbox_metrics.get('AP', 0),
            'final/mAP50': bbox_metrics.get('AP50', 0),
            'final/mAP75': bbox_metrics.get('AP75', 0),
            'final/mAP_small': bbox_metrics.get('APs', 0),
            'final/mAP_medium': bbox_metrics.get('APm', 0),
            'final/mAP_large': bbox_metrics.get('APl', 0),
        }
        
        # Log to WandB
        wandb.log(metrics)
        
        # Save to JSON
        metrics_path = os.path.join(cfg.OUTPUT_DIR, 'final_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Print results
        logger.info("\n" + "="*60)
        logger.info("FINAL EVALUATION RESULTS")
        logger.info("="*60)
        for k, v in metrics.items():
            logger.info(f"{k}: {v:.4f}")
        logger.info("="*60)
    
    return results

# ==========================================
# Main Training
# ==========================================
def train():
    logger.info("="*60)
    logger.info("FASTER R-CNN TRAINING - PRODUCTION CONFIG")
    logger.info("="*60)
    
    # Setup
    setup_datasets()
    cfg = get_config()
    
    # Log config to WandB
    config_dict = {}
    for key in cfg.keys():
        try:
            config_dict[key] = cfg[key]
        except:
            pass
    wandb.config.update(config_dict)
    
    # Create trainer
    trainer = MRITrainer(cfg)
    trainer.resume_or_load(resume=False)
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Final evaluation
    evaluate_model(cfg, trainer)
    
    # Save best model
    final_model_path = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    best_model_path = os.path.join(cfg.OUTPUT_DIR, "best_model.pth")
    
    if os.path.exists(final_model_path):
        torch.save(trainer.model.state_dict(), best_model_path)
        wandb.save(best_model_path)
        logger.info(f"✓ Model saved to {best_model_path}")
    
    logger.info("="*60)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*60)
    
    wandb.finish()

if __name__ == "__main__":
    train()
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
import os
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from transformers import DetrForObjectDetection, DetrImageProcessor
from PIL import Image
import json
from pathlib import Path
import logging
from typing import Dict, List
import albumentations as A
from torchmetrics.detection.mean_ap import MeanAveragePrecision

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# Constants
# ==========================================
NUM_CLASSES = 4
CLASSES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
ID2LABEL = {0: 'glioma', 1: 'meningioma', 2: 'no_tumor', 3: 'pituitary'}
LABEL2ID = {'glioma': 0, 'meningioma': 1, 'no_tumor': 2, 'pituitary': 3}
COCO_ID_TO_LABEL = {1: 0, 2: 1, 3: 2, 4: 3}

# Class weights (inverse frequency normalization)
CLASS_COUNTS = torch.tensor([1349., 1482., 711., 1775.])
CLASS_WEIGHTS = (1.0 / CLASS_COUNTS) / (1.0 / CLASS_COUNTS).sum() * len(CLASS_COUNTS)

# ==========================================
# Dataset with Advanced Augmentations
# ==========================================
class MRIDataset(torch.utils.data.Dataset):
    """Enhanced MRI dataset with medical-specific augmentations"""
    
    def __init__(self, img_dir, annotation_file, processor, train=True):
        self.img_dir = Path(img_dir)
        self.processor = processor
        self.is_train = train
        
        # Load COCO annotations
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)
        
        self.images = {img['id']: img for img in coco_data['images']}
        self.annotations = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)
        
        self.image_ids = list(self.images.keys())
        
        # Medical-grade augmentations (FIXED: correct parameter names)
        if self.is_train:
            self.transform = A.Compose([
                A.Resize(512, 512),
                
                # Geometric augmentations
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5, border_mode=0),
                A.Affine(
                    translate_percent={'x': (-0.05, 0.05), 'y': (-0.05, 0.05)},
                    scale=(0.9, 1.1),
                    rotate=(-10, 10),
                    p=0.3,
                    mode=0
                ),
                
                # Photometric augmentations (MRI-specific)
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, 
                    contrast_limit=0.2, 
                    p=0.5
                ),
                A.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=0.3),
                A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                
                # Advanced medical augmentations
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.4),
                A.ElasticTransform(
                    alpha=1, 
                    sigma=50,
                    p=0.2,
                    border_mode=0
                ),
                A.CoarseDropout(
                    max_holes=8,
                    max_height=32,
                    max_width=32,
                    fill_value=0,
                    p=0.3
                ),
            ], bbox_params=A.BboxParams(
                format='coco', 
                label_fields=['category_ids'],
                min_visibility=0.3
            ))
        else:
            self.transform = A.Compose([
                A.Resize(512, 512)
            ], bbox_params=A.BboxParams(
                format='coco', 
                label_fields=['category_ids']
            ))

    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_info = self.images[image_id]
        
        # Load image
        img_path = self.img_dir / img_info['file_name']
        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image)
        
        # Get annotations
        anns = self.annotations.get(image_id, [])
        
        bboxes = []
        category_ids = []
        
        for ann in anns:
            coco_cat_id = ann['category_id']
            label_id = COCO_ID_TO_LABEL.get(coco_cat_id)
            
            if label_id is not None:
                bboxes.append(ann['bbox'])
                category_ids.append(label_id)
        
        # Apply augmentations
        try:
            transformed = self.transform(
                image=image_np, 
                bboxes=bboxes, 
                category_ids=category_ids
            )
        except Exception as e:
            logger.warning(f"Augmentation failed for image {image_id}: {e}. Using original.")
            transformed = {
                'image': image_np,
                'bboxes': bboxes,
                'category_ids': category_ids
            }
        
        # Convert to PIL for processor
        image_pil = Image.fromarray(transformed['image'].astype('uint8'))
        
        # CRITICAL FIX: Reconstruct annotations in the format DETR processor expects
        # Processor expects dict with 'image_id' and 'annotations' keys
        annotation_list = []
        for bbox, cat_id in zip(transformed['bboxes'], transformed['category_ids']):
            annotation_list.append({
                'category_id': cat_id,
                'bbox': bbox,
                'area': bbox[2] * bbox[3],
                'iscrowd': 0
            })
        
        annotations_dict = {
            'image_id': image_id,
            'annotations': annotation_list
        }
        
        # Process with DETR processor
        encoding = self.processor(
            images=image_pil,
            annotations=annotations_dict if len(annotation_list) > 0 else None,
            return_tensors="pt"
        )
        
        pixel_values = encoding["pixel_values"].squeeze(0)
        
        # CRITICAL FIX: Handle case when no labels are returned (empty annotations)
        if "labels" in encoding and len(encoding["labels"]) > 0:
            labels = encoding["labels"][0]
        else:
            # Create empty labels for images without annotations
            labels = {
                'class_labels': torch.tensor([], dtype=torch.long),
                'boxes': torch.tensor([]).reshape(0, 4),
                'area': torch.tensor([]),
                'iscrowd': torch.tensor([], dtype=torch.long),
                'orig_size': torch.tensor([512, 512]),
                'size': torch.tensor([512, 512])
            }
        
        return pixel_values, labels

# ==========================================
# Lightning Module with mAP Calculation
# ==========================================
class DETRLightningModule(pl.LightningModule):
    def __init__(
        self, 
        lr=1e-4, 
        lr_backbone=1e-5, 
        weight_decay=1e-4,
        num_classes=NUM_CLASSES,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Load DETR model
        self.model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        
        self.model.config.id2label = id2label
        self.model.config.label2id = label2id
        
        # Class weights for handling imbalance
        self.register_buffer('class_weights', CLASS_WEIGHTS)
        
        # Metrics
        self.val_map = MeanAveragePrecision(
            box_format='xyxy',
            iou_type='bbox'
        )
        
        # Storage for validation predictions
        self.val_predictions = []
        self.val_targets = []
        
    def forward(self, pixel_values, pixel_mask=None):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
    
    def common_step(self, batch, batch_idx, stage='train'):
        pixel_values = batch[0]
        labels = batch[1]
        
        outputs = self.model(pixel_values=pixel_values, labels=labels)
        
        # Apply class weights to classification loss
        loss_dict = outputs.loss_dict
        weighted_ce_loss = loss_dict['loss_ce'] * self.class_weights.mean()
        
        # Total loss with weighted classification
        total_loss = weighted_ce_loss + loss_dict['loss_bbox'] + loss_dict['loss_giou']
        
        # Log individual losses
        self.log(f"{stage}_loss", total_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(f"{stage}_loss_ce", loss_dict['loss_ce'], on_step=True, on_epoch=True)
        self.log(f"{stage}_loss_bbox", loss_dict['loss_bbox'], on_step=True, on_epoch=True)
        self.log(f"{stage}_loss_giou", loss_dict['loss_giou'], on_step=True, on_epoch=True)
        
        return total_loss, outputs
    
    def training_step(self, batch, batch_idx):
        loss, _ = self.common_step(batch, batch_idx, 'train')
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, outputs = self.common_step(batch, batch_idx, 'val')
        
        # Get predictions for mAP calculation
        pixel_values = batch[0]
        
        # Post-process predictions
        target_sizes = torch.tensor([[512, 512]] * pixel_values.shape[0]).to(self.device)
        
        # Use processor for post-processing
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        results = processor.post_process_object_detection(
            outputs,
            threshold=0.05,
            target_sizes=target_sizes
        )
        
        # Convert to torchmetrics format
        for i, result in enumerate(results):
            pred_boxes = result['boxes']
            pred_scores = result['scores']
            pred_labels = result['labels']
            
            if len(pred_boxes) > 0:
                self.val_predictions.append({
                    'boxes': pred_boxes,
                    'scores': pred_scores,
                    'labels': pred_labels
                })
            else:
                # Empty prediction
                self.val_predictions.append({
                    'boxes': torch.zeros((0, 4)).to(self.device),
                    'scores': torch.zeros(0).to(self.device),
                    'labels': torch.zeros(0, dtype=torch.long).to(self.device)
                })
            
            # Ground truth
            gt_labels = batch[1][i]
            if 'boxes' in gt_labels and len(gt_labels['boxes']) > 0:
                self.val_targets.append({
                    'boxes': gt_labels['boxes'],
                    'labels': gt_labels['class_labels']
                })
            else:
                # Empty ground truth
                self.val_targets.append({
                    'boxes': torch.zeros((0, 4)).to(self.device),
                    'labels': torch.zeros(0, dtype=torch.long).to(self.device)
                })
        
        return loss
    
    def on_validation_epoch_end(self):
        # Calculate mAP
        if len(self.val_predictions) > 0 and len(self.val_targets) > 0:
            try:
                self.val_map.update(self.val_predictions, self.val_targets)
                map_dict = self.val_map.compute()
                
                self.log("val_map", map_dict['map'], prog_bar=True)
                self.log("val_map_50", map_dict['map_50'])
                self.log("val_map_75", map_dict['map_75'])
                self.log("val_map_small", map_dict['map_small'])
                self.log("val_map_medium", map_dict['map_medium'])
                self.log("val_map_large", map_dict['map_large'])
                
                logger.info(f"Validation mAP: {map_dict['map']:.4f}")
                
                self.val_map.reset()
            except Exception as e:
                logger.warning(f"mAP calculation failed: {e}")
        
        # Clear storage
        self.val_predictions.clear()
        self.val_targets.clear()
    
    def configure_optimizers(self):
        # Separate backbone and other parameters
        param_dicts = [
            {
                "params": [p for n, p in self.named_parameters() 
                          if "backbone" not in n and p.requires_grad],
                "lr": self.hparams.lr,
            },
            {
                "params": [p for n, p in self.named_parameters() 
                          if "backbone" in n and p.requires_grad],
                "lr": self.hparams.lr_backbone,
            },
        ]
        
        optimizer = torch.optim.AdamW(
            param_dicts,
            weight_decay=self.hparams.weight_decay
        )
        
        # Warmup + Cosine Annealing
        from torch.optim.lr_scheduler import OneCycleLR
        
        scheduler = OneCycleLR(
            optimizer,
            max_lr=[self.hparams.lr, self.hparams.lr_backbone],
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1,  # 10% warmup
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=1000.0
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            }
        }

# ==========================================
# Collate Function
# ==========================================
def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    pixel_values = torch.stack(pixel_values)
    labels = [item[1] for item in batch]
    return pixel_values, labels

# ==========================================
# Main Training
# ==========================================
def train_detr():
    # Processor
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    
    # Datasets
    train_dataset = MRIDataset(
        img_dir='/home/likhon/zksl/lmn/mri/project/data/coco/train/images',
        annotation_file='/home/likhon/zksl/lmn/mri/project/data/coco/train/annotations.json',
        processor=processor,
        train=True
    )
    
    val_dataset = MRIDataset(
        img_dir='/home/likhon/zksl/lmn/mri/project/data/coco/val/images',
        annotation_file='/home/likhon/zksl/lmn/mri/project/data/coco/val/annotations.json',
        processor=processor,
        train=False
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    
    # DataLoaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Model
    model = DETRLightningModule(
        lr=1e-4,
        lr_backbone=1e-5,
        weight_decay=1e-4
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_map',
        dirpath='checkpoints/detr',
        filename='detr-mri-{epoch:02d}-{val_map:.3f}',
        save_top_k=3,
        mode='max',
        save_last=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_map',
        patience=30,
        mode='max',
        verbose=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # WandB Logger
    wandb_logger = WandbLogger(
        project="mri-detr-detection",
        name=f"detr-resnet50-improved",
        log_model="all"
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=60,  # DETR needs more epochs
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        gradient_clip_val=0.1,
        accumulate_grad_batches=4,
        precision='16-mixed',
        log_every_n_steps=10,
        val_check_interval=0.5,
        deterministic=False
    )
    
    # Train
    logger.info("Starting DETR training with improved configuration...")
    trainer.fit(model, train_dataloader, val_dataloader)
    
    # Save final model
    save_path = 'checkpoints/detr/final_model.pth'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.model.state_dict(), save_path)
    logger.info(f"Training complete! Model saved to {save_path}")
    
    return model

if __name__ == '__main__':
    train_detr()
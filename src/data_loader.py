import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2  

def get_transforms(train=True):
    transforms = []
    if train:
        transforms.extend([
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8,8), p=0.5),  
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.GaussNoise(var_limit=(10, 50), p=0.2),
        ])
    transforms.extend([
        A.Normalize(mean=0.0, std=1.0),  # Standard for MRI
        ToTensorV2()
    ])
    return A.Compose(transforms, bbox_params=A.BboxParams(format='coco' if not train else 'yolo', label_fields=['category_id']))
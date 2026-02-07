import os
import shutil
import json
import logging
from tqdm import tqdm
from glob import glob
import cv2
from pathlib import Path  # Added for better path handling


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Class mapping
CLASSES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
NUM_CLASSES = len(CLASSES)
class_to_id = {cls: idx for idx, cls in enumerate(CLASSES)}

def get_all_image_files(images_dir):
    """Get all image files from directory with various extensions."""
    # Define common image extensions (case insensitive)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp'}
    
    img_files = []
    for ext in image_extensions:
        # Get both lowercase and uppercase extensions
        for pattern in [f'*{ext}', f'*{ext.upper()}']:
            img_files.extend(glob(os.path.join(images_dir, pattern)))
    
    # Remove duplicates (if any) and sort
    img_files = sorted(set(img_files))
    return img_files

def get_label_path_for_image(img_path, labels_dir):
    """Get corresponding label path for an image."""
    img_name = os.path.basename(img_path)
    # Remove any image extension and add .txt
    base_name = os.path.splitext(img_name)[0]
    return os.path.join(labels_dir, f"{base_name}.txt")

def merge_yolo_folders(raw_dir, merged_dir):
    """Merge class subfolders into single images/labels for train/val."""
    for split in ['train', 'val']:
        img_dest = os.path.join(merged_dir, split, 'images')
        lbl_dest = os.path.join(merged_dir, split, 'labels')
        os.makedirs(img_dest, exist_ok=True)
        os.makedirs(lbl_dest, exist_ok=True)
        
        src_dir = os.path.join(raw_dir, split)
        class_folders = [f for f in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, f))]
        
        for cls in tqdm(class_folders, desc=f"Merging {split} folders"):
            img_src = os.path.join(src_dir, cls, 'images')
            lbl_src = os.path.join(src_dir, cls, 'labels')
            if os.path.exists(img_src):
                # Get all image files from source
                img_files = get_all_image_files(img_src)
                
                # Copy image files
                for img_file in img_files:
                    dest_file = os.path.join(img_dest, os.path.basename(img_file))
                    shutil.copy(img_file, dest_file)
                
                # Copy corresponding label files
                for img_file in img_files:
                    img_name = os.path.basename(img_file)
                    base_name = os.path.splitext(img_name)[0]
                    label_file = os.path.join(lbl_src, f"{base_name}.txt")
                    
                    if os.path.exists(label_file):
                        dest_label = os.path.join(lbl_dest, f"{base_name}.txt")
                        shutil.copy(label_file, dest_label)
                    else:
                        logger.warning(f"Label file not found for image: {img_name}")
        
        # Count actual images copied
        img_count = len(get_all_image_files(img_dest))
        logger.info(f"Merged {split}: {img_count} images")

def create_yolo_yaml(merged_dir, yaml_path):
    """Create data.yaml for YOLO."""
    data = {
        'train': os.path.join(merged_dir, 'train/images'),
        'val': os.path.join(merged_dir, 'val/images'),
        'nc': NUM_CLASSES,
        'names': CLASSES
    }
    with open(yaml_path, 'w') as f:
        json.dump(data, f, indent=2)  # YAML is JSON-like, but can use yaml.dump if import yaml
    logger.info(f"Created {yaml_path}")

def yolo_to_coco(merged_yolo_dir, coco_dir, split):
    """Convert YOLO labels to COCO JSON for a split."""
    images_dir = os.path.join(merged_yolo_dir, split, 'images')
    labels_dir = os.path.join(merged_yolo_dir, split, 'labels')
    coco_images_dir = os.path.join(coco_dir, split, 'images')
    os.makedirs(coco_images_dir, exist_ok=True)
    
    images = []
    annotations = []
    ann_id = 1
    img_id = 1
    
    # Get all image files
    img_files = get_all_image_files(images_dir)
    
    for img_path in tqdm(img_files, desc=f"Converting {split} to COCO"):
        img_name = os.path.basename(img_path)
        shutil.copy(img_path, os.path.join(coco_images_dir, img_name))
        
        # Read image dimensions
        img_cv = cv2.imread(img_path)
        if img_cv is None:
            logger.error(f"Could not read image: {img_path}")
            continue
            
        h, w, _ = img_cv.shape
        
        images.append({
            'id': img_id,
            'width': w,
            'height': h,
            'file_name': img_name,
            'license': 0,
            'flickr_url': '',
            'coco_url': '',
            'date_captured': 0
        })
        
        # Get corresponding label file
        lbl_path = get_label_path_for_image(img_path, labels_dir)
        
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        parts = line.split()
                        if len(parts) != 5:
                            logger.warning(f"Invalid label format in {lbl_path}: {line}")
                            continue
                            
                        cls_id, x_c, y_c, bw, bh = map(float, parts)
                        
                        # Convert YOLO format to COCO bbox
                        x = (x_c - bw/2) * w
                        y = (y_c - bh/2) * h
                        width = bw * w
                        height = bh * h
                        
                        # Ensure bbox is within image bounds
                        x = max(0, min(x, w - 1))
                        y = max(0, min(y, h - 1))
                        width = max(0, min(width, w - x))
                        height = max(0, min(height, h - y))
                        
                        annotations.append({
                            'id': ann_id,
                            'image_id': img_id,
                            'category_id': int(cls_id) + 1,  # COCO starts at 1
                            'segmentation': [],
                            'area': width * height,
                            'bbox': [x, y, width, height],
                            'iscrowd': 0
                        })
                        ann_id += 1
                    except ValueError as e:
                        logger.error(f"Error processing label in {lbl_path}: {line} - {e}")
        else:
            logger.debug(f"No label file found for image: {img_name}")
            
        img_id += 1
    
    coco_json = {
        'images': images,
        'annotations': annotations,
        'categories': [{'id': i+1, 'name': cls} for i, cls in enumerate(CLASSES)]
    }
    
    json_path = os.path.join(coco_dir, split, 'annotations.json')
    with open(json_path, 'w') as f:
        json.dump(coco_json, f, indent=2)
    
    logger.info(f"Created {json_path} with {len(images)} images, {len(annotations)} annotations")
    return len(images), len(annotations)

if __name__ == '__main__':
    raw_dir = '/home/likhon/zksl/lmn/mri/project/data/raw'
    merged_yolo_dir = '/home/likhon/zksl/lmn/mri/project/data/yolo_merged'
    coco_dir = '/home/likhon/zksl/lmn/mri/project/data/coco'
    yaml_path = '/home/likhon/zksl/lmn/mri/project/configs/yolo_data.yaml'
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
    
    # Run the pipeline
    logger.info("Starting data processing pipeline...")
    merge_yolo_folders(raw_dir, merged_yolo_dir)
    create_yolo_yaml(merged_yolo_dir, yaml_path)
    
    total_images = 0
    total_annotations = 0
    for split in ['train', 'val']:
        images, annotations = yolo_to_coco(merged_yolo_dir, coco_dir, split)
        total_images += images
        total_annotations += annotations
    
    logger.info(f"Processing complete!")
    logger.info(f"Total images processed: {total_images}")
    logger.info(f"Total annotations processed: {total_annotations}")
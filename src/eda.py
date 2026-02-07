import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm
import logging
from glob import glob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Class mapping (same as in your main script)
CLASSES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

def get_all_label_files(labels_dir):
    """Get all label files from directory."""
    return glob(os.path.join(labels_dir, '*.txt'))

def get_image_from_label(label_path, images_dir):
    """Get corresponding image file for a label file, checking multiple extensions."""
    base_name = os.path.splitext(os.path.basename(label_path))[0]
    
    # Check for various image extensions
    image_extensions = ['.jpg', '.jpeg', '.JPG', '.JPEG', 
                        '.png', '.PNG', '.bmp', '.BMP',
                        '.tiff', '.tif', '.TIFF', '.TIF',
                        '.gif', '.GIF', '.webp', '.WEBP']
    
    for ext in image_extensions:
        image_path = os.path.join(images_dir, f"{base_name}{ext}")
        if os.path.exists(image_path):
            return image_path
    
    # Try case-insensitive search with glob if exact match fails
    for pattern in [f'{base_name}.*', f'{base_name.lower()}.*', f'{base_name.upper()}.*']:
        possible_images = glob(os.path.join(images_dir, pattern))
        if possible_images:
            # Filter for image extensions
            img_exts = {ext.lower() for ext in image_extensions}
            for img in possible_images:
                if os.path.splitext(img)[1].lower() in img_exts:
                    return img
    
    return None

def count_classes(merged_dir):
    """Count images and instances per class for train/val."""
    counts = {'split': [], 'class': [], 'image_count': [], 'instance_count': [], 'total_images': []}
    
    for split in ['train', 'val']:
        images_dir = os.path.join(merged_dir, split, 'images')
        labels_dir = os.path.join(merged_dir, split, 'labels')
        
        # Ensure directories exist
        if not os.path.exists(images_dir):
            logger.warning(f"Images directory not found: {images_dir}")
            continue
        if not os.path.exists(labels_dir):
            logger.warning(f"Labels directory not found: {labels_dir}")
            continue
        
        class_img = defaultdict(int)  # Count of images containing each class
        class_inst = defaultdict(int)  # Count of instances for each class
        processed_images = set()  # Track processed images
        images_without_labels = set()  # Track images without label files
        total_images = 0
        
        # First, get all label files
        lbl_files = get_all_label_files(labels_dir)
        logger.info(f"Found {len(lbl_files)} label files in {split}")
        
        # Count from label files
        for lbl_path in tqdm(lbl_files, desc=f"Counting {split} from labels"):
            base_name = os.path.splitext(os.path.basename(lbl_path))[0]
            
            # Check if corresponding image exists
            img_path = get_image_from_label(lbl_path, images_dir)
            if img_path:
                processed_images.add(os.path.basename(img_path))
                total_images += 1
            else:
                logger.debug(f"No corresponding image found for label: {lbl_path}")
            
            # Read and process the label file
            try:
                with open(lbl_path, 'r') as f:
                    lines = f.readlines()
                
                if lines:
                    imgs_per_class = set()  # For unique images per class
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        
                        parts = line.split()
                        if len(parts) < 5:
                            logger.warning(f"Invalid label format in {lbl_path}: {line}")
                            continue
                        
                        cls_id = int(float(parts[0]))  # Handle cases where class ID might be float
                        if 0 <= cls_id < len(CLASSES):
                            cls_name = CLASSES[cls_id]
                            class_inst[cls_name] += 1
                            imgs_per_class.add(cls_name)
                        else:
                            logger.warning(f"Invalid class ID {cls_id} in {lbl_path}")
                    
                    for cls in imgs_per_class:
                        class_img[cls] += 1
                else:
                    # Empty label file - count as 'no_tumor'
                    class_img['no_tumor'] += 1
                    class_inst['no_tumor'] += 0  # Add 0 instances
                    
            except Exception as e:
                logger.error(f"Error processing label file {lbl_path}: {e}")
        
        # Count images without label files (should be none in YOLO format)
        # Get all image files in the directory
        image_extensions = ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG', 
                           '*.png', '*.PNG', '*.bmp', '*.BMP',
                           '*.tiff', '*.tif', '*.TIFF', '*.TIF',
                           '*.gif', '*.GIF', '*.webp', '*.WEBP']
        
        all_images = set()
        for pattern in image_extensions:
            for img in glob(os.path.join(images_dir, pattern)):
                all_images.add(os.path.basename(img))
        
        images_without_labels = all_images - processed_images
        
        if images_without_labels:
            logger.warning(f"Found {len(images_without_labels)} images without label files in {split}")
            # Count these as 'no_tumor' images
            class_img['no_tumor'] += len(images_without_labels)
            class_inst['no_tumor'] += 0
            total_images += len(images_without_labels)
        
        # Record counts for each class
        for cls in CLASSES:
            counts['split'].append(split)
            counts['class'].append(cls)
            counts['image_count'].append(class_img.get(cls, 0))
            counts['instance_count'].append(class_inst.get(cls, 0))
            counts['total_images'].append(total_images)
    
    df = pd.DataFrame(counts)
    
    # Save the results
    os.makedirs('data/eda', exist_ok=True)
    df.to_csv('data/eda/class_counts.csv', index=False)
    
    # Also save summary statistics
    summary = df.groupby('split').agg({
        'total_images': 'first',
        'image_count': 'sum',
        'instance_count': 'sum'
    }).reset_index()
    summary.to_csv('data/eda/summary_stats.csv', index=False)
    
    logger.info("Saved class_counts.csv and summary_stats.csv")
    return df

def plot_eda(df):
    """Plot bar graphs for EDA."""
    os.makedirs('data/eda', exist_ok=True)
    sns.set(style='whitegrid')
    
    # Plot 1: Instance Count per Class
    plt.figure(figsize=(14, 8))
    ax1 = sns.barplot(data=df, x='class', y='instance_count', hue='split')
    plt.title('Instance Count per Class', fontsize=16, fontweight='bold')
    plt.xlabel('Class', fontsize=14)
    plt.ylabel('Number of Instances', fontsize=14)
    
    # Add value labels on bars
    for container in ax1.containers:
        ax1.bar_label(container, fmt='%d', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('data/eda/instance_dist.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Image Count per Class
    plt.figure(figsize=(14, 8))
    ax2 = sns.barplot(data=df, x='class', y='image_count', hue='split')
    plt.title('Image Count per Class', fontsize=16, fontweight='bold')
    plt.xlabel('Class', fontsize=14)
    plt.ylabel('Number of Images', fontsize=14)
    
    # Add value labels on bars
    for container in ax2.containers:
        ax2.bar_label(container, fmt='%d', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('data/eda/image_dist.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Split comparison pie charts
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    for idx, split in enumerate(['train', 'val']):
        split_df = df[df['split'] == split]
        axes[idx].pie(split_df['instance_count'], labels=split_df['class'], 
                     autopct='%1.1f%%', startangle=90)
        axes[idx].set_title(f'{split.capitalize()} Set - Instance Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('data/eda/pie_charts.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Create a detailed summary table visualization
    plt.figure(figsize=(12, 4))
    plt.axis('tight')
    plt.axis('off')
    
    # Create summary table
    summary_data = []
    for split in ['train', 'val']:
        split_df = df[df['split'] == split]
        total_imgs = split_df['total_images'].iloc[0]
        total_instances = split_df['instance_count'].sum()
        
        row_data = [split.capitalize(), total_imgs, total_instances]
        for cls in CLASSES:
            cls_data = split_df[split_df['class'] == cls]
            if not cls_data.empty:
                imgs = cls_data['image_count'].iloc[0]
                inst = cls_data['instance_count'].iloc[0]
                row_data.extend([f"{imgs} ({inst})"])
            else:
                row_data.extend(["0 (0)"])
        
        summary_data.append(row_data)
    
    # Create table
    columns = ['Split', 'Total Images', 'Total Instances'] + CLASSES
    table = plt.table(cellText=summary_data, colLabels=columns, 
                     cellLoc='center', loc='center',
                     colColours=['lightblue']*len(columns))
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    plt.title('Dataset Summary Statistics', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('data/eda/summary_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Saved EDA plots: instance_dist.png, image_dist.png, pie_charts.png, summary_table.png")

if __name__ == '__main__':
    merged_dir = '/home/likhon/zksl/lmn/mri/project/data/yolo_merged'
    
    # Check if merged directory exists
    if not os.path.exists(merged_dir):
        logger.error(f"Merged directory not found: {merged_dir}")
        logger.info("Please run the data preparation script first.")
    else:
        logger.info("Starting EDA analysis...")
        df = count_classes(merged_dir)
        
        # Print summary statistics
        print("\n" + "="*60)
        print("DATASET SUMMARY")
        print("="*60)
        
        for split in ['train', 'val']:
            split_df = df[df['split'] == split]
            if not split_df.empty:
                total_imgs = split_df['total_images'].iloc[0]
                total_instances = split_df['instance_count'].sum()
                print(f"\n{split.upper()} SET:")
                print(f"  Total Images: {total_imgs}")
                print(f"  Total Instances: {total_instances}")
                print(f"  Classes Distribution:")
                for cls in CLASSES:
                    cls_data = split_df[split_df['class'] == cls]
                    if not cls_data.empty:
                        imgs = cls_data['image_count'].iloc[0]
                        inst = cls_data['instance_count'].iloc[0]
                        print(f"    {cls}: {imgs} images, {inst} instances")
        
        print("\n" + "="*60)
        print("Generating visualizations...")
        plot_eda(df)
        logger.info("EDA analysis completed successfully!")
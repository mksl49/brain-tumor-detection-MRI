#!/usr/bin/env python3
"""
Minimal Pipeline for MRI Tumor Detection
Runs everything in sequence
"""

import os
import sys
import subprocess
import time
import argparse

def run_step(name, command, required=True):
    """Run a single step with nice formatting"""
    print(f"\n{'='*60}")
    print(f"STEP: {name}")
    print(f"{'='*60}")
    print(f"Command: {command}")
    
    start_time = time.time()
    
    try:
        # Run the command
        result = subprocess.run(command, shell=True, check=True)
        elapsed = time.time() - start_time
        print(f"✓ {name} completed in {elapsed:.1f}s")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {name} failed with error: {e}")
        if required:
            print("Stopping pipeline due to required step failure.")
            sys.exit(1)
        return False
    except KeyboardInterrupt:
        print(f"\n⚠ {name} interrupted by user")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Run MRI Detection Pipeline')
    parser.add_argument('--skip-data', action='store_true', help='Skip data preparation')
    parser.add_argument('--skip-eda', action='store_true', help='Skip EDA')
    parser.add_argument('--models', nargs='+', default=['yolo', 'faster_rcnn', 'detr'],
                       help='Models to train (yolo, faster_rcnn, detr)')
    parser.add_argument('--skip-eval', action='store_true', help='Skip evaluation')
    parser.add_argument('--skip-compare', action='store_true', help='Skip comparison')
    parser.add_argument('--src-dir', type=str, default='/home/likhon/zksl/lmn/mri/project/src',
                       help='Directory containing source files')
    args = parser.parse_args()
    
    # Change to src directory
    original_dir = os.getcwd()
    try:
        os.chdir(args.src_dir)
        print(f"Changed to directory: {args.src_dir}")
    except FileNotFoundError:
        print(f"Error: Directory not found: {args.src_dir}")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("MRI TUMOR DETECTION PIPELINE")
    print("="*60)
    print(f"Running from: {os.getcwd()}")
    
    # Track overall time
    pipeline_start = time.time()
    
    # 1. Data Preparation
    if not args.skip_data:
        run_step("Data Preparation", "python prepare_data.py")
    
    # 2. EDA
    if not args.skip_eda:
        run_step("Exploratory Data Analysis", "python eda.py", required=False)
    
    # 3. Model Training - Check which files actually exist
    model_map = {
        'yolo': 'YOLOv8',
        'faster_rcnn': 'Faster R-CNN', 
        'detr': 'DETR'
    }
    
    # Check which training files exist
    existing_files = os.listdir('.')
    training_commands = {}
    
    # Check for training files
    if 'train_yolo.py' in existing_files:
        training_commands['yolo'] = 'python train_yolo.py'
    elif 'train.py' in existing_files:  # Try alternative name
        training_commands['yolo'] = 'python train.py'
    
    if 'train_faster_rcnn.py' in existing_files:
        training_commands['faster_rcnn'] = 'python train_faster_rcnn.py'
    
    if 'train_detr.py' in existing_files:
        training_commands['detr'] = 'python train_detr.py'
    
    # Train only requested models that exist
    for model in args.models:
        if model in model_map:
            if model in training_commands:
                run_step(f"Training {model_map[model]}", training_commands[model])
            else:
                print(f"⚠ Training file for {model_map[model]} not found. Looking for:")
                print(f"  - train_{model}.py or train.py")
                print(f"  Available files: {[f for f in existing_files if 'train' in f]}")
                if model == 'yolo':  # yolo is usually required
                    print("Stopping pipeline.")
                    sys.exit(1)
        else:
            print(f"⚠ Unknown model: {model}")
    
    # 4. Evaluation
    if not args.skip_eval:
        if 'evaluate.py' in existing_files:
            run_step("Model Evaluation", "python evaluate.py", required=False)
        else:
            print("⚠ evaluate.py not found, skipping evaluation")
    
    # 5. Comparison
    if not args.skip_compare:
        if 'compare.py' in existing_files:
            run_step("Model Comparison", "python compare.py", required=False)
        else:
            print("⚠ compare.py not found, skipping comparison")
    
    # Change back to original directory
    os.chdir(original_dir)
    
    # Summary
    pipeline_elapsed = time.time() - pipeline_start
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE!")
    print(f"{'='*60}")
    print(f"Total time: {pipeline_elapsed:.1f}s ({pipeline_elapsed/60:.1f} minutes)")
    print("\nOutputs created in:")
    print("  - results/ (evaluation metrics)")
    print("  - output/ (model checkpoints)")
    print("  - checkpoints/ (saved models)")
    print("\nTo visualize results:")
    print("  - Check results/comparison_table.md")
    print("  - View results/*.png for charts")
    print("="*60)

if __name__ == "__main__":
    main()
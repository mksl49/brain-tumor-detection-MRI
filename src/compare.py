#!/usr/bin/env python3
"""
Streamlined Model Comparison Script
Compares evaluation results and generates visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==========================================
# Configuration
# ==========================================
RESULTS_DIR = Path('/home/likhon/zksl/lmn/mri/project/results')
METRICS_FILE = RESULTS_DIR / 'evaluation_metrics.csv'

# ==========================================
# Load Results
# ==========================================
def load_results():
    """Load evaluation metrics"""
    if not METRICS_FILE.exists():
        logger.error(f"Metrics file not found: {METRICS_FILE}")
        logger.info("Please run evaluate.py first!")
        return None
    
    df = pd.read_csv(METRICS_FILE)
    logger.info(f"Loaded metrics for {len(df)} models")
    return df

# ==========================================
# Visualization
# ==========================================
def create_comparison_plots(df):
    """Generate comparison plots"""
    logger.info("Creating comparison plots...")
    
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(16, 10))
    
    # 1. mAP Comparison
    plt.subplot(2, 3, 1)
    ax = sns.barplot(data=df, x='model', y='mAP_50-95', palette='viridis')
    plt.title('mAP@0.5:0.95 Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('mAP@0.5:0.95')
    plt.xlabel('')
    plt.ylim(0, 1.0)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f')
    
    # 2. mAP@50 vs mAP@75
    plt.subplot(2, 3, 2)
    x = np.arange(len(df))
    width = 0.35
    plt.bar(x - width/2, df['mAP_50'], width, label='mAP@0.5', alpha=0.8)
    plt.bar(x + width/2, df['mAP_75'], width, label='mAP@0.75', alpha=0.8)
    plt.xlabel('Model')
    plt.ylabel('mAP')
    plt.title('mAP@0.5 vs mAP@0.75', fontsize=14, fontweight='bold')
    plt.xticks(x, df['model'])
    plt.legend()
    plt.ylim(0, 1.0)
    
    # 3. Performance by object size
    plt.subplot(2, 3, 3)
    if all(col in df.columns for col in ['mAP_small', 'mAP_medium', 'mAP_large']):
        x = np.arange(len(df))
        width = 0.25
        plt.bar(x - width, df['mAP_small'], width, label='Small', alpha=0.8)
        plt.bar(x, df['mAP_medium'], width, label='Medium', alpha=0.8)
        plt.bar(x + width, df['mAP_large'], width, label='Large', alpha=0.8)
        plt.xlabel('Model')
        plt.ylabel('mAP')
        plt.title('Performance by Object Size', fontsize=14, fontweight='bold')
        plt.xticks(x, df['model'])
        plt.legend()
    
    # 4. Precision vs Recall
    if 'precision' in df.columns and 'recall' in df.columns:
        plt.subplot(2, 3, 4)
        for i, row in df.iterrows():
            plt.scatter(row['recall'], row['precision'], s=200, alpha=0.6, label=row['model'])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Trade-off', fontsize=14, fontweight='bold')
        plt.xlim(0, 1.0)
        plt.ylim(0, 1.0)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 5. F1 Score
    if 'f1' in df.columns:
        plt.subplot(2, 3, 5)
        ax = sns.barplot(data=df, x='model', y='f1', palette='mako')
        plt.title('F1-Score Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('F1-Score')
        plt.xlabel('')
        plt.ylim(0, 1.0)
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f')
    
    # 6. Multi-metric radar
    plt.subplot(2, 3, 6)
    metrics_cols = ['mAP_50-95', 'mAP_50', 'mAP_75']
    if 'precision' in df.columns:
        metrics_cols.append('precision')
    if 'recall' in df.columns:
        metrics_cols.append('recall')
    
    angles = np.linspace(0, 2*np.pi, len(metrics_cols), endpoint=False).tolist()
    angles += angles[:1]
    
    ax = plt.subplot(2, 3, 6, projection='polar')
    for _, row in df.iterrows():
        values = [row[m] for m in metrics_cols]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=row['model'])
        ax.fill(angles, values, alpha=0.15)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_cols, size=8)
    ax.set_ylim(0, 1.0)
    plt.title('Multi-Metric Comparison', fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    
    plot_path = RESULTS_DIR / 'model_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved plot: {plot_path}")
    plt.close()

# ==========================================
# Generate Report
# ==========================================
def generate_report(df):
    """Generate markdown report"""
    logger.info("Generating report...")
    
    report_path = RESULTS_DIR / 'comparison_report.md'
    
    with open(report_path, 'w') as f:
        f.write("# MRI Brain Tumor Detection - Model Comparison\n\n")
        
        # Summary table
        f.write("## Performance Metrics\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n")
        
        # Best models
        f.write("## Best Models by Category\n\n")
        
        best_map = df.loc[df['mAP_50-95'].idxmax()]
        f.write(f"### 🏆 Best Overall Accuracy\n")
        f.write(f"**{best_map['model']}** - mAP@0.5:0.95: {best_map['mAP_50-95']:.4f}\n\n")
        
        if 'f1' in df.columns:
            best_f1 = df.loc[df['f1'].idxmax()]
            f.write(f"### ⚖️ Best Balance (F1-Score)\n")
            f.write(f"**{best_f1['model']}** - F1: {best_f1['f1']:.4f}\n\n")
        
        # Recommendations
        f.write("## Recommendations\n\n")
        f.write("### For Production Deployment:\n")
        f.write(f"- **Highest Accuracy**: Use {best_map['model']} for maximum detection performance\n")
        
        if 'precision' in df.columns and 'recall' in df.columns:
            high_precision = df.loc[df['precision'].idxmax()]
            high_recall = df.loc[df['recall'].idxmax()]
            f.write(f"- **Minimize False Positives**: Use {high_precision['model']} (Precision: {high_precision['precision']:.4f})\n")
            f.write(f"- **Minimize False Negatives**: Use {high_recall['model']} (Recall: {high_recall['recall']:.4f})\n")
        
        f.write("\n---\n*Generated automatically by compare.py*\n")
    
    logger.info(f"✓ Saved report: {report_path}")

# ==========================================
# Print Summary
# ==========================================
def print_summary(df):
    """Print comparison summary"""
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80)
    print(f"\n{df.to_string(index=False)}")
    print("\n" + "="*80)
    
    # Best models
    print("\nBEST MODELS:")
    print("-"*80)
    
    best_map = df.loc[df['mAP_50-95'].idxmax()]
    print(f"🏆 Best Accuracy:  {best_map['model']:<15} mAP@0.5:0.95 = {best_map['mAP_50-95']:.4f}")
    
    if 'f1' in df.columns:
        best_f1 = df.loc[df['f1'].idxmax()]
        print(f"⚖️  Best Balance:   {best_f1['model']:<15} F1-Score = {best_f1['f1']:.4f}")
    
    if 'precision' in df.columns:
        best_precision = df.loc[df['precision'].idxmax()]
        print(f"🎯 Best Precision: {best_precision['model']:<15} Precision = {best_precision['precision']:.4f}")
    
    if 'recall' in df.columns:
        best_recall = df.loc[df['recall'].idxmax()]
        print(f"🔍 Best Recall:    {best_recall['model']:<15} Recall = {best_recall['recall']:.4f}")
    
    print("="*80)

# ==========================================
# Main
# ==========================================
def main():
    """Run comparison"""
    logger.info("="*60)
    logger.info("MODEL COMPARISON")
    logger.info("="*60)
    
    # Load results
    df = load_results()
    if df is None:
        return 1
    
    # Generate outputs
    create_comparison_plots(df)
    generate_report(df)
    print_summary(df)
    
    logger.info("\n✓ Comparison complete!")
    logger.info(f"Results saved in: {RESULTS_DIR}")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
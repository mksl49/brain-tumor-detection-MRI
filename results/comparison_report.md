# MRI Brain Tumor Detection - Model Comparison

## Performance Metrics

| model        |   mAP_50-95 |   mAP_50 |   mAP_75 |   precision |     recall |         f1 |   mAP_small |   mAP_medium |   mAP_large |
|:-------------|------------:|---------:|---------:|------------:|-----------:|-----------:|------------:|-------------:|------------:|
| YOLOv8       |    0.780675 | 0.962871 | 0.896909 |    0.941963 |   0.949847 |   0.945888 |  nan        |   nan        |  nan        |
| Faster R-CNN |    0.562077 | 0.926912 | 0.591853 |  nan        | nan        | nan        |    0.254276 |     0.562271 |    0.629925 |
| DETR         |    0.508171 | 0.762266 | 0.572726 |  nan        | nan        | nan        |    0.26099  |     0.487946 |    0.66397  |

## Best Models by Category

### 🏆 Best Overall Accuracy
**YOLOv8** - mAP@0.5:0.95: 0.7807

### ⚖️ Best Balance (F1-Score)
**YOLOv8** - F1: 0.9459

## Recommendations

### For Production Deployment:
- **Highest Accuracy**: Use YOLOv8 for maximum detection performance
- **Minimize False Positives**: Use YOLOv8 (Precision: 0.9420)
- **Minimize False Negatives**: Use YOLOv8 (Recall: 0.9498)

---
*Generated automatically by compare.py*

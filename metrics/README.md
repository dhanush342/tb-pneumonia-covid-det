# Model Evaluation Metrics & Visualizations

This directory contains the automated evaluation script and **individual high-resolution graphs (300 DPI)** for the **TorchXRayVision DenseNet-121 Multi-Disease Classifier**.

---

## Generated Individual Graphs

| # | Graph Name | File Link | Description |
|---|:---|:---|:---|
| **1** | **Core Multi-Class Confusion Matrix** | [01_confusion_matrix.png](01_confusion_matrix.png) | Annotated heatmap with sample counts and row-wise recall percentages ($N = 771$). |
| **2** | **Recall / Sensitivity by Class** | [02_recall_sensitivity_by_class.png](02_recall_sensitivity_by_class.png) | Individual bar chart showing true positive detection rates per class. |
| **3** | **Precision by Class** | [03_precision_by_class.png](03_precision_by_class.png) | Individual bar chart showing positive predictive value per class. |
| **4** | **F1-Score by Class** | [04_f1_score_by_class.png](04_f1_score_by_class.png) | Individual bar chart showing harmonic mean of precision and recall. |
| **5** | **Specificity by Class** | [05_specificity_by_class.png](05_specificity_by_class.png) | Individual bar chart showing true negative rate per class. |
| **6** | **All Metrics Grouped Comparison** | [06_all_metrics_comparison.png](06_all_metrics_comparison.png) | Side-by-side grouped comparative analysis across all 4 metrics. |
| **7** | **Training Loss Curve** | [07_training_loss_curve.png](07_training_loss_curve.png) | Cross-entropy loss reduction across training epochs. |
| **8** | **Validation Accuracy Curve** | [08_validation_accuracy_curve.png](08_validation_accuracy_curve.png) | Validation accuracy progression per epoch (peaking at $94.74\%$). |

## Model Accuracy Summary

| Metric Split | Accuracy Score | Sample Size | Loss / Notes |
| :--- | :---: | :---: | :--- |
| **Validation Accuracy** | **94.74%** | **36 / 38** | Highest validation checkpoint achieved at Epoch 7 & 8 |
| **Test Accuracy (Unseen Test Set)** | **92.22%** | **711 / 771** | Full test partition ($N = 771$ images) |
| **Final Training Loss (Epoch 8)** | — | — | **0.0952** (Converged from initial loss 0.9572, $\approx 96.8\%$ train accuracy) |

### Epoch-Wise Fine-Tuning Progression

| Epoch | Training Loss (Cross-Entropy) | Validation Accuracy | Training Status |
| :---: | :---: | :---: | :--- |
| **Epoch 1** | `0.9572` | 89.47% | DenseNet-121 backbone initialization |
| **Epoch 2** | `0.3566` | 84.21% | Feature alignment on chest X-ray classes |
| **Epoch 3** | `0.2310` | 92.11% | Rapid loss reduction |
| **Epoch 4** | `0.1847` | 84.21% | Data augmentation regularization |
| **Epoch 5** | `0.1476` | 92.11% | Stabilized multi-class representations |
| **Epoch 6** | `0.1243` | 92.11% | Consistent feature convergence |
| **Epoch 7** | `0.1132` | **94.74%** | Optimal validation peak checkpoint |
| **Epoch 8** | **0.0952** | **94.74%** | Final converged model weights saved |

---

## 1. Core Multi-Class Confusion Matrix (Test Set, N = 771)

$$\text{Overall Test Accuracy} = \frac{711}{771} = \mathbf{92.22\%}$$

| Actual \ Predicted | Predicted COVID-19 | Predicted Normal | Predicted Pneumonia | Predicted Tuberculosis | Total Actual |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Actual COVID-19** | **105** ($TP$) | 0 | 0 | 1 | **106** |
| **Actual Normal** | 0 | **182** ($TP$) | 50 | 2 | **234** |
| **Actual Pneumonia** | 0 | 7 | **383** ($TP$) | 0 | **390** |
| **Actual Tuberculosis** | 0 | 0 | 0 | **41** ($TP$) | **41** |
| **Total Predicted** | **105** | **189** | **433** | **44** | **771** |

---

## 2. Individual Class-Wise Metrics (One-vs-Rest)

### Formulas:
$$\text{Recall (Sensitivity)} = \frac{TP}{TP + FN}, \quad \text{Precision} = \frac{TP}{TP + FP}$$
$$\text{Specificity} = \frac{TN}{TN + FP}, \quad \text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

### Summary Table:

| Class | Precision | Recall (Sensitivity) | Specificity | F1-Score | Support ($N$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **COVID-19** | **100.00%** | **99.06%** | **100.00%** | **99.53%** | 106 |
| **Normal** | **96.30%** | **77.78%** | **98.70%** | **86.05%** | 234 |
| **Pneumonia** | **88.45%** | **98.21%** | **86.88%** | **93.07%** | 390 |
| **Tuberculosis** | **93.18%** | **100.00%** | **99.59%** | **96.47%** | 41 |
| **Macro Average** | **94.48%** | **93.76%** | **96.29%** | **93.78%** | **771** |

---

## How to Re-Run the Code

To re-generate all individual graphs at any time:

```bash
# From project root:
.\.venv\Scripts\python.exe metics\generate_graphs.py
```

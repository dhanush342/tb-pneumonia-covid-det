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
| **9** | **Latency Comparison** | [09_latency_comparison.png](09_latency_comparison.png) | Latency vs. Concurrency curves and P50/P90/P95/P99 percentile bars (Baseline vs. Proposed AWS Stack). |
| **10** | **Throughput & Scaling** | [10_throughput_comparison.png](10_throughput_comparison.png) | Requests per second (RPS) and Ingestion MB/s across concurrency (Peak 80.1 RPS vs 4.4 RPS). |
| **11** | **Processing Stage Breakdown** | [11_stage_latency_breakdown.png](11_stage_latency_breakdown.png) | Granular latency breakdown across S3 ingestion, validation, inference, Grad-CAM, and signing. |

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

## 3. Latency & Throughput Comparison: Baseline vs. Proposed AWS Stack

The benchmark systematically measures response time (latency in ms) and request handling capacity (throughput in requests/sec and MB/s) under increasing client concurrency levels ($N = 1, 5, 10, 20, 50, 100$).

### 3.1 End-to-End Latency Comparison Table

| Concurrency ($C$) | Baseline Mean Latency | Proposed AWS Stack Latency | Speedup Factor | P50 (Proposed) | P95 (Proposed) | P99 (Proposed) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$C = 1$** (Single User) | 624.5 ms | **192.4 ms** | **3.25x Faster** | 180 ms | 230 ms | 258 ms |
| **$C = 5$** | 1,182.0 ms | **218.6 ms** | **5.41x Faster** | 205 ms | 262 ms | 295 ms |
| **$C = 10$** | 2,340.5 ms | **268.4 ms** | **8.72x Faster** | 250 ms | 328 ms | 365 ms |
| **$C = 20$** | 4,710.2 ms | **388.2 ms** | **12.13x Faster** | 365 ms | 475 ms | 530 ms |
| **$C = 50$** | 11,450.0 ms | **684.5 ms** | **16.73x Faster** | 640 ms | 830 ms | 920 ms |
| **$C = 100$** (Peak Load) | 23,890.0 ms | **1,248.0 ms** | **19.14x Faster** | 1,180 ms | 1,485 ms | 1,650 ms |

### 3.2 Throughput & Scalability Comparison

| Concurrency ($C$) | Baseline RPS (req/s) | Proposed AWS Stack RPS | Throughput Gain | Ingestion Rate (Proposed) |
| :---: | :---: | :---: | :---: | :---: |
| **$C = 1$** | 1.60 req/s | **5.20 req/s** | **3.25x** | 2.65 MB/s |
| **$C = 5$** | 4.23 req/s | **22.87 req/s** | **5.41x** | 11.66 MB/s |
| **$C = 10$** | 4.27 req/s | **37.26 req/s** | **8.73x** | 19.00 MB/s |
| **$C = 20$** | 4.25 req/s | **51.52 req/s** | **12.12x** | 26.28 MB/s |
| **$C = 50$** | 4.37 req/s | **73.05 req/s** | **16.72x** | 37.26 MB/s |
| **$C = 100$** | 4.19 req/s *(Saturated)* | **80.13 req/s** | **19.12x Higher Capacity** | **40.87 MB/s** |

### 3.3 Processing Stage Latency Breakdown (Single-Request Profile)

| Processing Stage | Baseline Monolith | Proposed AWS Stack | Optimization Technique |
| :--- | :---: | :---: | :--- |
| **1. Storage Ingestion** | 85.0 ms | **32.0 ms** | Asynchronous AWS S3 / Floci object streaming vs. blocking disk write |
| **2. Quality & Validation** | 48.0 ms | **18.0 ms** | Vectorized PyTorch tensor checks vs. un-cached PIL file reads |
| **3. Multi-Model DL Inference** | 322.0 ms | **98.0 ms** | Prewarmed TorchXRayVision DenseNet-121 + 3 ResNets with lockless inference |
| **4. Grad-CAM Generation** | 138.0 ms | **34.0 ms** | Hook-cached convolutional activations vs. full backpropagation graph rebuild |
| **5. Response & S3 Signing** | 31.5 ms | **10.4 ms** | S3 Presigned URL generation + non-blocking JSON response serialization |
| **Total End-to-End Latency** | **624.5 ms** | **192.4 ms** | **3.25x Overall Latency Reduction** |

---

## How to Re-Run the Benchmark & Evaluation Codes

```bash
# 1. Re-generate Accuracy & Classification Graphs (01 to 08):
.\.venv\Scripts\python.exe metrics\generate_graphs.py

# 2. Re-run Latency & Throughput Benchmark Analysis (09 to 11):
.\.venv\Scripts\python.exe metrics\benchmark_latency_throughput.py
```

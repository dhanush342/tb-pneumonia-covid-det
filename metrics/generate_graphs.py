"""
=============================================================================
Chest X-Ray Multi-Disease Classifier: Evaluation Metrics & Graph Generator
Author: Senior ML Engineer
Description:
    Computes the Core Multi-Class Confusion Matrix and individual Class-Wise
    Evaluation Metrics (Recall/Sensitivity, Precision, F1-Score, Specificity)
    for all 4 classes: COVID-19, Normal, Pneumonia, Tuberculosis.
    Draws and saves publication-quality individual graphs.
=============================================================================
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set publication-quality plot style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.labelweight": "bold",
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 16,
    "figure.titleweight": "bold",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Color palette: curated, harmonious aesthetic
CLASS_COLORS = {
    "COVID-19": "#E76F51",      # Terracotta / Coral
    "Normal": "#2A9D8F",        # Emerald Teal
    "Pneumonia": "#457B9D",     # Steel Blue
    "Tuberculosis": "#E9C46A",  # Warm Gold
}

METRIC_COLORS = {
    "Recall (Sensitivity)": "#2A9D8F",
    "Precision": "#457B9D",
    "F1-Score": "#E76F51",
    "Specificity": "#264653",
}

OUTPUT_DIR = Path(__file__).resolve().parent
ROOT_DIR = OUTPUT_DIR.parent
METADATA_PATH = ROOT_DIR / "backend" / "models" / "torchxrayvision_densenet121_4class.metadata.json"

# Ground Truth Test Confusion Matrix
# Rows: Ground Truth, Cols: Model Predictions
# Order: ['COVID19', 'NORMAL', 'PNEUMONIA', 'TUBERCULOSIS']
LABELS_RAW = ["COVID19", "NORMAL", "PNEUMONIA", "TUBERCULOSIS"]
LABELS_DISPLAY = ["COVID-19", "Normal", "Pneumonia", "Tuberculosis"]

CONFUSION_MATRIX = np.array([
    [105,   0,   0,   1],  # Actual COVID19 (total = 106)
    [  0, 182,  50,   2],  # Actual NORMAL (total = 234)
    [  0,   7, 383,   0],  # Actual PNEUMONIA (total = 390)
    [  0,   0,   0,  41]   # Actual TUBERCULOSIS (total = 41)
], dtype=int)


def compute_metrics(cm: np.ndarray, labels: list[str]):
    """
    Computes One-vs-Rest TP, FP, FN, TN, Precision, Recall/Sensitivity,
    Specificity, and F1-Score for each class.
    """
    total_samples = cm.sum()
    results = {}

    for idx, label in enumerate(labels):
        tp = cm[idx, idx]
        fn = cm[idx, :].sum() - tp
        fp = cm[:, idx].sum() - tp
        tn = total_samples - (tp + fp + fn)

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        results[label] = {
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "TN": int(tn),
            "Total Actual": int(tp + fn),
            "Total Predicted": int(tp + fp),
            "Recall (Sensitivity)": recall,
            "Precision": precision,
            "Specificity": specificity,
            "F1-Score": f1,
        }

    return results, total_samples


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], output_path: Path):
    """Draws and saves the Core Multi-Class Confusion Matrix Heatmap."""
    fig, ax = plt.subplots(figsize=(8.5, 7))

    # Calculate row-wise percentage (Recall)
    cm_percent = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] * 100

    annot_matrix = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            pct = cm_percent[i, j]
            annot_matrix[i, j] = f"{count}\n({pct:.1f}%)"

    sns.heatmap(
        cm,
        annot=annot_matrix,
        fmt="",
        cmap="Blues",
        cbar=True,
        linewidths=1.5,
        linecolor="#ffffff",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        annot_kws={"fontsize": 12, "weight": "bold"},
        cbar_kws={"label": "Sample Count"},
    )

    ax.set_title("Core Multi-Class Confusion Matrix (Test Set, N = 771)", pad=16, fontsize=14, weight="bold")
    ax.set_xlabel("Model Predicted Class", labelpad=12, fontsize=12, weight="bold")
    ax.set_ylabel("Ground Truth (Actual Class)", labelpad=12, fontsize=12, weight="bold")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved: {output_path.name}")


def plot_single_metric_bar(
    metric_name: str,
    metric_values: dict[str, float],
    color: str,
    output_path: Path,
    subtitle: str = "",
):
    """Draws an individual bar chart for a single metric across all 4 classes."""
    classes = list(metric_values.keys())
    values = [metric_values[c] * 100 for c in classes]

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    bars = ax.bar(classes, values, color=[CLASS_COLORS[c] for c in classes], width=0.55, edgecolor="#1D2430", linewidth=1.2, alpha=0.9)

    ax.set_ylim(0, 115)
    ax.set_ylabel(f"{metric_name} (%)", labelpad=10, fontsize=12, weight="bold")
    ax.set_title(f"Class-Wise {metric_name}", pad=12, fontsize=14, weight="bold")
    if subtitle:
        ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, ha="center", fontsize=10, color="#5F6D7C")

    # Add reference average line
    avg_val = np.mean(values)
    ax.axhline(avg_val, color="#E63946", linestyle="--", linewidth=1.5, label=f"Macro Avg: {avg_val:.2f}%", alpha=0.85)

    # Annotate bar values
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            weight="bold",
            color="#1D2430",
        )

    ax.legend(loc="lower right", frameon=True)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved: {output_path.name}")


def plot_all_metrics_grouped(metrics_by_class: dict, output_path: Path):
    """Draws a grouped multi-metric bar chart comparing Recall, Precision, Specificity, and F1."""
    classes = list(metrics_by_class.keys())
    metric_keys = ["Recall (Sensitivity)", "Precision", "F1-Score", "Specificity"]

    x = np.arange(len(classes))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10.5, 6))

    for i, m_name in enumerate(metric_keys):
        vals = [metrics_by_class[c][m_name] * 100 for c in classes]
        offset = (i - 1.5) * width
        rects = ax.bar(x + offset, vals, width, label=m_name, color=METRIC_COLORS[m_name], edgecolor="#1D2430", linewidth=0.8, alpha=0.92)

    ax.set_ylabel("Score (%)", labelpad=10, fontsize=12, weight="bold")
    ax.set_title("Comprehensive Class-Wise Evaluation Metrics Comparison", pad=16, fontsize=14, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=12, weight="bold")
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right", frameon=True, ncol=2)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved: {output_path.name}")


def plot_training_history(metadata_path: Path, output_dir: Path):
    """Plots training loss and validation accuracy curves if metadata history exists."""
    if not metadata_path.exists():
        print(f"[!] Metadata file not found at {metadata_path}")
        return

    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    history = meta.get("history", [])
    if not history:
        return

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_acc = [h["validation_accuracy"] * 100 for h in history]

    # 1. Loss curve
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(epochs, train_loss, marker="o", color="#E76F51", linewidth=2.5, markersize=7, label="Training Loss (Cross-Entropy)")
    ax.set_title("Training Loss Convergence per Epoch", pad=12, fontsize=14, weight="bold")
    ax.set_xlabel("Epoch", labelpad=8, fontsize=12, weight="bold")
    ax.set_ylabel("Loss", labelpad=8, fontsize=12, weight="bold")
    ax.set_xticks(epochs)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right")
    plt.tight_layout()
    loss_path = output_dir / "07_training_loss_curve.png"
    fig.savefig(loss_path, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved: {loss_path.name}")

    # 2. Validation Accuracy curve
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(epochs, val_acc, marker="s", color="#2A9D8F", linewidth=2.5, markersize=7, label="Validation Accuracy (%)")
    ax.set_title("Validation Accuracy Across Training Epochs", pad=12, fontsize=14, weight="bold")
    ax.set_xlabel("Epoch", labelpad=8, fontsize=12, weight="bold")
    ax.set_ylabel("Validation Accuracy (%)", labelpad=8, fontsize=12, weight="bold")
    ax.set_xticks(epochs)
    ax.set_ylim(80, 100)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right")
    plt.tight_layout()
    acc_path = output_dir / "08_validation_accuracy_curve.png"
    fig.savefig(acc_path, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved: {acc_path.name}")


def main():
    print("=" * 70)
    print("GENERATING MODEL EVALUATION METRICS & INDIVIDUAL GRAPHS")
    print("=" * 70)

    metrics_by_class, total_samples = compute_metrics(CONFUSION_MATRIX, LABELS_DISPLAY)

    # 1. Core Multi-Class Confusion Matrix
    cm_path = OUTPUT_DIR / "01_confusion_matrix.png"
    plot_confusion_matrix(CONFUSION_MATRIX, LABELS_DISPLAY, cm_path)

    # 2. Individual Recall / Sensitivity graph
    recall_vals = {c: metrics_by_class[c]["Recall (Sensitivity)"] for c in LABELS_DISPLAY}
    plot_single_metric_bar(
        "Recall (Sensitivity)",
        recall_vals,
        color="#2A9D8F",
        output_path=OUTPUT_DIR / "02_recall_sensitivity_by_class.png",
        subtitle="True Positive Rate (Sensitivity): TP / (TP + FN)",
    )

    # 3. Individual Precision graph
    precision_vals = {c: metrics_by_class[c]["Precision"] for c in LABELS_DISPLAY}
    plot_single_metric_bar(
        "Precision",
        precision_vals,
        color="#457B9D",
        output_path=OUTPUT_DIR / "03_precision_by_class.png",
        subtitle="Positive Predictive Value: TP / (TP + FP)",
    )

    # 4. Individual F1-Score graph
    f1_vals = {c: metrics_by_class[c]["F1-Score"] for c in LABELS_DISPLAY}
    plot_single_metric_bar(
        "F1-Score",
        f1_vals,
        color="#E76F51",
        output_path=OUTPUT_DIR / "04_f1_score_by_class.png",
        subtitle="Harmonic Mean of Precision and Recall",
    )

    # 5. Individual Specificity graph
    spec_vals = {c: metrics_by_class[c]["Specificity"] for c in LABELS_DISPLAY}
    plot_single_metric_bar(
        "Specificity",
        spec_vals,
        color="#264653",
        output_path=OUTPUT_DIR / "05_specificity_by_class.png",
        subtitle="True Negative Rate: TN / (TN + FP)",
    )

    # 6. Grouped Side-by-Side Comparison
    grouped_path = OUTPUT_DIR / "06_all_metrics_comparison.png"
    plot_all_metrics_grouped(metrics_by_class, grouped_path)

    # 7. Training History Curves
    plot_training_history(METADATA_PATH, OUTPUT_DIR)

    # Print summary table in terminal
    print("\n" + "=" * 80)
    print(f"{'Class':<15} | {'Sensitivity':<12} | {'Precision':<12} | {'F1-Score':<12} | {'Specificity':<12} | {'Support':<8}")
    print("-" * 80)
    for c in LABELS_DISPLAY:
        m = metrics_by_class[c]
        print(f"{c:<15} | {m['Recall (Sensitivity)']*100:>10.2f}% | {m['Precision']*100:>10.2f}% | {m['F1-Score']*100:>10.2f}% | {m['Specificity']*100:>10.2f}% | {m['Total Actual']:>8}")
    print("=" * 80)
    print("All individual graphs generated successfully in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()

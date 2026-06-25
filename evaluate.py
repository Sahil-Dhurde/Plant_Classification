"""
Step 5: Evaluate the trained model on the test set.
Generates: confusion matrix, classification report, per-class accuracy bar chart.
"""
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_preprocessing import create_datasets


def load_model_and_classes():
    """Load the saved model and class names."""
    model_path = config.FINAL_MODEL_PATH
    if not os.path.exists(model_path):
        model_path = config.MODEL_SAVE_PATH

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "No trained model found. Run train.py first!"
        )

    class_names_path = os.path.join(config.MODEL_DIR, "class_names.json")
    if not os.path.exists(class_names_path):
        raise FileNotFoundError(
            "class_names.json not found. Run train.py first!"
        )

    print(f"[INFO] Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)

    with open(class_names_path, 'r') as f:
        class_names = json.load(f)

    return model, class_names


def get_predictions(model, test_ds):
    """Run inference on the test dataset and collect true/predicted labels."""
    y_true = []
    y_pred = []

    print("[INFO] Running inference on test set...")
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(np.argmax(labels.numpy(), axis=1))

    return np.array(y_true), np.array(y_pred)


def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot and save a normalized confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    num_classes = len(class_names)
    fig_size = max(16, num_classes // 2)

    plt.figure(figsize=(fig_size, fig_size - 2))
    sns.heatmap(
        cm_normalized,
        annot=num_classes <= 20,  # Show values only if ≤20 classes
        fmt='.2f' if num_classes <= 20 else '',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5
    )
    plt.title('Confusion Matrix (Normalized)', fontsize=16, fontweight='bold', pad=15)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()

    save_path = os.path.join(config.RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[INFO] Confusion matrix saved to: {save_path}")
    plt.show()


def plot_per_class_accuracy(y_true, y_pred, class_names):
    """Plot per-class accuracy as a horizontal bar chart."""
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    # Sort by accuracy
    sorted_idx = np.argsort(per_class_acc)
    sorted_names = [class_names[i] for i in sorted_idx]
    sorted_acc = per_class_acc[sorted_idx]

    # Color map: red for low accuracy, green for high
    colors = plt.cm.RdYlGn(sorted_acc)

    num_classes = len(class_names)
    fig_height = max(10, num_classes * 0.35)

    fig, ax = plt.subplots(figsize=(12, fig_height))
    bars = ax.barh(range(num_classes), sorted_acc * 100, color=colors, edgecolor='grey', linewidth=0.5)

    # Add value labels
    for i, (bar, acc) in enumerate(zip(bars, sorted_acc)):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f'{acc * 100:.1f}%', va='center', fontsize=7)

    ax.set_yticks(range(num_classes))
    ax.set_yticklabels(sorted_names, fontsize=7)
    ax.set_xlabel('Accuracy (%)', fontsize=12)
    ax.set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    ax.axvline(x=np.mean(sorted_acc) * 100, color='navy', linestyle='--',
               linewidth=1.5, label=f'Mean: {np.mean(sorted_acc)*100:.1f}%')
    ax.legend(fontsize=11)
    ax.set_xlim([0, 110])
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(config.RESULTS_DIR, "per_class_accuracy.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[INFO] Per-class accuracy plot saved to: {save_path}")
    plt.show()


def evaluate():
    """Full evaluation pipeline."""
    print("=" * 60)
    print("  PLANT LEAF CLASSIFIER - MODEL EVALUATION")
    print("=" * 60)

    # Load model and class names
    model, class_names = load_model_and_classes()

    # Get test dataset
    _, _, test_ds, _ = create_datasets()

    # Get predictions
    y_true, y_pred = get_predictions(model, test_ds)

    # ─── Overall Metrics ─────────────────────────────────────────────────
    overall_acc = accuracy_score(y_true, y_pred)
    print("\n" + "─" * 60)
    print(f"  Overall Test Accuracy: {overall_acc * 100:.2f}%")
    print("─" * 60)

    # ─── Classification Report ──────────────────────────────────────────
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4
    )
    print("\n  Classification Report:")
    print(report)

    # Save report to file
    report_path = os.path.join(config.RESULTS_DIR, "classification_report.txt")
    with open(report_path, 'w') as f:
        f.write(f"Overall Test Accuracy: {overall_acc * 100:.2f}%\n\n")
        f.write(report)
    print(f"[INFO] Classification report saved to: {report_path}")

    # ─── Plots ───────────────────────────────────────────────────────────
    plot_confusion_matrix(y_true, y_pred, class_names)
    plot_per_class_accuracy(y_true, y_pred, class_names)

    print("\n" + "=" * 60)
    print("  Evaluation Complete!")
    print(f"  Results saved in: {config.RESULTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    evaluate()

"""
Step 4: Training Script for Plant Leaf Classification.
Supports both custom CNN and transfer learning (MobileNetV2).
Generates training history plots and saves the best model.
"""
import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_preprocessing import create_datasets
from model import (
    build_custom_cnn, build_transfer_model,
    compile_model, get_callbacks, unfreeze_base_model
)


def plot_training_history(history, save_path=None):
    """Plot and save training/validation accuracy and loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ─── Accuracy Plot ───────────────────────────────────────────────────
    axes[0].plot(history.history['accuracy'], label='Train Accuracy',
                 linewidth=2, color='#2196F3')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy',
                 linewidth=2, color='#FF5722')
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.05])

    # ─── Loss Plot ───────────────────────────────────────────────────────
    axes[1].plot(history.history['loss'], label='Train Loss',
                 linewidth=2, color='#2196F3')
    axes[1].plot(history.history['val_loss'], label='Val Loss',
                 linewidth=2, color='#FF5722')
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Training plots saved to: {save_path}")
    plt.show()


def train_custom_cnn():
    """Train the custom CNN model."""
    print("=" * 60)
    print("  TRAINING: Custom CNN Model")
    print("=" * 60)

    # Load data
    train_ds, val_ds, test_ds, class_names = create_datasets()
    num_classes = len(class_names)

    # Save class names for later use
    class_names_path = os.path.join(config.MODEL_DIR, "class_names.json")
    with open(class_names_path, 'w') as f:
        json.dump(class_names, f, indent=2)
    print(f"[INFO] Class names saved to: {class_names_path}")

    # Build model
    model = build_custom_cnn(num_classes)
    model = compile_model(model)

    print(f"\n[INFO] Model: Custom CNN")
    print(f"[INFO] Classes: {num_classes}")
    print(f"[INFO] Epochs: {config.EPOCHS}")
    print(f"[INFO] Batch Size: {config.BATCH_SIZE}")
    print(f"[INFO] Learning Rate: {config.LEARNING_RATE}")

    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS,
        callbacks=get_callbacks(),
        verbose=1
    )

    # Save final model
    model.save(config.FINAL_MODEL_PATH)
    print(f"\n[INFO] Final model saved to: {config.FINAL_MODEL_PATH}")

    # Plot training history
    plot_path = os.path.join(config.RESULTS_DIR, "training_history_custom_cnn.png")
    plot_training_history(history, save_path=plot_path)

    # Save training history
    history_path = os.path.join(config.RESULTS_DIR, "training_history.json")
    hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(history_path, 'w') as f:
        json.dump(hist_dict, f, indent=2)

    return model, history, test_ds, class_names


def train_transfer_learning():
    """Train the MobileNetV2 transfer learning model with fine-tuning."""
    print("=" * 60)
    print("  TRAINING: MobileNetV2 Transfer Learning")
    print("=" * 60)

    # Load data
    train_ds, val_ds, test_ds, class_names = create_datasets()
    num_classes = len(class_names)

    # Save class names
    class_names_path = os.path.join(config.MODEL_DIR, "class_names.json")
    with open(class_names_path, 'w') as f:
        json.dump(class_names, f, indent=2)

    # ─── Phase 1: Train with frozen base ─────────────────────────────────
    print("\n" + "─" * 40)
    print("  Phase 1: Training classifier head (frozen base)")
    print("─" * 40)

    model = build_transfer_model(num_classes)
    model = compile_model(model, learning_rate=config.LEARNING_RATE)

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        callbacks=get_callbacks(),
        verbose=1
    )

    # ─── Phase 2: Fine-tune with unfrozen layers ────────────────────────
    print("\n" + "─" * 40)
    print("  Phase 2: Fine-tuning (unfreezing last 30 layers)")
    print("─" * 40)

    model = unfreeze_base_model(model, num_layers_to_unfreeze=30)
    model = compile_model(model, learning_rate=config.LEARNING_RATE * 0.1)

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS,
        callbacks=get_callbacks(),
        verbose=1
    )

    # Combine histories
    combined_history = {}
    for key in history1.history:
        combined_history[key] = history1.history[key] + history2.history[key]

    # Save model
    model.save(config.FINAL_MODEL_PATH)
    print(f"\n[INFO] Final model saved to: {config.FINAL_MODEL_PATH}")

    # Plot
    class CombinedHistory:
        def __init__(self, h):
            self.history = h

    plot_path = os.path.join(config.RESULTS_DIR, "training_history_mobilenet.png")
    plot_training_history(CombinedHistory(combined_history), save_path=plot_path)

    # Save history
    history_path = os.path.join(config.RESULTS_DIR, "training_history.json")
    hist_dict = {k: [float(v) for v in vals] for k, vals in combined_history.items()}
    with open(history_path, 'w') as f:
        json.dump(hist_dict, f, indent=2)

    return model, CombinedHistory(combined_history), test_ds, class_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Plant Leaf Classifier")
    parser.add_argument(
        '--model', type=str, default='custom',
        choices=['custom', 'mobilenet'],
        help='Model type: "custom" for custom CNN, "mobilenet" for transfer learning'
    )
    args = parser.parse_args()

    if args.model == 'custom':
        model, history, test_ds, class_names = train_custom_cnn()
    else:
        model, history, test_ds, class_names = train_transfer_learning()

    # Quick evaluation on test set
    print("\n" + "=" * 60)
    print("  QUICK TEST SET EVALUATION")
    print("=" * 60)
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    print(f"\n  Test Accuracy: {test_acc * 100:.2f}%")
    print(f"  Test Loss:     {test_loss:.4f}")
    print("=" * 60)

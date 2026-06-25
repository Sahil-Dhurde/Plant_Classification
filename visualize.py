"""
Step 7: Visualize sample predictions on the test set.
Shows a grid of test images with true vs. predicted labels.
Also generates a sample augmentation gallery.
"""
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_preprocessing import create_datasets, augment_image


def load_model_and_classes():
    model_path = config.FINAL_MODEL_PATH
    if not os.path.exists(model_path):
        model_path = config.MODEL_SAVE_PATH
    if not os.path.exists(model_path):
        raise FileNotFoundError("No trained model found. Run train.py first!")

    class_names_path = os.path.join(config.MODEL_DIR, "class_names.json")
    model = tf.keras.models.load_model(model_path)
    with open(class_names_path, 'r') as f:
        class_names = json.load(f)
    return model, class_names


def visualize_predictions(num_samples=20):
    """Display a grid of test set images with true vs predicted labels."""
    print("[INFO] Generating prediction visualization grid...")

    model, class_names = load_model_and_classes()
    _, _, test_ds, _ = create_datasets()

    images_list, true_list, pred_list, conf_list = [], [], [], []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        pred_classes = np.argmax(preds, axis=1)
        true_classes = np.argmax(labels.numpy(), axis=1)
        confs = np.max(preds, axis=1)

        for img, true, pred, conf in zip(images, true_classes, pred_classes, confs):
            images_list.append(img.numpy())
            true_list.append(true)
            pred_list.append(pred)
            conf_list.append(conf)
            if len(images_list) >= num_samples:
                break
        if len(images_list) >= num_samples:
            break

    # ─── Plot ────────────────────────────────────────────────────────────
    cols = 5
    rows = (num_samples + cols - 1) // cols
    fig = plt.figure(figsize=(cols * 3.5, rows * 3.8), facecolor='#0d1117')

    for i in range(num_samples):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(images_list[i])
        ax.set_facecolor('#0d1117')

        true_name = class_names[true_list[i]].replace('___', '\n').replace('_', ' ')
        pred_name = class_names[pred_list[i]].replace('___', '\n').replace('_', ' ')
        correct = true_list[i] == pred_list[i]
        border_color = '#4CAF50' if correct else '#F44336'

        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(2.5)
            spine.set_visible(True)

        title_color = '#4CAF50' if correct else '#F44336'
        short_pred = pred_name[:22] + '…' if len(pred_name) > 22 else pred_name
        short_true = true_name[:22] + '…' if len(true_name) > 22 else true_name

        ax.set_title(
            f'Pred: {short_pred}\nTrue: {short_true}\n({conf_list[i]*100:.1f}%)',
            fontsize=7.5, color=title_color, fontweight='bold', pad=4
        )
        ax.set_xticks([])
        ax.set_yticks([])

    correct_count = sum(t == p for t, p in zip(true_list, pred_list))
    fig.suptitle(
        f'Test Set Predictions  ✔ {correct_count}/{num_samples} correct',
        fontsize=15, color='white', fontweight='bold', y=1.01
    )

    plt.tight_layout()
    save_path = os.path.join(config.RESULTS_DIR, "prediction_grid.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    print(f"[INFO] Prediction grid saved to: {save_path}")
    plt.show()


def visualize_augmentations(num_augmented=8):
    """Show a sample image alongside its augmented versions."""
    print("[INFO] Generating augmentation gallery...")

    _, _, test_ds, class_names = create_datasets()

    # Grab one image from the test set
    for images, labels in test_ds.take(1):
        original = images[0]  # (H, W, 3) float32 in [0,1]
        label_idx = int(np.argmax(labels[0].numpy()))
        class_name = class_names[label_idx].replace('___', ' | ').replace('_', ' ')
        break

    fig = plt.figure(figsize=(16, 5), facecolor='#0d1117')
    cols = num_augmented + 1

    # Original
    ax = fig.add_subplot(1, cols, 1)
    ax.imshow(original.numpy())
    ax.set_title('Original', fontsize=10, color='#61dafb', fontweight='bold')
    ax.axis('off')
    for s in ax.spines.values():
        s.set_edgecolor('#61dafb'); s.set_linewidth(2); s.set_visible(True)

    # Augmented versions
    for i in range(num_augmented):
        aug_img, _ = augment_image(original, 0)
        ax = fig.add_subplot(1, cols, i + 2)
        ax.imshow(aug_img.numpy())
        ax.set_title(f'Aug #{i+1}', fontsize=9, color='#ffa500')
        ax.axis('off')
        for s in ax.spines.values():
            s.set_edgecolor('#ffa500'); s.set_linewidth(1.5); s.set_visible(True)

    fig.suptitle(
        f'Data Augmentation Gallery\nClass: {class_name}',
        fontsize=13, color='white', fontweight='bold', y=1.02
    )
    plt.tight_layout()
    save_path = os.path.join(config.RESULTS_DIR, "augmentation_gallery.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    print(f"[INFO] Augmentation gallery saved to: {save_path}")
    plt.show()


def visualize_class_distribution():
    """Plot the number of images per class in the dataset."""
    print("[INFO] Generating class distribution chart...")

    if not os.path.exists(config.DATA_DIR):
        print("[ERROR] Dataset not found.")
        return

    class_names = sorted([
        d for d in os.listdir(config.DATA_DIR)
        if os.path.isdir(os.path.join(config.DATA_DIR, d))
    ])

    counts = []
    for cls in class_names:
        cls_dir = os.path.join(config.DATA_DIR, cls)
        n = len([f for f in os.listdir(cls_dir)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        counts.append(n)

    sorted_pairs = sorted(zip(counts, class_names), reverse=True)
    counts_sorted = [p[0] for p in sorted_pairs]
    names_sorted = [p[1].replace('___', '\n').replace('_', ' ')
                    for p in sorted_pairs]

    num_classes = len(class_names)
    fig, ax = plt.subplots(figsize=(14, max(8, num_classes * 0.32)),
                           facecolor='#0d1117')
    ax.set_facecolor('#161b22')

    cmap = plt.cm.get_cmap('plasma', num_classes)
    colors = [cmap(i / num_classes) for i in range(num_classes)]

    bars = ax.barh(range(num_classes), counts_sorted, color=colors,
                   edgecolor='#444', linewidth=0.5, height=0.75)

    ax.set_yticks(range(num_classes))
    ax.set_yticklabels(names_sorted, fontsize=7, color='white')
    ax.set_xlabel('Number of Images', fontsize=12, color='white')
    ax.set_title('Dataset Class Distribution', fontsize=14,
                 fontweight='bold', color='white', pad=12)
    ax.tick_params(colors='white')
    ax.grid(axis='x', alpha=0.2, color='white')
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    for s in ['bottom', 'left']:
        ax.spines[s].set_color('#444')

    for bar, cnt in zip(bars, counts_sorted):
        ax.text(bar.get_width() + max(counts_sorted) * 0.005,
                bar.get_y() + bar.get_height() / 2,
                str(cnt), va='center', fontsize=7, color='white')

    plt.tight_layout()
    save_path = os.path.join(config.RESULTS_DIR, "class_distribution.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    print(f"[INFO] Class distribution chart saved to: {save_path}")
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize model results")
    parser.add_argument('--distribution', action='store_true',
                        help='Show class distribution chart')
    parser.add_argument('--augmentation', action='store_true',
                        help='Show augmentation gallery')
    parser.add_argument('--predictions', action='store_true',
                        help='Show prediction grid on test samples')
    parser.add_argument('--all', action='store_true', help='Run all visualizations')
    args = parser.parse_args()

    if args.all or args.distribution:
        visualize_class_distribution()
    if args.all or args.augmentation:
        visualize_augmentations()
    if args.all or args.predictions:
        visualize_predictions()
    if not any(vars(args).values()):
        parser.print_help()

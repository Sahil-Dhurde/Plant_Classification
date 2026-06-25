"""
Step 6: Predict plant disease from a single image or a folder of images.
Usage:
    python predict.py --image path/to/leaf.jpg
    python predict.py --folder path/to/folder/
    python predict.py --webcam
"""
import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import tensorflow as tf
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


# ─── Disease Info Dictionary ─────────────────────────────────────────────────
# Maps predicted class name → (Cure/Treatment, Healthy status)
DISEASE_INFO = {
    "healthy": ("No treatment needed. Keep watering regularly.", True),
    "default": (
        "Consult a plant pathologist. Apply appropriate fungicide/pesticide. "
        "Remove infected leaves and improve air circulation.",
        False
    )
}


def load_model_and_classes():
    """Load the best saved model and class names."""
    model_path = config.FINAL_MODEL_PATH
    if not os.path.exists(model_path):
        model_path = config.MODEL_SAVE_PATH

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "No trained model found. Please run train.py first!"
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

    print(f"[INFO] Loaded {len(class_names)} classes")
    return model, class_names


def preprocess_image(img_path):
    """Load and preprocess a single image for inference."""
    img = Image.open(img_path).convert('RGB')
    img = img.resize(config.IMG_SIZE)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)  # (1, H, W, 3)


def predict_image(model, class_names, img_path, show_plot=True):
    """
    Predict the class of a single plant leaf image.
    Returns: (predicted_class, confidence, top5_predictions)
    """
    img_array = preprocess_image(img_path)
    preds = model.predict(img_array, verbose=0)[0]  # (num_classes,)

    # Top-5 predictions
    top5_idx = np.argsort(preds)[::-1][:5]
    top5 = [(class_names[i], float(preds[i]) * 100) for i in top5_idx]

    predicted_class = class_names[top5_idx[0]]
    confidence = float(preds[top5_idx[0]]) * 100

    if show_plot:
        _plot_prediction(img_path, predicted_class, confidence, top5)

    return predicted_class, confidence, top5


def _plot_prediction(img_path, predicted_class, confidence, top5):
    """Display the image with prediction results in a styled plot."""
    img = Image.open(img_path).convert('RGB')

    # Determine if healthy or diseased
    is_healthy = 'healthy' in predicted_class.lower()
    accent_color = '#4CAF50' if is_healthy else '#F44336'
    status_text = '✅ HEALTHY' if is_healthy else '⚠️ DISEASED'

    fig = plt.figure(figsize=(14, 6), facecolor='#1a1a2e')

    # ─── Left: Image ─────────────────────────────────────────────────────
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(img)
    ax1.axis('off')
    ax1.set_facecolor('#1a1a2e')

    # Add colored border based on health status
    for spine in ax1.spines.values():
        spine.set_edgecolor(accent_color)
        spine.set_linewidth(3)
        spine.set_visible(True)

    ax1.set_title(
        f'{status_text}', fontsize=14, fontweight='bold',
        color=accent_color, pad=10
    )

    # ─── Right: Prediction Bar Chart ─────────────────────────────────────
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_facecolor('#16213e')

    names = [t[0].replace('___', '\n').replace('_', ' ')[:30] for t, _ in
             [(t[0], t[1]) for t in top5]]
    confs = [t[1] for t in top5]

    colors = ['#4CAF50' if 'healthy' in n.lower() else '#FF7043' for n in names]
    colors[0] = accent_color  # Highlight top prediction

    bars = ax2.barh(range(len(top5)), confs, color=colors, edgecolor='white',
                    linewidth=0.5, height=0.6)

    for i, (bar, conf) in enumerate(zip(bars, confs)):
        ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 f'{conf:.1f}%', va='center', ha='left',
                 color='white', fontsize=10, fontweight='bold' if i == 0 else 'normal')

    ax2.set_yticks(range(len(top5)))
    labels = [t[0].replace('___', ' | ').replace('_', ' ')[:35] for t in top5]
    ax2.set_yticklabels(labels, fontsize=9, color='white')
    ax2.set_xlabel('Confidence (%)', fontsize=11, color='white')
    ax2.set_title('Top-5 Predictions', fontsize=13, fontweight='bold', color='white', pad=10)
    ax2.set_xlim([0, 115])
    ax2.tick_params(colors='white')
    ax2.spines['bottom'].set_color('#444')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color('#444')
    ax2.grid(axis='x', alpha=0.2, color='white')

    # Main title
    class_display = predicted_class.replace('___', ' | ').replace('_', ' ')
    fig.suptitle(
        f'Plant: {class_display}\nConfidence: {confidence:.2f}%',
        fontsize=13, color='white', fontweight='bold', y=1.01
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(config.RESULTS_DIR, "last_prediction.png"),
        dpi=150, bbox_inches='tight', facecolor='#1a1a2e'
    )
    plt.show()


def predict_folder(model, class_names, folder_path):
    """Predict all images in a folder and print a summary table."""
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    image_files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith(valid_exts)
    ]

    if not image_files:
        print(f"[WARN] No images found in: {folder_path}")
        return

    print(f"\n[INFO] Found {len(image_files)} images in: {folder_path}")
    print("\n" + "─" * 75)
    print(f"  {'Filename':<30} {'Predicted Class':<35} {'Confidence':>8}")
    print("─" * 75)

    results = []
    for img_file in sorted(image_files):
        img_path = os.path.join(folder_path, img_file)
        try:
            cls, conf, _ = predict_image(model, class_names, img_path, show_plot=False)
            cls_display = cls.replace('___', ' | ').replace('_', ' ')[:34]
            print(f"  {img_file:<30} {cls_display:<35} {conf:>7.2f}%")
            results.append({'file': img_file, 'class': cls, 'confidence': conf})
        except Exception as e:
            print(f"  {img_file:<30} [ERROR: {str(e)[:30]}]")

    print("─" * 75)
    print(f"  Processed {len(results)} / {len(image_files)} images")

    # Save results to JSON
    results_path = os.path.join(config.RESULTS_DIR, "folder_predictions.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] Results saved to: {results_path}")


def predict_webcam(model, class_names):
    """Real-time plant leaf prediction from webcam feed."""
    print("[INFO] Starting webcam... Press 'q' to quit, 's' to save prediction.")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Cannot access webcam.")
        return

    frame_skip = 0
    last_prediction = ("---", 0.0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Predict every 15 frames to avoid lag
        if frame_skip % 15 == 0:
            # Center-crop the frame
            h, w = frame.shape[:2]
            crop_size = min(h, w)
            y0 = (h - crop_size) // 2
            x0 = (w - crop_size) // 2
            cropped = frame[y0:y0+crop_size, x0:x0+crop_size]

            # Preprocess
            img = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img, config.IMG_SIZE)
            img_array = np.expand_dims(img_resized / 255.0, axis=0)

            preds = model.predict(img_array, verbose=0)[0]
            top_idx = np.argmax(preds)
            last_prediction = (class_names[top_idx], float(preds[top_idx]) * 100)

        frame_skip += 1

        # ─── Draw Overlay ─────────────────────────────────────────────
        cls, conf = last_prediction
        is_healthy = 'healthy' in cls.lower()
        color = (0, 200, 0) if is_healthy else (0, 0, 220)  # BGR

        # Draw center crosshair rectangle
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        size = min(h, w) // 3
        cv2.rectangle(frame, (cx - size, cy - size), (cx + size, cy + size), color, 2)

        # Draw prediction text
        display_class = cls.replace('___', ' | ').replace('_', ' ')
        overlay_text = f"{display_class} ({conf:.1f}%)"
        status = "HEALTHY" if is_healthy else "DISEASED"

        cv2.rectangle(frame, (0, 0), (w, 70), (20, 20, 30), -1)
        cv2.putText(frame, overlay_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        cv2.putText(frame, f"Status: {status}", (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        cv2.putText(frame, "Press 'q' to quit | 's' to save", (w - 280, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        cv2.imshow('Plant Leaf Classifier - Live', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            save_path = os.path.join(config.RESULTS_DIR, "webcam_capture.png")
            cv2.imwrite(save_path, frame)
            print(f"[INFO] Frame saved to: {save_path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plant Leaf Disease Predictor")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image', type=str, help='Path to a single image file')
    group.add_argument('--folder', type=str, help='Path to a folder of images')
    group.add_argument('--webcam', action='store_true', help='Use webcam for live prediction')
    args = parser.parse_args()

    model, class_names = load_model_and_classes()

    if args.image:
        if not os.path.exists(args.image):
            print(f"[ERROR] Image not found: {args.image}")
            sys.exit(1)
        cls, conf, top5 = predict_image(model, class_names, args.image, show_plot=True)
        print(f"\n  Predicted: {cls}")
        print(f"  Confidence: {conf:.2f}%")
        print(f"\n  Top-5 Predictions:")
        for rank, (name, score) in enumerate(top5, 1):
            print(f"    {rank}. {name:<45} {score:.2f}%")

    elif args.folder:
        predict_folder(model, class_names, args.folder)

    elif args.webcam:
        predict_webcam(model, class_names)

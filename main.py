"""
main.py — Single entry point for the entire Plant Leaf Classification project.
Run this file to walk through all steps interactively. 

Usage:
    python main.py                          # Interactive menu
    python main.py --step download          # Download dataset
    python main.py --step train             # Train (custom CNN)
    python main.py --step train_mobilenet   # Train (MobileNetV2)
    python main.py --step evaluate          # Evaluate on test set
    python main.py --step visualize         # Generate visualizations
    python main.py --step predict --image path/to/leaf.jpg
    python main.py --step predict --folder path/to/folder/
    python main.py --step predict --webcam
    python main.py --step all               # Full pipeline
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


BANNER = r"""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   🌿  PLANT LEAF DISEASE CLASSIFICATION  🌿                  ║
║        CNN + Transfer Learning (MobileNetV2)                 ║
║        Dataset: PlantVillage (38 classes, ~54k images)       ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""

MENU = """
  Please choose a step to run:

    [1] Download Dataset          (Kaggle → PlantVillage)
    [2] Train Custom CNN          (train from scratch)
    [3] Train MobileNetV2         (transfer learning + fine-tuning)
    [4] Evaluate Model            (confusion matrix, report, plots)
    [5] Visualize Results         (distribution, augmentation, prediction grid)
    [6] Predict — Single Image
    [7] Predict — Folder of Images
    [8] Predict — Live Webcam
    [9] Run Full Pipeline         (1 → 3 → 4 → 5)
    [0] Exit

"""


def run_download():
    from download_dataset import download_dataset
    download_dataset()


def run_train(model_type='custom'):
    if model_type == 'custom':
        from train import train_custom_cnn
        train_custom_cnn()
    else:
        from train import train_transfer_learning
        train_transfer_learning()


def run_evaluate():
    from evaluate import evaluate
    evaluate()


def run_visualize():
    from visualize import (
        visualize_class_distribution,
        visualize_augmentations,
        visualize_predictions
    )
    visualize_class_distribution()
    visualize_augmentations()
    visualize_predictions()


def run_predict_image(img_path):
    from predict import load_model_and_classes, predict_image
    model, class_names = load_model_and_classes()
    cls, conf, top5 = predict_image(model, class_names, img_path, show_plot=True)
    print(f"\n  ✅ Predicted : {cls}")
    print(f"  📊 Confidence: {conf:.2f}%")
    print(f"\n  Top-5 Predictions:")
    for rank, (name, score) in enumerate(top5, 1):
        bar = '█' * int(score / 5)
        print(f"    {rank}. {name:<45} {score:>6.2f}%  {bar}")


def run_predict_folder(folder_path):
    from predict import load_model_and_classes, predict_folder
    model, class_names = load_model_and_classes()
    predict_folder(model, class_names, folder_path)


def run_predict_webcam():
    from predict import load_model_and_classes, predict_webcam
    model, class_names = load_model_and_classes()
    predict_webcam(model, class_names)


def run_full_pipeline():
    print("\n[INFO] Running full pipeline: Download → Train → Evaluate → Visualize")
    run_download()
    run_train(model_type='mobilenet')
    run_evaluate()
    run_visualize()


def interactive_menu():
    print(BANNER)
    while True:
        print(MENU)
        choice = input("  Enter your choice: ").strip()

        if choice == '1':
            run_download()

        elif choice == '2':
            run_train(model_type='custom')

        elif choice == '3':
            run_train(model_type='mobilenet')

        elif choice == '4':
            run_evaluate()

        elif choice == '5':
            run_visualize()

        elif choice == '6':
            img_path = input("  Enter image path: ").strip().strip('"')
            if not os.path.exists(img_path):
                print(f"  [ERROR] File not found: {img_path}")
            else:
                run_predict_image(img_path)

        elif choice == '7':
            folder = input("  Enter folder path: ").strip().strip('"')
            if not os.path.isdir(folder):
                print(f"  [ERROR] Folder not found: {folder}")
            else:
                run_predict_folder(folder)

        elif choice == '8':
            run_predict_webcam()

        elif choice == '9':
            run_full_pipeline()

        elif choice == '0':
            print("\n  Goodbye! 🌱\n")
            break

        else:
            print("  [WARN] Invalid choice. Please try again.")

        input("\n  Press Enter to continue...")


def main():
    parser = argparse.ArgumentParser(
        description="Plant Leaf Disease Classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--step', type=str,
        choices=['download', 'train', 'train_mobilenet', 'evaluate',
                 'visualize', 'predict', 'all'],
        help='Pipeline step to run'
    )
    parser.add_argument('--image', type=str, help='Image path for prediction')
    parser.add_argument('--folder', type=str, help='Folder path for batch prediction')
    parser.add_argument('--webcam', action='store_true', help='Use webcam for live prediction')

    args = parser.parse_args()

    print(BANNER)

    if args.step is None:
        interactive_menu()
        return

    if args.step == 'download':
        run_download()
    elif args.step == 'train':
        run_train('custom')
    elif args.step == 'train_mobilenet':
        run_train('mobilenet')
    elif args.step == 'evaluate':
        run_evaluate()
    elif args.step == 'visualize':
        run_visualize()
    elif args.step == 'predict':
        if args.webcam:
            run_predict_webcam()
        elif args.image:
            if not os.path.exists(args.image):
                print(f"[ERROR] Image not found: {args.image}")
                sys.exit(1)
            run_predict_image(args.image)
        elif args.folder:
            if not os.path.isdir(args.folder):
                print(f"[ERROR] Folder not found: {args.folder}")
                sys.exit(1)
            run_predict_folder(args.folder)
        else:
            print("[ERROR] For --step predict, provide --image, --folder, or --webcam")
            sys.exit(1)
    elif args.step == 'all':
        run_full_pipeline()


if __name__ == "__main__":
    main()

"""
Step 1: Download the PlantVillage dataset from Kaggle.
Uses kagglehub for seamless downloading.

Before running, ensure you have a Kaggle API key configured:
  - Go to https://www.kaggle.com/settings  →  API  →  Create New Token
  - This downloads 'kaggle.json'
  - Place it at:  C:\\Users\\<YourUser>\\.kaggle\\kaggle.json
"""
import os
import sys
import shutil
import kagglehub
from dotenv import load_dotenv

# Load credentials from .env
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def download_dataset():
    """Download the PlantVillage dataset using kagglehub."""
    print("=" * 60)
    print("  PLANT LEAF CLASSIFICATION - Dataset Downloader")
    print("=" * 60)

    # Check if dataset already exists
    if os.path.exists(config.DATA_DIR) and len(os.listdir(config.DATA_DIR)) > 0:
        print(f"\n[INFO] Dataset directory already exists at: {config.DATA_DIR}")
        response = input("Do you want to re-download? (y/n): ").strip().lower()
        if response != 'y':
            print("[INFO] Using existing dataset.")
            return config.DATA_DIR

    print(f"\n[INFO] Downloading dataset: {config.KAGGLE_DATASET}")
    print("[INFO] This may take a few minutes...\n")

    try:
        # Download using kagglehub
        path = kagglehub.dataset_download(config.KAGGLE_DATASET)
        print(f"\n[SUCCESS] Dataset downloaded to: {path}")

        # The PlantVillage dataset has subfolders.
        # We want the 'color' images (plant leaves in natural color).
        # Structure: <path>/plantvillage dataset/color/
        color_dir = None
        for root, dirs, files in os.walk(path):
            if 'color' in dirs:
                color_dir = os.path.join(root, 'color')
                break

        if color_dir and os.path.exists(color_dir):
            # Copy color images to our data directory
            if os.path.exists(config.DATA_DIR):
                shutil.rmtree(config.DATA_DIR)
            shutil.copytree(color_dir, config.DATA_DIR)
            print(f"[INFO] Color images copied to: {config.DATA_DIR}")
        else:
            # If no 'color' subfolder, try to find any directory with class folders
            for root, dirs, files in os.walk(path):
                if len(dirs) > 10:  # Likely the class directory
                    if os.path.exists(config.DATA_DIR):
                        shutil.rmtree(config.DATA_DIR)
                    shutil.copytree(root, config.DATA_DIR)
                    print(f"[INFO] Dataset copied to: {config.DATA_DIR}")
                    break

        # Final Validation: Check if we actually have data now
        if not os.path.exists(config.DATA_DIR) or len(os.listdir(config.DATA_DIR)) == 0:
            print("\n[ERROR] Download appeared to succeed but 'dataset' folder is empty.")
            print("[HELP] Please check your internet connection and Kaggle disk space.")
            sys.exit(1)

        # Print dataset statistics
        print_dataset_stats()
        return config.DATA_DIR

    except Exception as e:
        print(f"\n[ERROR] Failed to download dataset: {e}")
        print("\n[HELP] Make sure you have configured your Kaggle API key:")
        print("  1. Go to https://www.kaggle.com/settings")
        print("  2. Click 'Create New Token' under the API section")
        print("  3. Place the downloaded 'kaggle.json' file at:")
        print(f"     C:\\Users\\{os.getlogin()}\\.kaggle\\kaggle.json")
        sys.exit(1)


def print_dataset_stats():
    """Print statistics about the downloaded dataset."""
    print("\n" + "=" * 60)
    print("  DATASET STATISTICS")
    print("=" * 60)

    if not os.path.exists(config.DATA_DIR):
        print("[ERROR] Dataset directory not found!")
        return

    classes = sorted([
        d for d in os.listdir(config.DATA_DIR)
        if os.path.isdir(os.path.join(config.DATA_DIR, d))
    ])

    total_images = 0
    print(f"\n{'Class Name':<45} {'Images':>8}")
    print("-" * 55)

    for cls in classes:
        cls_dir = os.path.join(config.DATA_DIR, cls)
        num_images = len([
            f for f in os.listdir(cls_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ])
        total_images += num_images
        print(f"  {cls:<43} {num_images:>8}")

    print("-" * 55)
    print(f"  {'TOTAL':<43} {total_images:>8}")
    print(f"  {'Number of Classes':<43} {len(classes):>8}")
    print("=" * 60)


if __name__ == "__main__":
    download_dataset()

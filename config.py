"""
Configuration file for Plant Leaf Classification Project.
All hyperparameters and paths are centralized here.
"""
import os

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Create directories if they don't exist
for d in [DATA_DIR, MODEL_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Dataset ─────────────────────────────────────────────────────────────────
# Kaggle dataset identifier for PlantVillage
KAGGLE_DATASET = "abdallahalidev/plantvillage-dataset"

# ─── Image Settings ──────────────────────────────────────────────────────────
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
BATCH_SIZE = 32

# ─── Training Hyperparameters ────────────────────────────────────────────────
EPOCHS = 25
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1            # 10% of data held out for final testing
DROPOUT_RATE = 0.5

# ─── Data Augmentation ──────────────────────────────────────────────────────
ROTATION_RANGE = 30
WIDTH_SHIFT = 0.2
HEIGHT_SHIFT = 0.2
SHEAR_RANGE = 0.2
ZOOM_RANGE = 0.2
HORIZONTAL_FLIP = True
BRIGHTNESS_RANGE = (0.8, 1.2)

# ─── Model ───────────────────────────────────────────────────────────────────
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "plant_classifier_best.keras")
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "plant_classifier_final.keras")

# ─── Early Stopping / Callbacks ─────────────────────────────────────────────
PATIENCE = 5
MIN_DELTA = 0.001

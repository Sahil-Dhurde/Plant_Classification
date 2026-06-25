"""
Step 2: Data Preprocessing & Augmentation Pipeline.
Handles loading, splitting, and augmenting the plant leaf images.
"""
import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def get_class_names():
    """Get sorted list of class names from the dataset directory."""
    if not os.path.exists(config.DATA_DIR):
        raise FileNotFoundError(
            f"Dataset not found at {config.DATA_DIR}. "
            "Run download_dataset.py first!"
        )
    classes = sorted([
        d for d in os.listdir(config.DATA_DIR)
        if os.path.isdir(os.path.join(config.DATA_DIR, d))
    ])
    return classes


def load_data():
    """
    Load all image paths and labels from the dataset directory.
    Returns: (image_paths, labels, class_names)
    """
    class_names = get_class_names()
    image_paths = []
    labels = []

    for idx, cls_name in enumerate(class_names):
        cls_dir = os.path.join(config.DATA_DIR, cls_name)
        for img_file in os.listdir(cls_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_paths.append(os.path.join(cls_dir, img_file))
                labels.append(idx)

    print(f"[INFO] Loaded {len(image_paths)} images across {len(class_names)} classes")
    
    if len(image_paths) == 0:
        print("\n[ERROR] No images found in the 'dataset' folder!")
        print("[HELP] Please re-run 'python download_dataset.py' or choose Option 1 in the menu.")
        print("[HELP] Ensure the download completes without interruption.\n")
        sys.exit(1)
        
    return np.array(image_paths), np.array(labels), class_names


def create_datasets():
    """
    Create train, validation, and test datasets using tf.data pipeline.
    Returns: (train_ds, val_ds, test_ds, class_names)
    """
    image_paths, labels, class_names = load_data()
    num_classes = len(class_names)

    # Split: Train + Val | Test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        image_paths, labels,
        test_size=config.TEST_SPLIT,
        random_state=42,
        stratify=labels
    )

    # Split: Train | Val
    val_ratio = config.VALIDATION_SPLIT / (1 - config.TEST_SPLIT)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_ratio,
        random_state=42,
        stratify=y_trainval
    )

    print(f"[INFO] Training samples:   {len(X_train)}")
    print(f"[INFO] Validation samples: {len(X_val)}")
    print(f"[INFO] Test samples:       {len(X_test)}")

    # Create tf.data datasets
    train_ds = create_tf_dataset(X_train, y_train, num_classes, augment=True)
    val_ds = create_tf_dataset(X_val, y_val, num_classes, augment=False)
    test_ds = create_tf_dataset(X_test, y_test, num_classes, augment=False)

    return train_ds, val_ds, test_ds, class_names


def parse_image(file_path, label):
    """Read and preprocess a single image."""
    # Read file
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    # Resize
    img = tf.image.resize(img, config.IMG_SIZE)
    # Normalize to [0, 1]
    img = img / 255.0
    return img, label


def augment_image(image, label):
    """Apply data augmentation to a single image."""
    # Random horizontal flip
    image = tf.image.random_flip_left_right(image)
    # Random vertical flip
    image = tf.image.random_flip_up_down(image)
    # Random brightness
    image = tf.image.random_brightness(image, max_delta=0.2)
    # Random contrast
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    # Random saturation
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    # Random rotation (approximate with crop and resize)
    image = tf.image.random_crop(
        tf.image.resize(image, [config.IMG_HEIGHT + 30, config.IMG_WIDTH + 30]),
        size=[config.IMG_HEIGHT, config.IMG_WIDTH, 3]
    )
    # Clip values to [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


def create_tf_dataset(file_paths, labels, num_classes, augment=False):
    """Create a tf.data.Dataset from file paths and labels."""
    # One-hot encode labels
    labels_onehot = tf.keras.utils.to_categorical(labels, num_classes)

    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels_onehot))
    dataset = dataset.shuffle(buffer_size=len(file_paths)) if augment else dataset
    dataset = dataset.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(config.BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


if __name__ == "__main__":
    print("=" * 60)
    print("  DATA PREPROCESSING TEST")
    print("=" * 60)
    train_ds, val_ds, test_ds, class_names = create_datasets()
    print(f"\n[INFO] Number of classes: {len(class_names)}")
    print(f"[INFO] Classes: {class_names[:5]}... (showing first 5)")

    # Test a batch
    for images, labels in train_ds.take(1):
        print(f"\n[INFO] Batch shape: {images.shape}")
        print(f"[INFO] Labels shape: {labels.shape}")
        print(f"[INFO] Image dtype: {images.dtype}")
        print(f"[INFO] Image value range: [{images.numpy().min():.3f}, {images.numpy().max():.3f}]")

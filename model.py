"""
Step 3: CNN Model Architecture for Plant Leaf Classification.
Provides both a custom CNN and a transfer learning model (MobileNetV2).
"""
import os
import sys
import tensorflow as tf
from tensorflow.keras import layers, models, applications

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def build_custom_cnn(num_classes):
    """
    Build a custom CNN architecture from scratch.
    Architecture: 4 Conv blocks + Global Average Pooling + Dense layers.
    """
    model = models.Sequential(name="PlantLeaf_CustomCNN")

    # ─── Block 1 ─────────────────────────────────────────────────────────
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu',
                            input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # ─── Block 2 ─────────────────────────────────────────────────────────
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # ─── Block 3 ─────────────────────────────────────────────────────────
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # ─── Block 4 ─────────────────────────────────────────────────────────
    model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # ─── Classifier Head ─────────────────────────────────────────────────
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(config.DROPOUT_RATE))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(config.DROPOUT_RATE))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model


def build_transfer_model(num_classes):
    """
    Build a transfer learning model using MobileNetV2 as the base.
    MobileNetV2 is lightweight and works great for leaf classification.
    """
    # Load pre-trained MobileNetV2 (without top classification layers)
    base_model = applications.MobileNetV2(
        input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 3),
        include_top=False,
        weights='imagenet'
    )

    # Freeze the base model layers initially
    base_model.trainable = False

    # Build the full model
    model = models.Sequential(name="PlantLeaf_MobileNetV2")
    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(config.DROPOUT_RATE))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(config.DROPOUT_RATE))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model


def compile_model(model, learning_rate=None):
    """Compile the model with optimizer, loss, and metrics."""
    lr = learning_rate or config.LEARNING_RATE

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def get_callbacks():
    """Get training callbacks: EarlyStopping, ModelCheckpoint, ReduceLR."""
    callbacks = [
        # Save the best model based on validation accuracy
        tf.keras.callbacks.ModelCheckpoint(
            filepath=config.MODEL_SAVE_PATH,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        # Stop training if no improvement
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=config.PATIENCE,
            min_delta=config.MIN_DELTA,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate on plateau
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
    ]
    return callbacks


def unfreeze_base_model(model, num_layers_to_unfreeze=30):
    """
    Unfreeze the last N layers of the base model for fine-tuning.
    Used after initial training with frozen base.
    """
    # The base model is the first layer in our Sequential model
    base = model.layers[0]
    base.trainable = True

    # Freeze all layers except the last N
    for layer in base.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False

    print(f"[INFO] Unfroze last {num_layers_to_unfreeze} layers of base model")
    print(f"[INFO] Total trainable weights: {len(model.trainable_weights)}")
    return model


if __name__ == "__main__":
    print("=" * 60)
    print("  MODEL ARCHITECTURE SUMMARY")
    print("=" * 60)

    num_classes = 38  # PlantVillage has 38 classes

    print("\n📌 Custom CNN Model:")
    print("-" * 60)
    custom_model = build_custom_cnn(num_classes)
    custom_model = compile_model(custom_model)
    custom_model.summary()

    print("\n\n📌 Transfer Learning Model (MobileNetV2):")
    print("-" * 60)
    transfer_model = build_transfer_model(num_classes)
    transfer_model = compile_model(transfer_model)
    transfer_model.summary()

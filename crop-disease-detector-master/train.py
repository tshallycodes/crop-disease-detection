"""
Train EfficientNetV2B0 on PlantVillage dataset.
Run once before starting the web app.
"""
import os, json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ── Config ────────────────────────────────────────────────
DATA_DIR   = os.path.expanduser('~/Documents/PlantVillage/PlantVillage')
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
EPOCHS     = 15
SEED       = 42

# ── Load datasets ─────────────────────────────────────────
print("[INFO] Loading dataset...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset='training',
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset='validation',
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"[INFO] Classes: {num_classes} — {class_names}")

# Save class names
with open('class_names.json', 'w') as f:
    json.dump(class_names, f)

# ── Performance optimisation ──────────────────────────────
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds   = val_ds.prefetch(AUTOTUNE)

# ── Data augmentation ─────────────────────────────────────
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# ── Build model ───────────────────────────────────────────
print("[INFO] Building model...")
base_model = EfficientNetV2B0(
    include_top=False,
    weights='imagenet',
    pooling='avg'
)
base_model.trainable = False

inputs  = tf.keras.Input(shape=(*IMG_SIZE, 3))
x       = data_augmentation(inputs)
x       = preprocess_input(x)
x       = base_model(x, training=False)
x       = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)
model   = Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ── Train ─────────────────────────────────────────────────
callbacks = [
    ModelCheckpoint('crop_disease_model.keras', save_best_only=True, monitor='val_accuracy', verbose=1),
    EarlyStopping(patience=4, restore_best_weights=True, monitor='val_accuracy', verbose=1)
]

print("[INFO] Training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

print(f"\n✅ Training complete!")
print(f"   Best val accuracy: {max(history.history['val_accuracy']):.2%}")
print(f"   Classes: {num_classes}")
print(f"   Model saved: crop_disease_model.keras")

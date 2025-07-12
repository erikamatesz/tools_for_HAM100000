#!/usr/bin/env python3
"""
train_model.py

Builds and trains a CNN on the HAM10000 dataset using the generators
prepared by `load_preprocess.py`.

Key points
----------
* **NO direct `from tensorflow.keras.applications import …` import.**  
  We use the safest style: `import tensorflow as tf` and then
  `tf.keras.applications.MobileNetV2` so it works across TF/Keras package
  layouts.
* Reads CLI arguments for batch‑size, epochs, etc., but falls back to
  sensible defaults so you can just run `python train_model.py`.
* Saves the trained model to disk (`ham10000_mobilenetv2.h5`).

Usage example
-------------
```bash
python train_model.py \
    --csv HAM10000_metadata.csv \
    --images HAM10000_224 \
    --batch 32 \
    --epochs 15
```

Author: Erika C. Matesz Bueno
Date: Jul 2025
"""

import argparse
from pathlib import Path
import tensorflow as tf
from load_preprocess import load_generators

# -------------------------------------------------------------
# CLI
# -------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train MobileNetV2 on HAM10000.")
    p.add_argument('--csv', default='HAM10000_metadata.csv', type=Path, help='CSV metadata file')
    p.add_argument('--images', default='HAM10000_224', type=Path, help='Root folder with processed images')
    p.add_argument('--batch', type=int, default=32, help='Batch size (default 32)')
    p.add_argument('--epochs', type=int, default=15, help='Training epochs (default 15)')
    p.add_argument('--val_split', type=float, default=0.2, help='Validation split (default 0.2)')
    p.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate (default 1e-3)')
    return p.parse_args()


# -------------------------------------------------------------
# Build model helper
# -------------------------------------------------------------

def build_model(num_classes: int, lr: float = 1e-3) -> tf.keras.Model:
    """Constructs a MobileNetV2‑based classifier and compiles it."""
    base = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet',
    )
    base.trainable = False  # freeze for initial training phase

    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    out = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=base.input, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------

def main():
    args = parse_args()

    # 1️⃣  Load generators ----------------------------------------------------
    train_gen, val_gen, stats = load_generators(
        csv_path=args.csv,
        images_root=args.images,
        img_size=(224, 224),
        batch_size=args.batch,
        val_split=args.val_split,
    )

    # 2️⃣  Build model --------------------------------------------------------
    model = build_model(num_classes=stats['num_classes'], lr=args.lr)
    model.summary()

    # 3️⃣  Train --------------------------------------------------------------
    history = model.fit(
        train_gen,
        epochs=args.epochs,
        steps_per_epoch=stats['train_steps'],
        validation_data=val_gen,
        validation_steps=stats['val_steps'],
    )

    # 4️⃣  Save ---------------------------------------------------------------
    model.save('ham10000_mobilenetv2.h5')
    print("\nModel saved as ham10000_mobilenetv2.h5")


if __name__ == '__main__':
    main()

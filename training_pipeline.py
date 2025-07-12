"""
training_pipeline.py

Trains a single multi-input deep learning model on the HAM10000 dataset,
combining image data and clinical metadata.

Key Features:
-------------
1. Loads image paths and clinical metadata from a CSV file.
2. Applies data augmentation (flip, rotation) and rescaling to images.
3. Normalizes and one-hot encodes clinical features (age, biological sex, 
   localization).
4. Supports multiple model architectures (Dense, CNN, ResNet50, LSTM),
   but only one model is trained per execution.
5. Saves the checkpoint of the model that achieves the lowest validation loss
   (val_loss) during the training epochs, along with training logs (CSV).

Important:
----------
- The script trains one model at a time, based on the architecture specified via
  the `--model_name` argument.
- To evaluate different architectures, run the script multiple times, once for each
  model, and manually compare the generated training logs.
- The `ModelCheckpoint` callback monitors the validation loss throughout all epochs
  in a single run and saves the best model checkpoint accordingly.
- For this mechanism to be effective, set the `--epochs` argument to more than 1.
  If training runs for only 1 epoch, the saved model will be the single trained model
  without any comparison.

Usage example:

python train_models.py \
    --csv metadata.csv \
    --images HAM10000_224 \
    --model_output my_model \
    --model_name ConvNet_3ConvLayers \
    --epochs 30 \
    --batch_size 64 \
    --val_split 0.2

CLI Arguments
-------------

--csv           Path to the HAM10000 CSV metadata file.  
--images        Root directory containing processed images (e.g., `HAM10000_224`).  
--size          Square image size. Default is 224.  
--batch_size    Number of samples per batch. Default is 32.  
--val_split     Fraction of the dataset used for validation. Default is 0.2.  
--epochs        Number of training epochs. Default is 20.  
--model_output  Base filename for saving the trained model.  
--model_name    Architecture to use. Options:  
               - Dense_2Layers  
               - Dense_5Layers  
               - ConvNet_3ConvLayers  
               - ConvNet_6ConvLayers  
               - ResNet50  
               - LSTM  

Outputs
-------

- Trained Keras model (saved as `.keras`) representing the best checkpoint
  from the training epochs based on lowest validation loss.
- Training logs in CSV format, including the model name on each row.
- Console model summary and training progress reports.

Notes
-----

- Image preprocessing includes horizontal flipping and slight rotation (for training only).
- Clinical features include normalized age, and one-hot encoded biological sex and localization.
- The model combines both image and clinical inputs using concatenation before the final classification layer.
- To compare different model architectures, run this script separately for each one and compare their logs manually.
"""

import argparse
from pathlib import Path
from datetime import datetime
import csv

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# shortcut aliases for common Keras components
layers = tf.keras.layers
models = tf.keras.models
optimizers = tf.keras.optimizers
callbacks = tf.keras.callbacks
Sequence = tf.keras.utils.Sequence


def load_data(csv_path: Path, images_root: Path, img_size=(224, 224), val_split=0.2, batch_size=32):
    df = pd.read_csv(csv_path)

    # maps each image_id to the appropriate folder path (part_1 or part_2)
    def map_image_path(image_id: str):
        part1 = images_root / 'HAM10000_images_part_1' / f'{image_id}.jpg'
        if part1.exists():
            return f'HAM10000_images_part_1/{image_id}.jpg'
        part2 = images_root / 'HAM10000_images_part_2' / f'{image_id}.jpg'
        if part2.exists():
            return f'HAM10000_images_part_2/{image_id}.jpg'
        return None

    df['image_path'] = df['image_id'].apply(map_image_path)
    df = df.dropna(subset=['image_path'])  # remove rows without valid image path

    labels = sorted(df['dx'].unique())  # extract sorted list of class labels

    # stratified split ensures label balance across train and validation sets
    train_df, val_df = train_test_split(df, test_size=val_split, stratify=df['dx'], random_state=42)

    # image data augmentation for training
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255,
        horizontal_flip=True,
        rotation_range=15,
    )

    # only rescaling for validation images
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

    # generate batches of augmented image data from dataframe
    train_gen = train_datagen.flow_from_dataframe(
        train_df,
        directory=str(images_root),
        x_col='image_path',
        y_col='dx',
        target_size=img_size,
        class_mode='categorical',
        classes=labels,
        batch_size=batch_size,
        shuffle=True,
        seed=42,
    )

    val_gen = val_datagen.flow_from_dataframe(
        val_df,
        directory=str(images_root),
        x_col='image_path',
        y_col='dx',
        target_size=img_size,
        class_mode='categorical',
        classes=labels,
        batch_size=batch_size,
        shuffle=False,
        seed=42,
    )

    # prepares tabular clinical data (age, biological sex, localization)
    def preprocess_clinical_features(df_subset):
        df_subset = df_subset.copy()
        df_subset['age'] = df_subset['age'].fillna(df_subset['age'].mean())  # missing age
        scaler = StandardScaler()
        age_norm = scaler.fit_transform(df_subset[['age']])
        sex_ohe = pd.get_dummies(df_subset['sex'])  # one-hot encode biological sex
        if 'unknown' in sex_ohe.columns:
            sex_ohe = sex_ohe.drop(columns=['unknown'])  # drop unknown
        loc_ohe = pd.get_dummies(df_subset['localization'])  # one-hot encode localization
        clinical_features = np.hstack([age_norm, sex_ohe.values, loc_ohe.values])
        return clinical_features

    train_clinical = preprocess_clinical_features(train_df)
    val_clinical = preprocess_clinical_features(val_df)

    return train_gen, val_gen, train_clinical, val_clinical, labels


class MultiInputGenerator(Sequence):
    # custom generator that puts image and clinical data together
    def __init__(self, image_gen, clinical_data, image_input_name='input_layer', **kwargs):
        super().__init__(**kwargs)
        self.image_gen = image_gen
        self.clinical_data = clinical_data
        self.batch_size = image_gen.batch_size
        self.image_input_name = image_input_name

    def __len__(self):
        return len(self.image_gen)

    def __getitem__(self, index):
        imgs, labels = self.image_gen[index]
        batch_start = index * self.batch_size
        batch_end = batch_start + imgs.shape[0]
        clinical_batch = self.clinical_data[batch_start:batch_end]

        # prepare dictionary of inputs matching model's input layer names
        inputs_dict = {
            self.image_input_name: imgs.astype(np.float32),
            "clinical_input": clinical_batch.astype(np.float32),
        }
        return inputs_dict, labels

    def on_epoch_end(self):
        self.image_gen.on_epoch_end()  # ensure shuffling if enabled


def build_model(model_name, num_classes, clinical_input_dim, img_size=(224, 224, 3)):
    # selects model architecture based on the name passed

    if model_name == 'Dense_2Layers':
        # 2 dense layers
        image_input = layers.Input(shape=img_size, name='input_layer')
        x = layers.Flatten()(image_input)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dense(64, activation='relu')(x)
        image_input_name = image_input.name.split(':')[0]

    elif model_name == 'Dense_5Layers':
        # 5 dense layers
        image_input = layers.Input(shape=img_size, name='input_layer')
        x = layers.Flatten()(image_input)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dense(16, activation='relu')(x)
        image_input_name = image_input.name.split(':')[0]

    elif model_name == 'ConvNet_3ConvLayers':
        # cnn with 3 convolutional layers
        image_input = layers.Input(shape=img_size, name='input_layer')
        x = layers.Conv2D(32, (3, 3), activation='relu')(image_input)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(128, (3, 3), activation='relu')(x)
        x = layers.GlobalAveragePooling2D()(x)
        image_input_name = image_input.name.split(':')[0]

    elif model_name == 'ConvNet_6ConvLayers':
        # cnn with 6 conv layers in repeated blocks
        image_input = layers.Input(shape=img_size, name='input_layer')
        x = layers.Conv2D(32, (3, 3), activation='relu')(image_input)
        x = layers.Conv2D(32, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(128, (3, 3), activation='relu')(x)
        x = layers.Conv2D(128, (3, 3), activation='relu')(x)
        x = layers.GlobalAveragePooling2D()(x)
        image_input_name = image_input.name.split(':')[0]

    elif model_name == 'ResNet50':
        # pretrained resnet50
        image_input = layers.Input(shape=img_size, name='input_layer')
        base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_tensor=image_input)
        base_model.trainable = False
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        image_input_name = image_input.name.split(':')[0]

    elif model_name == 'LSTM':
        # reshape image to sequence format and feed into lstm
        image_input = layers.Input(shape=img_size, name='input_layer')
        x = layers.Reshape((img_size[0], img_size[1]*img_size[2]))(image_input)
        x = layers.LSTM(64)(x)
        image_input_name = image_input.name.split(':')[0]

    else:
        raise ValueError(f"Unknown model_name '{model_name}'. Supported: Dense_2Layers, Dense_5Layers, ConvNet_3ConvLayers, ConvNet_6ConvLayers, ResNet50, LSTM")

    # clinical input processing (dense + dropout)
    clinical_input = layers.Input(shape=(clinical_input_dim,), name='clinical_input')
    y = layers.Dense(32, activation='relu')(clinical_input)
    y = layers.Dropout(0.3)(y)

    # fuse image and clinical branches
    concatenated = layers.concatenate([x, y])
    z = layers.Dense(64, activation='relu')(concatenated)
    z = layers.Dropout(0.3)(z)
    output = layers.Dense(num_classes, activation='softmax')(z)

    # create final model
    model = models.Model(inputs=[image_input, clinical_input], outputs=output)
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model, image_input_name


class CSVLoggerWithModelName(tf.keras.callbacks.CSVLogger):
    # custom csv logger that includes the model name in every row
    def __init__(self, filename, model_name, **kwargs):
        super().__init__(filename, **kwargs)
        self.model_name = model_name
        self.append = False

    def on_train_begin(self, logs=None):
        self.append = False
        super().on_train_begin(logs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # write header once with added 'model_name' field
        if not self.append:
            keys = list(logs.keys())
            keys.append('model_name')
            self.writer = csv.DictWriter(self.csv_file, fieldnames=keys)
            self.writer.writeheader()
            self.append = True

        row = logs.copy()
        row['model_name'] = self.model_name
        self.writer.writerow(row)
        self.csv_file.flush()


def main(args):
    print("Loading data...")
    train_gen, val_gen, train_clinical, val_clinical, labels = load_data(
        csv_path=Path(args.csv),
        images_root=Path(args.images),
        img_size=(args.size, args.size),
        val_split=args.val_split,
        batch_size=args.batch_size,
    )
    print(f"Classes: {labels}")

    print("Building model...")
    model, image_input_name = build_model(
        model_name=args.model_name,
        num_classes=len(labels),
        clinical_input_dim=train_clinical.shape[1],
        img_size=(args.size, args.size, 3)
    )
    model.summary()

    # generate output model file name if extension is missing
    model_name = args.model_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = Path(args.model_output)
    if not model_path.suffix:
        model_path = Path(f"{args.model_output}_{timestamp}.keras")

    csv_logger = CSVLoggerWithModelName(f'training_log_{model_name}.csv', model_name=model_name)

    # training callbacks
    cb = [
        callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        callbacks.ModelCheckpoint(str(model_path), save_best_only=True),
        csv_logger,
    ]

    # wrap generators to feed both image and clinical inputs
    train_multi_gen = MultiInputGenerator(train_gen, train_clinical, image_input_name=image_input_name)
    val_multi_gen = MultiInputGenerator(val_gen, val_clinical, image_input_name=image_input_name)

    print("Training model...")
    model.fit(
        train_multi_gen,
        validation_data=val_multi_gen,
        epochs=args.epochs,
        callbacks=cb,
        verbose=1,
    )

    print(f"Model saved to {model_path}")


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description="Train HAM10000 model combining images and clinical data.")
    parser.add_argument('--csv', required=True, help="CSV metadata file path")
    parser.add_argument('--images', required=True, help="Root folder with processed images (HAM10000_224)")
    parser.add_argument('--size', type=int, default=224, help="Image size (square)")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--val_split', type=float, default=0.2, help="Validation split fraction")
    parser.add_argument('--epochs', type=int, default=20, help="Number of epochs")
    parser.add_argument('--model_output', default='model', help="Base name for saved model (no extension needed)")
    parser.add_argument('--model_name', default='ConvNet_3ConvLayers', 
                        help="Model architecture name: Dense_2Layers, Dense_5Layers, ConvNet_3ConvLayers, ConvNet_6ConvLayers, ResNet50, LSTM")
    args = parser.parse_args()

    main(args)

# Training Pipeline for HAM10000

Tooling to train deep learning models on the **HAM10000** skin-lesion dataset by combining image data and structured clinical metadata.

## Overview

`training_pipeline.py` trains a multi-input model using both dermoscopic images and patient metadata (age, biological sex, and lesion localization). The script supports a variety of architectures (Dense, CNN, ResNet50, LSTM), and each execution trains a single model variant. The best model checkpoint is saved automatically based on validation loss.

## Features

* **Image and metadata fusion:** Uses both image data and clinical attributes for better classification.
* **Multiple architectures:** Supports Dense, ConvNet, ResNet50, and LSTM.
* **ModelCheckpoint support:** Automatically saves the best model (lowest validation loss).
* **Custom training log:** Training progress saved as CSV with model name per row.
* **Flexible CLI interface:** Control model architecture, batch size, epochs, and more via arguments.

## Requirements

```
tensorflow
pandas
numpy
scikit-learn
Pillow
```

Install with `pip install tensorflow pandas numpy scikit-learn pillow` or `pip install -r requirements.txt`.

## Usage

```bash
python training_pipeline.py \
    --csv metadata.csv \
    --images HAM10000_224 \
    --model_output my_model \
    --model_name ConvNet_3ConvLayers \
    --epochs 30 \
    --batch_size 64 \
    --val_split 0.2
```

## Arguments

| Flag            | Description                                                                 | Default                  |
|------------------|------------------------------------------------------------------------------|---------------------------|
| `--csv`          | Path to the CSV metadata file (e.g., `metadata.csv`).                        | *Required*                |
| `--images`       | Root directory of processed images (e.g., `HAM10000_224`).                  | *Required*                |
| `--size`         | Square image size used in training (e.g., 224 for 224x224).                 | `224`                     |
| `--batch_size`   | Number of samples per training batch.                                       | `32`                      |
| `--val_split`    | Fraction of data reserved for validation.                                   | `0.2`                     |
| `--epochs`       | Number of training epochs.                                                   | `20`                      |
| `--model_output` | Base name for saving the best trained model. Extension added automatically. | `model`                   |
| `--model_name`   | Architecture to train. Options:                                              | `ConvNet_3ConvLayers`     |
|                  | - `Dense_2Layers`                                                           |                           |
|                  | - `Dense_5Layers`                                                           |                           |
|                  | - `ConvNet_3ConvLayers`                                                     |                           |
|                  | - `ConvNet_6ConvLayers`                                                     |                           |
|                  | - `ResNet50`                                                                |                           |
|                  | - `LSTM`                                                                    |                           |

## Outputs

- Trained Keras model (`.keras` format), saved at best epoch according to validation loss.
- Training log in CSV format with fields: epoch, accuracy, loss, val_accuracy, val_loss, and model name.
- Model architecture printed to the console via `model.summary()`.

## Model Architectures

| Model Name           | Description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| `Dense_2Layers`      | Two fully connected layers on image input.                                 |
| `Dense_5Layers`      | Deeper version with five dense layers.                                     |
| `ConvNet_3ConvLayers`| CNN with 3 convolutional + pooling blocks.                                 |
| `ConvNet_6ConvLayers`| CNN with 6 convolutional layers in repeated blocks.                        |
| `ResNet50`           | Pre-trained ResNet50 from ImageNet (frozen weights).                       |
| `LSTM`               | Treats image as a sequence and uses LSTM on flattened spatial dimensions.  |

## Example Directory Layout

```
project_root/
│
├─ metadata.csv           # clinical + label metadata
├─ HAM10000_224/                   # preprocessed square images (from prepare_images.py)
│   ├─ HAM10000_images_part_1/
│   └─ HAM10000_images_part_2/
└─ training_pipeline.py
```

## Notes

- Image augmentation includes horizontal flipping and light rotation during training only.
- Age is normalized using `StandardScaler`; sex and localization are one-hot encoded.
- Clinical features are processed via a small dense network before being concatenated with image features.
- To compare architectures, run the script multiple times with different `--model_name` values and compare logs.

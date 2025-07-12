# Tools for HAM10000

This repository contains the tools developed for the final project of the Deep Learning course, part of the Master’s program in Information Systems at UNIRIO. 

In addition to supporting the completion of the final project, it is also intended to help anyone interested in studying the topic and working with the same dataset, by providing reusable code and a Jupyter notebook. The notebook was used to perform exploratory data analysis, helping to understand the structure and distribution of the dataset before modeling. The mentioned notebook is located in the `exploratory_analysis` directory.

**Important:** The original HAM10000 dataset images, as well as the processed and resized images (224×224 pixels), will NOT be included in this repository due to their large size and licensing restrictions. The original dataset images can be downloaded directly from [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000).

This repository provides all the code and tools necessary to preprocess and work with these images once you have obtained them.

## 📊 Dataset Columns Explanation

- **lesion_id**  
  Unique identifier for the lesion itself. Multiple images can belong to the same lesion, so this ID groups images of the same lesion.

- **image_id**  
  Unique identifier for the image, usually matching the image filename.

- **dx** (Diagnosis)  
  Short code representing the diagnosis of the lesion. Possible values include:
  - `nv` : Melanocytic nevus (benign mole)
  - `mel` : Melanoma
  - `bkl` : Benign keratosis-like lesions (solar lentigo, seborrheic keratosis, lichen planus-like keratosis)
  - `bcc` : Basal cell carcinoma
  - `akiec` : Actinic keratoses and intraepithelial carcinoma / Bowen's disease
  - `vasc` : Vascular lesions (angiomas, angiokeratomas, pyogenic granulomas, hemorrhage)
  - `df` : Dermatofibroma

- **dx_type** (Diagnosis Type)  
  Indicates how the diagnosis was confirmed. Possible values:
  - `histo` : Diagnosis confirmed by histopathological examination (most reliable)
  - `follow_up` : Diagnosis confirmed by follow-up
  - `consensus` : Diagnosis determined by expert consensus
  - `confocal` : Diagnosis based on confocal microscopy examination

- **age**  
  Patient's age in years at the time the image/lesion was collected.

- **sex**  
  Patient's biological sex, `male` or `female`. There are some `unknown` entries.

- **localization**  
  Anatomical localization of the lesion.

## 🗂️ Dataset Source

[Skin Cancer MNIST: HAM10000 @ Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

## 💻 Env Setup

Create and activate a virtual env, and run `pip install -r requirements.txt` to install dependencies.

## 🛠️ Tools

[Image Processor Documentation](IMAGE_PREPROCESSOR.md)

[Training Pipeline Documentation](TRAINING_PIPELINE.md)

## 📝 License

This repository contains materials developed for the final project of the Deep Learning course at UNIRIO. 

While it was originally created to fulfill the project submission requirements, it is shared here as a contribution to the community. Anyone is free to use, modify, and build upon the content provided.

By making this available, the goal is to support learning and further research in the field, encouraging reuse and adaptation beyond the original academic context.

Enjoy it 😉
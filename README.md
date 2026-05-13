# Cancer Detection AI

A professional cancer detection system built with deep learning, transfer learning, and explainability. This repository is designed to be modular, beginner-friendly, and production-ready.

## Features

- Medical image cancer detection pipeline
- Data preprocessing and augmentation for image-based datasets
- Transfer learning with EfficientNet backbone
- Model training with early stopping and checkpoint saving
- Validation metrics: Accuracy, Precision, Recall, F1-score, AUC
- Confusion matrix plotting and classification reports
- Grad-CAM explainability for predictions
- Custom image upload inference
- Streamlit UI for interactive prediction
- Config-based architecture and centralized logging
- GPU support with TensorFlow if available

## Tech Stack

- Python
- TensorFlow
- OpenCV
- Streamlit
- scikit-learn
- Matplotlib / Seaborn
- YAML configuration
- Loguru logging

## Repository Structure

```text
cancer_detection_ai/
│
├── data/
├── notebooks/
├── src/
│   ├── preprocessing/
│   ├── models/
│   ├── training/
│   ├── inference/
│   ├── visualization/
│   └── utils/
├── tests/
├── outputs/
├── configs/
├── requirements.txt
├── README.md
├── CONTRIBUTING.md
├── LICENSE
├── .gitignore
├── train.py
├── predict.py
├── app.py
└── setup.py
```

## Project Overview

This project delivers a complete end-to-end AI solution for cancer detection from medical images. It includes dataset preprocessing, model training, evaluation, inference, and explainability using Grad-CAM.

## Dataset

This repository is compatible with public medical imaging datasets such as:

- HAM10000 skin lesion dataset
- Breast cancer histopathology dataset
- Lung CT scan datasets

### Recommended dataset layout

```text
data/
  train/
    benign/
    malignant/
  val/
    benign/
    malignant/
  test/
    benign/
    malignant/
```

### Dataset download steps

1. Download the HAM10000 dataset from the [Kaggle HAM10000 dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) or an equivalent public dataset.
2. Extract the image folders into `data/`.
3. Confirm that `configs/config.yaml` points to the correct dataset paths and class names.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/cancer-detection-ai.git
cd cancer-detection-ai
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

Open `configs/config.yaml` and update the values to match your environment:

- `data.base_dir`: dataset root directory
- `data.image_size`: image dimensions for model input
- `data.batch_size`: training batch size
- `training.epochs`: total number of epochs
- `training.patience`: early stopping patience
- `model.backbone`: transfer learning backbone model

## Training

Train the model using:

```bash
python train.py --config configs/config.yaml
```

### What the training script does

- Loads training and validation datasets
- Applies image augmentation
- Builds a transfer learning model
- Trains with checkpointing and early stopping
- Saves final model weights and metrics
- Produces training history plots and confusion matrix output

## Prediction

Run inference on a single image:

```bash
python predict.py --config configs/config.yaml --image_path path/to/image.jpg
```

Example output:

```text
Predicted label: malignant
Confidence: 0.9123
```

## Streamlit UI

Launch the interactive app:

```bash
streamlit run app.py
```

App capabilities:

- Upload medical images
- Display classification label and confidence
- Show Grad-CAM explainability overlay
- Support custom checkpoint paths

## Results

- Model checkpoints: `outputs/checkpoints/best_model.h5`
- Final model export: `outputs/final_model.h5`
- Confusion matrix image: `outputs/confusion_matrix.png`
- Training history plot: `outputs/training_history.png`
- Grad-CAM overlay: `outputs/gradcam.png`
- Classification report: `outputs/classification_report.txt`

### Screenshot placeholders

Add screenshots after running the pipeline:

- `outputs/screenshots/app_ui.png`
- `outputs/screenshots/confusion_matrix.png`
- `outputs/screenshots/gradcam_overlay.png`

## Deployment

### Streamlit Cloud

1. Push the repository to GitHub.
2. Create a new app on Streamlit Cloud.
3. Set the repository and branch.
4. Use `streamlit run app.py` as the start command.

### Render

1. Connect the GitHub repository to Render.
2. Set the build command to `pip install -r requirements.txt`.
3. Set the start command to `streamlit run app.py`.

### Hugging Face Spaces

1. Create a new Space with the Streamlit SDK.
2. Upload repository files.
3. Use `streamlit run app.py` as the startup command.

## Contribution

Contributions are welcome and appreciated. Please follow these steps:

1. Fork the repository.
2. Create a feature branch:
   `git checkout -b feature/description`
3. Add tests and documentation.
4. Open a pull request.

For more details, see `CONTRIBUTING.md`.

## Future Improvements

- Multi-class cancer subtype classification
- Support for ResNet and MobileNet backbones
- Cross-validation and hyperparameter tuning
- API deployment with FastAPI and Docker
- Extended explainability dashboards
- Additional dataset pipelines and preprocessing options

## License

This project is released under the MIT License.

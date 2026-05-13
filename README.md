# Cancer Detection AI

A professional cancer detection system built with deep learning, transfer learning, and explainability. This repository is designed to be modular, beginner-friendly, and production-ready.

## Features

- Medical image cancer classification pipeline
- Dataset preprocessing and augmentation
- Transfer learning with EfficientNet backbone
- Model training, validation, and checkpointing
- Accuracy, Precision, Recall, F1-score reporting
- Confusion matrix and training metrics visualization
- Grad-CAM explainability for model predictions
- Custom image upload inference
- Streamlit UI for interactive prediction
- Config-based architecture and logging
- GPU support when available

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
├── .gitignore
├── train.py
├── predict.py
├── app.py
└── setup.py
```

## Dataset

This project is compatible with public medical imaging datasets such as:

- HAM10000 skin lesion dataset
- Breast Cancer Histopathology dataset
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

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/cancer-detection-ai.git
cd cancer-detection-ai
```

2. Create a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

Update `configs/config.yaml` with dataset directories, training parameters, and model settings.

## Training

```bash
python train.py --config configs/config.yaml
```

## Prediction

```bash
python predict.py --config configs/config.yaml --image_path path/to/image.jpg
```

## Streamlit UI

```bash
streamlit run app.py
```

## Results

- Training and validation metrics are saved to `outputs/`
- Model checkpoint files are saved to `outputs/checkpoints/`
- Visualizations are saved during evaluation and inference

## Deployment

Deploy to:

- Streamlit Cloud
- Render
- Hugging Face Spaces

## Contribution

Contributions are welcome. Please open issues or pull requests and follow the contribution guide below.

## Future Improvements

- Add multi-class cancer subtype classification
- Add cross-validation and enriched transformers
- Add end-to-end explainable AI reporting
- Add API deployment with FastAPI
- Add AutoML and hyperparameter tuning

## License

This project is released under the MIT License.

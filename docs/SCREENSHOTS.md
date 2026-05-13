# Screenshot Placeholders

This project includes example screenshot placeholder references to help document results after running the pipeline.

Suggested screenshot file names:

- `outputs/screenshots/app_ui.png` — Streamlit user interface with uploaded image and prediction result.
- `outputs/screenshots/confusion_matrix.png` — Confusion matrix generated after evaluation.
- `outputs/screenshots/gradcam_overlay.png` — Grad-CAM overlay visualization for model explainability.
- `outputs/screenshots/training_history.png` — Training loss and accuracy plots.

To generate screenshot files:

1. Run the training pipeline: `python train.py --config configs/config.yaml`
2. Run the web app and capture the UI.
3. Save the generated plots into `outputs/screenshots/`.

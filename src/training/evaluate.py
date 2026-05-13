import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

from src.visualization.plots import plot_confusion_matrix


def evaluate_model(model, dataset, class_names, output_dir: Path, logger):
    """Evaluate a trained model and save metrics reports and confusion matrix."""
    if dataset is None:
        logger.warning("No validation or test dataset available for evaluation.")
        return

    logger.info("Evaluating model on dataset")
    y_true = []
    y_pred = []

    for batch_images, batch_labels in dataset:
        predictions = model.predict(batch_images, verbose=0)
        y_true.extend(batch_labels.numpy().tolist())
        y_pred.extend(np.argmax(predictions, axis=-1).tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    logger.info("Classification report:\n{}", report)

    cm = confusion_matrix(y_true, y_pred)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(cm, class_names, output_dir / "confusion_matrix.png")
    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as report_file:
        report_file.write(report)

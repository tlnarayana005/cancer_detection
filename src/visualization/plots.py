import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_training_history(history, output_path):
    """Plot accuracy and loss curves for training history."""
    history_dir = output_path
    history_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(2, 1, figsize=(10, 12))
    metrics = history.history

    ax[0].plot(metrics.get("accuracy", []), label="train_accuracy")
    ax[0].plot(metrics.get("val_accuracy", []), label="val_accuracy")
    ax[0].set_title("Accuracy")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Accuracy")
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(metrics.get("loss", []), label="train_loss")
    ax[1].plot(metrics.get("val_loss", []), label="val_loss")
    ax[1].set_title("Loss")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Loss")
    ax[1].legend()
    ax[1].grid(True)

    fig.tight_layout()
    fig.savefig(history_dir / "training_history.png")
    plt.close(fig)


def plot_confusion_matrix(cm, class_names, output_path):
    """Plot and save a confusion matrix image."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=class_names,
        yticklabels=class_names,
        cmap="Blues",
    )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

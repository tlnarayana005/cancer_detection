import numpy as np
import tensorflow as tf
from pathlib import Path


def set_seed(seed: int = 42):
    """Set seeds for reproducible training behavior."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_directory(path: str):
    """Create the directory if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def decode_prediction(prediction_vector: np.ndarray, class_names: list[str]) -> tuple[str, float]:
    """Decode classification vector into a label and confidence score."""
    index = int(np.argmax(prediction_vector, axis=-1)[0])
    confidence = float(np.max(prediction_vector, axis=-1)[0])
    return class_names[index], confidence

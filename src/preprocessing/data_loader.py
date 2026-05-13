from pathlib import Path
from typing import Optional

import tensorflow as tf


def build_image_datasets(config: dict) -> tuple[tf.data.Dataset, tf.data.Dataset, Optional[tf.data.Dataset], list[str]]:
    """Load image datasets from structured directories for training, validation, and testing."""
    base_dir = Path(config["data"]["base_dir"])
    image_size = tuple(config["data"]["image_size"])
    batch_size = config["data"]["batch_size"]
    seed = config["data"].get("seed", 42)

    train_dir = base_dir / config["data"].get("train_dir", "train")
    val_dir = base_dir / config["data"].get("val_dir", "val")
    test_dir = base_dir / config["data"].get("test_dir", "test")

    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")

    def load_directory(path, subset=None, validation_split=None):
        return tf.keras.utils.image_dataset_from_directory(
            directory=path,
            labels="inferred",
            label_mode="int",
            batch_size=batch_size,
            image_size=image_size,
            shuffle=True,
            seed=seed,
            validation_split=validation_split,
            subset=subset,
        )

    train_ds = load_directory(train_dir)
    class_names = train_ds.class_names
    val_ds = None
    test_ds = None

    if val_dir.exists():
        val_ds = load_directory(val_dir)
    elif config["data"].get("validation_split", 0.0) > 0:
        train_ds = load_directory(train_dir, subset="training", validation_split=config["data"]["validation_split"])
        val_ds = load_directory(train_dir, subset="validation", validation_split=config["data"]["validation_split"])

    if test_dir.exists():
        test_ds = load_directory(test_dir)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)

    if val_ds is not None:
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    if test_ds is not None:
        test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names

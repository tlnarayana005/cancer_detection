from tensorflow import keras


def get_data_augmentation() -> keras.Sequential:
    """Return a data augmentation pipeline for image training."""
    return keras.Sequential(
        [
            keras.layers.Resizing(224, 224),
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.1),
            keras.layers.RandomZoom(0.1),
            keras.layers.RandomContrast(0.1),
        ],
        name="data_augmentation",
    )

from pathlib import Path

import tensorflow as tf

from src.models.model_builder import build_model
from src.preprocessing.augmentation import get_data_augmentation
from src.preprocessing.data_loader import build_image_datasets
from src.training.evaluate import evaluate_model
from src.utils.logger import create_logger
from src.utils.utils import ensure_directory, set_seed
from src.visualization.plots import plot_training_history


class Trainer:
    def __init__(self, config: dict, logger=None):
        self.config = config
        self.logger = logger or create_logger()
        self.checkpoint_dir = Path(self.config["training"]["checkpoint_dir"])
        self.output_dir = Path(self.config["training"]["model_export"])
        ensure_directory(self.checkpoint_dir)
        ensure_directory(self.output_dir)

    def run(self):
        set_seed(self.config["project"].get("seed", 42))
        self.logger.info("Loading datasets")
        train_ds, val_ds, test_ds, class_names = build_image_datasets(self.config)

        augmentation = get_data_augmentation()
        train_ds = train_ds.map(
            lambda x, y: (augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        self.logger.info("Building model")
        model = build_model(self.config, num_classes=len(class_names))
        model.summary(print_fn=self.logger.info)

        checkpoint_path = self.checkpoint_dir / "best_model.h5"
        monitor_metric = self.config["training"].get("monitor_metric", "val_loss")
        if val_ds is None and monitor_metric.startswith("val_"):
            self.logger.warning(
                "No validation dataset found; switching monitor metric to 'loss'."
            )
            monitor_metric = "loss"

        callbacks = self._build_callbacks(checkpoint_path, monitor_metric)

        self.logger.info("Starting model training")
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.config["training"].get("epochs", 20),
            callbacks=callbacks,
        )

        self.logger.info("Saving final model")
        model_save_path = self.output_dir / "final_model.h5"
        model.save(model_save_path, save_format="h5")

        plot_training_history(history, self.output_dir)
        evaluate_model(
            model=model,
            dataset=test_ds if test_ds is not None else val_ds,
            class_names=class_names,
            output_dir=self.output_dir,
            logger=self.logger,
        )

    def _build_callbacks(self, checkpoint_path: Path, monitor_metric: str):
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor=monitor_metric,
                save_best_only=True,
                save_weights_only=False,
                verbose=1,
                mode=self.config["training"].get("mode", "min"),
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor=monitor_metric,
                patience=self.config["training"].get("patience", 4),
                restore_best_weights=True,
                verbose=1,
                mode=self.config["training"].get("mode", "min"),
            ),
        ]
        return callbacks

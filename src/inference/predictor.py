import numpy as np
from pathlib import Path
from PIL import Image
import tensorflow as tf

from src.models.gradcam import find_last_conv_layer, make_gradcam_heatmap, overlay_gradcam
from src.utils.config import load_config
from src.utils.logger import create_logger
from src.utils.utils import decode_prediction


class CancerPredictor:
    def __init__(self, config: dict | str, logger=None):
        self.config = load_config(config) if isinstance(config, str) else config
        self.logger = logger or create_logger()
        self.image_size = tuple(self.config["data"]["image_size"])
        self.class_names = self.config["inference"].get("class_names", [])
        self.model = None

    def load_model(self, checkpoint_path: str | None = None):
        checkpoint = checkpoint_path or Path(self.config["training"]["model_export"]) / "final_model.h5"
        if not Path(checkpoint).exists():
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint}")
        self.model = tf.keras.models.load_model(str(checkpoint))
        self.logger.info("Loaded model from {}", checkpoint)
        return self.model

    def _load_image(self, image_source):
        if hasattr(image_source, "read"):
            image = Image.open(image_source)
        else:
            image = Image.open(Path(image_source))
        image = image.convert("RGB")
        image = image.resize(self.image_size)
        image_array = np.array(image).astype("float32") / 255.0
        return np.expand_dims(image_array, axis=0), np.array(image)

    def predict(self, image_path: str, checkpoint_path: str | None = None) -> tuple[str, float]:
        model = self.model or self.load_model(checkpoint_path)
        image_tensor, _ = self._load_image(image_path)
        predictions = model.predict(image_tensor, verbose=0)
        label, confidence = decode_prediction(predictions, self.class_names)
        self.logger.info("Image predicted as {} with confidence {}", label, confidence)
        return label, confidence

    def predict_with_explainability(self, image_source, checkpoint_path: str | None = None):
        model = self.model or self.load_model(checkpoint_path)
        image_tensor, original_image = self._load_image(image_source)
        predictions = model.predict(image_tensor, verbose=0)
        label, confidence = decode_prediction(predictions, self.class_names)

        last_layer_name = find_last_conv_layer(model)
        heatmap = make_gradcam_heatmap(image_tensor, model, last_layer_name)
        overlay = overlay_gradcam(np.array(original_image), heatmap)

        gradcam_path = Path("outputs") / "gradcam.png"
        gradcam_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(overlay).save(gradcam_path)

        return label, confidence, str(gradcam_path)

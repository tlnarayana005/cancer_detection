from pathlib import Path
from src.inference.predictor import CancerPredictor


def test_predictor_loads_model():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"
    predictor = CancerPredictor(str(config_path))
    assert predictor.class_names == ["benign", "malignant"]

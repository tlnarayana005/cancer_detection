from pathlib import Path
from src.utils.config import load_config


def test_load_config():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"
    config = load_config(str(config_path))
    assert isinstance(config, dict)
    assert "project" in config
    assert "data" in config
    assert "model" in config

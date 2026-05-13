import yaml
from pathlib import Path


def load_config(config_path: str) -> dict:
    """Load YAML configuration file for the cancer detection pipeline."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    return config

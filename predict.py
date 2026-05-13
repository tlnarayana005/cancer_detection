import argparse
from pathlib import Path

from src.inference.predictor import CancerPredictor
from src.utils.config import load_config
from src.utils.logger import create_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Make prediction on a single medical image")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to configuration file")
    parser.add_argument("--image_path", type=str, required=True, help="Path to image for prediction")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional model checkpoint path")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    logger = create_logger("predict.log")

    predictor = CancerPredictor(config=config, logger=logger)
    label, probability = predictor.predict(args.image_path, checkpoint_path=args.checkpoint)

    logger.info("Prediction complete")
    print(f"Predicted label: {label}")
    print(f"Confidence: {probability:.4f}")


if __name__ == "__main__":
    main()

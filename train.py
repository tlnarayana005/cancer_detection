import argparse
import os
from pathlib import Path

from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.logger import create_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Train cancer detection model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to configuration file")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    logger = create_logger("train.log")

    logger.info("Starting training pipeline")
    trainer = Trainer(config=config, logger=logger)
    trainer.run()
    logger.success("Training pipeline complete")


if __name__ == "__main__":
    main()

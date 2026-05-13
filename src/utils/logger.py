from pathlib import Path

from loguru import logger


def create_logger(log_file: str = "outputs/logs/app.log"):
    """Create a rotating logger that writes to file and stdout."""
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(
        log_path,
        rotation="10 MB",
        retention="10 days",
        backtrace=True,
        diagnose=False,
        enqueue=True,
        level="INFO",
    )
    logger.add(lambda msg: print(msg, end=""), colorize=True, level="INFO")
    return logger

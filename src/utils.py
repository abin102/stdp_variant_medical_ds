import os
import random
import logging
from pathlib import Path

import yaml
import numpy as np
import torch


def load_config(path: str) -> dict:
    """Load YAML configuration file."""
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logging(log_dir: str, experiment_name: str) -> logging.Logger:
    """Configure logging to file and console."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = os.path.join(log_dir, f"{experiment_name}.log")

    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


def setup_directories(config: dict):
    """Create output directories from config."""
    Path(config["experiment"]["log_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["experiment"]["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)


def get_device(config: dict) -> torch.device:
    """Get compute device from config."""
    device_str = config["experiment"].get("device", "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA requested but not available, falling back to CPU")
        device_str = "cpu"
    if device_str == "mps" and not torch.backends.mps.is_available():
        logging.warning("MPS requested but not available, falling back to CPU")
        device_str = "cpu"
    return torch.device(device_str)


def save_checkpoint(state: dict, path: str):
    """Save training checkpoint."""
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, device: torch.device = None) -> dict:
    """Load training checkpoint."""
    if device is None:
        device = torch.device("cpu")
    return torch.load(path, map_location=device, weights_only=False)

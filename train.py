"""Entry point for STDP training experiments."""

import argparse

from src.utils import load_config, set_seed, setup_logging, setup_directories
from src.datasets.loader import DatasetLoader
from src.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train STDP spiking neural network")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    parser.add_argument(
        "--data-root", type=str, default="./data", help="Root directory for datasets"
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Setup
    set_seed(config["experiment"]["seed"])
    setup_directories(config)
    logger = setup_logging(
        config["experiment"]["log_dir"],
        config["experiment"]["name"],
    )

    logger.info(f"Experiment: {config['experiment']['name']}")
    logger.info(f"Config: {args.config}")

    # Load data
    logger.info("Loading dataset...")
    loader = DatasetLoader(config, data_root=args.data_root)
    train_data, test_data = loader.load()
    logger.info(f"  Train: {len(train_data)} samples, Test: {len(test_data)} samples")

    # Create trainer and run
    trainer = Trainer(config, logger)

    logger.info("Starting training...")
    trainer.train(train_data)

    # Evaluate
    eval_cfg = config.get("evaluation", {})
    n_presentations = eval_cfg.get("n_presentations", 1)
    accuracy = trainer.evaluate(train_data, test_data, n_presentations)

    logger.info(f"Final accuracy: {accuracy * 100:.2f}%")

    trainer.finish()


if __name__ == "__main__":
    main()

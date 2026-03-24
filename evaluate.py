"""Standalone evaluation script for trained STDP networks."""

import argparse

import torch

from src.utils import load_config, set_seed, setup_logging
from src.datasets.loader import DatasetLoader
from src.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained STDP network")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint .pt file"
    )
    parser.add_argument(
        "--data-root", type=str, default="./data", help="Root directory for datasets"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["experiment"]["seed"])
    logger = setup_logging(
        config["experiment"]["log_dir"],
        config["experiment"]["name"] + "_eval",
    )

    # Load data
    loader = DatasetLoader(config, data_root=args.data_root)
    train_data, test_data = loader.load()
    logger.info(f"Train: {len(train_data)}, Test: {len(test_data)}")

    # Create trainer and load checkpoint
    trainer = Trainer(config, logger)

    checkpoint = torch.load(args.checkpoint, map_location=trainer.device, weights_only=False)
    trainer.topology.weights = checkpoint["weights"].to(trainer.device)

    if "homeostasis_theta" in checkpoint:
        trainer.homeostasis.theta = checkpoint["homeostasis_theta"].to(trainer.device)
    if "neuron_theta" in checkpoint and hasattr(trainer.neurons, "theta"):
        trainer.neurons.theta = checkpoint["neuron_theta"].to(trainer.device)

    logger.info(f"Loaded checkpoint: {args.checkpoint}")
    logger.info(f"  Epoch: {checkpoint.get('epoch', '?')}, Step: {checkpoint.get('global_step', '?')}")

    # Evaluate
    eval_cfg = config.get("evaluation", {})
    n_presentations = eval_cfg.get("n_presentations", 1)
    accuracy = trainer.evaluate(train_data, test_data, n_presentations)

    logger.info(f"Final accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()

"""Entry point for 2-Layer SDNN training.

Usage:
    # Train STDP layers only:
    python train_conv.py --config configs/conv_triplet_stdp_mnist.yaml

    # Train STDP layers + supervised readout:
    python train_conv.py --config configs/conv_triplet_stdp_mnist.yaml --readout

    # Train readout only (from existing STDP checkpoint):
    python train_conv.py --config configs/conv_triplet_stdp_mnist.yaml --readout-only --stdp-checkpoint checkpoints/sdnn_epoch_3_layer1.pt
"""

import argparse
import os

from src.utils import load_config, set_seed, setup_logging, setup_directories
from src.datasets.loader import DatasetLoader
from src.conv_trainer import ConvTrainer


def main():
    parser = argparse.ArgumentParser(description="Train 2-Layer SDNN with Triplet STDP")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--data-root", type=str, default="./data", help="Dataset root")
    parser.add_argument("--readout", action="store_true", help="Also train supervised readout after STDP")
    parser.add_argument("--readout-only", action="store_true", help="Skip STDP, train readout from checkpoint")
    parser.add_argument("--stdp-checkpoint", type=str, default=None, help="STDP checkpoint for readout-only mode")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["experiment"]["seed"])
    setup_directories(config)
    logger = setup_logging(config["experiment"]["log_dir"], config["experiment"]["name"])

    logger.info(f"Experiment: {config['experiment']['name']}")

    # Load data (unflattened for conv)
    loader = DatasetLoader(config, data_root=args.data_root, flatten=False)
    train_data, test_data = loader.load()

    if not args.readout_only:
        # Phase 1: Train STDP layers
        trainer = ConvTrainer(config, logger, in_shape=loader.get_input_shape())
        trainer.train(train_data)

        # Baseline evaluation (channel voting)
        accuracy = trainer.evaluate(train_data, test_data)
        logger.info(f"Channel voting accuracy: {accuracy * 100:.2f}%")

        trainer.finish()

        # Find the latest checkpoint for readout
        ckpt_dir = config["experiment"]["checkpoint_dir"]
        stdp_checkpoint = _find_latest_checkpoint(ckpt_dir, prefix="sdnn_")
    else:
        stdp_checkpoint = args.stdp_checkpoint
        if not stdp_checkpoint:
            raise ValueError("--stdp-checkpoint required with --readout-only")

    # Phase 2: Train supervised readout
    if args.readout or args.readout_only:
        readout_cfg = config.get("readout", {})
        if not readout_cfg.get("enabled", False):
            logger.warning("Readout not enabled in config. Skipping.")
            return

        from src.readout_trainer import ReadoutTrainer

        logger.info(f"\nTraining supervised SNN readout from: {stdp_checkpoint}")
        readout_trainer = ReadoutTrainer(config, stdp_checkpoint, logger)
        readout_trainer.train(train_data)

        mc_samples = readout_cfg.get("mc_samples", 20)
        metrics = readout_trainer.evaluate_uncertainty(test_data, mc_samples=mc_samples)
        logger.info(f"Final MC accuracy: {metrics.get('accuracy', 'N/A')}")


def _find_latest_checkpoint(ckpt_dir: str, prefix: str = "sdnn_") -> str:
    """Find the most recent checkpoint file."""
    if not os.path.exists(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint dir not found: {ckpt_dir}")

    checkpoints = [
        f for f in os.listdir(ckpt_dir)
        if f.startswith(prefix) and f.endswith(".pt")
    ]
    if not checkpoints:
        raise FileNotFoundError(f"No {prefix}*.pt checkpoints in {ckpt_dir}")

    checkpoints.sort(key=lambda f: os.path.getmtime(os.path.join(ckpt_dir, f)))
    return os.path.join(ckpt_dir, checkpoints[-1])


if __name__ == "__main__":
    main()

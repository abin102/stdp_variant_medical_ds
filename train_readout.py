import argparse
from src.utils import load_config, set_seed, setup_logging, setup_directories
from src.datasets.loader import DatasetLoader
from src.readout_trainer import ReadoutTrainer

def main():
    parser = argparse.ArgumentParser(description="Train Supervised Readout with MC Dropout")
    parser.add_argument("--config", type=str, required=True, help="Path to config")
    parser.add_argument("--stdp-ckpt", type=str, required=True, help="Path to frozen STDP checkpoint")
    parser.add_argument("--data-root", type=str, default="./data", help="Data root")
    parser.add_argument("--resume", type=str, default=None, help="Path to readout checkpoint to resume from")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["experiment"]["seed"])
    setup_directories(config)
    logger = setup_logging(config["experiment"]["log_dir"], config["experiment"]["name"])

    loader = DatasetLoader(config, data_root=args.data_root, flatten=False)
    train_data, test_data = loader.load()
    
    trainer = ReadoutTrainer(config, args.stdp_ckpt, logger, resume_path=args.resume)
    trainer.train(train_data)
    trainer.evaluate_uncertainty(test_data, mc_samples=config["readout"]["mc_samples"])

if __name__ == "__main__":
    main()

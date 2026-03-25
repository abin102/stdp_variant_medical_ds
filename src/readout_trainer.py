"""Supervised SNN Readout Trainer for SDNN features.

Extracts features from the frozen 2-layer SDNN (global average pooled
spike counts), then trains a supervised SNN readout with surrogate
gradients and MC Dropout for uncertainty quantification.
"""

import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from .network.readout import SNNReadout
from .evaluation.uncertainty import UncertaintyEvaluator
from .conv_trainer import ConvTrainer
from .utils import load_checkpoint


class ReadoutTrainer:
    """Train a supervised SNN readout on frozen SDNN features."""

    def __init__(self, config, stdp_checkpoint_path, logger=None, resume_path=None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

        device_str = config["experiment"].get("device", "cpu")
        if device_str == "cuda" and not torch.cuda.is_available():
            device_str = "cpu"
        if device_str == "mps" and not torch.backends.mps.is_available():
            device_str = "cpu"
        self.device = torch.device(device_str)

        # Build the SDNN (frozen feature extractor)
        self.stdp_trainer = ConvTrainer(config, self.logger)

        # Load trained STDP weights
        self.logger.info(f"Loading frozen STDP features from {stdp_checkpoint_path}")
        checkpoint = load_checkpoint(stdp_checkpoint_path, device=self.device)

        for layer in self.stdp_trainer.layers:
            key = f"weights_{layer.name}"
            if key in checkpoint:
                layer.topology.weights = checkpoint[key].to(self.device)
                self.logger.info(f"  Loaded weights for {layer.name}")
            theta_key = f"theta_{layer.name}"
            if theta_key in checkpoint and hasattr(layer.neurons, "theta"):
                layer.neurons.theta = checkpoint[theta_key].to(self.device)

        # Freeze STDP weights
        for layer in self.stdp_trainer.layers:
            layer.topology.weights.requires_grad = False

        # The SDNN outputs global avg pooled features of shape (n_channels,)
        # where n_channels is the last layer's out_channels
        in_features = self.stdp_trainer.layers[-1].pooled_shape[0]
        readout_cfg = config.get("readout", {})

        self.readout = SNNReadout(
            in_features=in_features,
            num_classes=10,
            dropout_p=readout_cfg.get("dropout_p", 0.3),
            alpha=readout_cfg.get("surrogate_alpha", 2.0),
            tau_m=readout_cfg.get("tau_m", 10.0),
            dt=config["encoding"]["dt"],
            v_thresh=readout_cfg.get("v_thresh", 1.0),
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.readout.parameters(),
            lr=readout_cfg.get("lr", 0.001),
            weight_decay=readout_cfg.get("weight_decay", 0.0001),
        )
        self.criterion = nn.CrossEntropyLoss()

        # Wandb
        self.use_wandb = False
        wandb_cfg = config.get("wandb", {})
        if wandb_cfg.get("enabled", False):
            try:
                import wandb
                self.use_wandb = True
                if not wandb.run:
                    wandb.init(
                        project=wandb_cfg.get("project", "stdp-spiking"),
                        entity=wandb_cfg.get("entity"),
                        name=config["experiment"]["name"] + "_readout",
                        config=config,
                        tags=wandb_cfg.get("tags", []) + ["readout"],
                    )
            except ImportError:
                self.logger.warning("wandb not installed.")

        self.global_step = 0
        self.epoch = 0

        if resume_path and os.path.exists(resume_path):
            self.logger.info(f"Resuming readout from: {resume_path}")
            ckpt = torch.load(resume_path, map_location=self.device, weights_only=False)
            self.readout.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.epoch = ckpt["epoch"] + 1
            self.global_step = ckpt.get("global_step", 0)

    def extract_features(self, spikes: torch.Tensor) -> torch.Tensor:
        """Extract SDNN features from a spike train.

        Runs spikes through the frozen 2-layer SDNN and returns
        per-timestep global-avg-pooled feature vectors.

        Args:
            spikes: Encoded spike train [T, C, H, W].

        Returns:
            Feature sequence [T, 1, n_features] for SNNReadout.
        """
        with torch.no_grad():
            for layer in self.stdp_trainer.layers:
                layer.neurons.reset()
                layer.synapse.reset()
                layer.pooling.reset(layer.out_shape)

            features = []
            for t in range(spikes.shape[0]):
                current_input = spikes[t]
                for li, layer in enumerate(self.stdp_trainer.layers):
                    exc = layer.synapse.compute_current(
                        current_input, layer.topology.weights
                    )
                    post = layer.neurons.step(exc)
                    post = layer.spatial_wta(post, layer.neurons.v)
                    layer.neurons.spikes = post
                    pooled = layer.pooling.step(post)
                    current_input = pooled

                # Global avg pool the last layer's pre-pooling spikes → (n_channels,)
                feat = post.mean(dim=(1, 2))  # last layer's post_spikes
                features.append(feat)

            # [T, n_features]
            return torch.stack(features)

    def train(self, train_data: list):
        """Train the supervised readout on SDNN features."""
        readout_cfg = self.config.get("readout", {})
        n_epochs = readout_cfg.get("n_epochs", 100)
        batch_size = readout_cfg.get("batch_size", 32)
        log_interval = self.config.get("wandb", {}).get("log_interval", 50)

        start_epoch = self.epoch
        for epoch in range(start_epoch, n_epochs):
            self.epoch = epoch
            self.readout.train()
            self.logger.info(f"Readout Epoch {epoch + 1}/{n_epochs}")

            indices = np.random.permutation(len(train_data))
            epoch_loss = 0.0
            correct = 0
            total = 0

            for i in tqdm(range(0, len(indices), batch_size), desc="Readout"):
                batch_idx = indices[i : min(i + batch_size, len(indices))]
                loss_batch = 0

                for idx in batch_idx:
                    image, label = train_data[idx]

                    # DoG + encode
                    if self.stdp_trainer.use_dog:
                        image = self.stdp_trainer.dog_filter(image)
                    spikes = self.stdp_trainer.encoder.encode(image)

                    # Extract features → [T, n_features]
                    feats = self.extract_features(spikes)
                    feats = feats.unsqueeze(1)  # [T, 1, n_features]

                    out_spikes = self.readout(feats)  # [T, 1, n_classes]
                    spike_counts = out_spikes.sum(dim=0)  # [1, n_classes]

                    target = torch.tensor([label], device=self.device)
                    loss = self.criterion(spike_counts, target)
                    loss_batch = loss_batch + loss

                    if spike_counts.argmax() == label:
                        correct += 1
                    total += 1

                loss_batch = loss_batch / len(batch_idx)
                self.optimizer.zero_grad()
                loss_batch.backward()
                self.optimizer.step()

                epoch_loss += loss_batch.item()
                self.global_step += 1

                if self.use_wandb and self.global_step % log_interval == 0:
                    import wandb
                    wandb.log({
                        "readout_step": self.global_step,
                        "train/readout_loss": loss_batch.item(),
                        "train/readout_acc": correct / max(total, 1),
                    })

            acc = correct / max(total, 1)
            avg_loss = epoch_loss / max(len(indices) / batch_size, 1)
            self.logger.info(
                f"  Epoch {epoch + 1} — Loss: {avg_loss:.4f}, Acc: {acc:.4f}"
            )

            if self.use_wandb:
                import wandb
                wandb.log({
                    "readout_epoch": epoch + 1,
                    "epoch/readout_loss": avg_loss,
                    "epoch/readout_acc": acc,
                })

            self._save()

    def evaluate_uncertainty(self, test_data: list, mc_samples: int = 20):
        """Evaluate with MC Dropout for uncertainty quantification."""
        self.readout.train()  # keep dropout active

        all_mean_probs = []
        all_var_probs = []
        all_labels = []

        self.logger.info(f"MC Dropout evaluation with {mc_samples} samples...")

        for image, label in tqdm(test_data, desc="MC Eval"):
            if self.stdp_trainer.use_dog:
                image = self.stdp_trainer.dog_filter(image)
            spikes = self.stdp_trainer.encoder.encode(image)
            feats = self.extract_features(spikes).unsqueeze(1)

            sample_probs = []
            for _ in range(mc_samples):
                with torch.no_grad():
                    out_spikes = self.readout(feats)
                    spike_counts = out_spikes.sum(dim=0)
                    probs = F.softmax(spike_counts, dim=-1)
                    sample_probs.append(probs.squeeze(0))

            sample_probs = torch.stack(sample_probs)
            all_mean_probs.append(sample_probs.mean(dim=0))
            all_var_probs.append(sample_probs.var(dim=0))
            all_labels.append(label)

        all_mean_probs = torch.stack(all_mean_probs).cpu()
        all_var_probs = torch.stack(all_var_probs).cpu()
        all_labels = torch.tensor(all_labels).cpu()

        metrics = UncertaintyEvaluator.compute_metrics(
            all_mean_probs, all_var_probs, all_labels
        )
        self.logger.info(f"Uncertainty metrics: {metrics}")

        if self.use_wandb:
            import wandb
            log_dict = {f"eval_mc/{k}": v for k, v in metrics.items()}

            # Per-class accuracy
            preds = all_mean_probs.argmax(dim=1)
            for c in range(10):
                mask = all_labels == c
                if mask.sum() > 0:
                    class_acc = (preds[mask] == c).float().mean().item()
                    log_dict[f"eval/class_{c}_accuracy"] = class_acc

            # Confusion matrix
            log_dict["eval/confusion_matrix"] = wandb.plot.confusion_matrix(
                y_true=all_labels.tolist(),
                preds=preds.tolist(),
                class_names=[str(i) for i in range(10)],
            )

            wandb.log(log_dict)

        return metrics

    def _save(self):
        path = os.path.join(
            self.config["experiment"]["checkpoint_dir"],
            f"readout_epoch_{self.epoch + 1}.pt",
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model": self.readout.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

"""2-Layer SDNN Trainer with Triplet STDP.

Architecture:
    Input (1,28,28)
      → DoG preprocessing → (2,28,28) [ON + OFF channels]
      → Conv1 + Triplet STDP → Spatial WTA + Lateral Inhibition → Max Pool
      → Conv2 + Triplet STDP → Spatial WTA + Lateral Inhibition → Max Pool
      → Global Avg Pool → SNN Readout (surrogate gradients + MC Dropout)

Training is layer-by-layer: train Conv1 first, freeze it, then train Conv2.

Reference: Kheradpisheh et al. (2018) — STDP-based spiking deep
convolutional neural network for object recognition.
"""

import os
import time
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn.functional as torchF
import numpy as np
from tqdm import tqdm

from .neurons.lif import LIFNeuron
from .neurons.adaptive_lif import AdaptiveLIFNeuron
from .synapses.conv_conductance import ConvConductanceSynapse
from .encoding.poisson import PoissonEncoder
from .encoding.latency import LatencyEncoder
from .learning_rules.conv_triplet_stdp import ConvTripletSTDP
from .network.conv_topology import ConvNetworkTopology
from .network.inhibition import LateralInhibition
from .network.pooling import SpikeMaxPool
from .network.spatial_wta import SpatialWTA
from .preprocessing.dog import DoGFilter
from .utils import save_checkpoint

try:
    import wandb
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


@dataclass
class LayerComponents:
    """All components for a single convolutional STDP layer."""
    name: str
    topology: ConvNetworkTopology
    neurons: object  # LIFNeuron or AdaptiveLIFNeuron
    synapse: ConvConductanceSynapse
    learning_rule: ConvTripletSTDP
    inhibition: LateralInhibition
    spatial_wta: SpatialWTA
    pooling: SpikeMaxPool
    in_shape: tuple
    out_shape: tuple  # pre-pooling
    pooled_shape: tuple  # post-pooling


class ConvTrainer:
    """Trainer for 2-Layer SDNN with Convolutional STDP."""

    def __init__(self, config: dict, logger: logging.Logger = None, in_shape: tuple = (1, 28, 28)):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.device = self._get_device()
        self.raw_in_shape = in_shape  # before DoG

        if self.device.type == "mps":
            self.logger.info("Using Apple Silicon MPS backend")

        # Wandb
        self.use_wandb = False
        wandb_cfg = config.get("wandb", {})
        if wandb_cfg.get("enabled", False) and HAS_WANDB:
            self.use_wandb = True
            self.wandb_log_interval = wandb_cfg.get("log_interval", 50)
            wandb.init(
                project=wandb_cfg.get("project", "stdp-spiking"),
                entity=wandb_cfg.get("entity"),
                name=config["experiment"]["name"],
                config=config,
                tags=wandb_cfg.get("tags", []),
            )

        # DoG preprocessing
        preproc_cfg = config.get("preprocessing", {})
        dog_cfg = preproc_cfg.get("dog", {})
        self.use_dog = dog_cfg.get("enabled", False)
        if self.use_dog:
            self.dog_filter = DoGFilter(
                sigma1=dog_cfg.get("sigma1", 1.0),
                sigma2=dog_cfg.get("sigma2", 2.0),
                kernel_size=dog_cfg.get("kernel_size", 7),
                device=self.device,
            )
            # DoG outputs 2 channels (ON + OFF)
            self.in_shape = (2, in_shape[1], in_shape[2])
            self.logger.info(f"DoG preprocessing enabled: {in_shape} → {self.in_shape}")
        else:
            self.in_shape = in_shape

        # Encoder
        self.encoder = self._build_encoder()

        # Build layers
        self.layers: List[LayerComponents] = self._build_layers()
        self.out_shape = self.layers[-1].pooled_shape

        # Training state
        self.epoch = 0
        self.global_step = 0

        # Cache for wandb STDP diagnostics
        self._last_stdp_diag: Dict[str, dict] = {}

    def _get_device(self) -> torch.device:
        device_str = self.config["experiment"].get("device", "cpu")
        if device_str == "cuda" and not torch.cuda.is_available():
            device_str = "cpu"
        if device_str == "mps" and not torch.backends.mps.is_available():
            device_str = "cpu"
        return torch.device(device_str)

    def _build_encoder(self):
        enc_cfg = self.config["encoding"]
        method = enc_cfg.get("method", "poisson")
        if method == "latency":
            return LatencyEncoder(
                time_window=enc_cfg["time_window"],
                dt=enc_cfg["dt"],
                device=self.device,
            )
        else:
            return PoissonEncoder(
                time_window=enc_cfg["time_window"],
                dt=enc_cfg["dt"],
                max_firing_rate=enc_cfg.get("max_firing_rate", 63.75),
                min_firing_rate=enc_cfg.get("min_firing_rate", 0.0),
                device=self.device,
            )

    def _build_layers(self) -> List[LayerComponents]:
        """Build all convolutional STDP layers from config."""
        layers_cfg = self.config.get("layers", [])
        if not layers_cfg:
            # Fallback: single layer from old-style 'network' config
            layers_cfg = [self.config["network"]]

        lr_cfg = self.config["learning_rule"]
        syn_cfg = self.config["synapse"]
        n_cfg = self.config["neuron"]
        dt = self.config["encoding"]["dt"]

        layers = []
        current_in_shape = self.in_shape

        for i, lcfg in enumerate(layers_cfg):
            name = lcfg.get("name", f"conv{i+1}")
            in_channels = lcfg["in_channels"]
            out_channels = lcfg["out_channels"]
            kernel_size = lcfg["kernel_size"]
            stride = lcfg.get("stride", 1)
            padding = lcfg.get("padding", 0)
            pool_size = lcfg.get("pool_size", 2)
            inh_strength = lcfg.get("inhibition_strength", 8.0)

            # Build per-layer config dict for ConvNetworkTopology
            layer_net_cfg = {
                "in_channels": in_channels,
                "out_channels": out_channels,
                "kernel_size": kernel_size,
                "stride": stride,
                "padding": padding,
            }
            layer_config = {"network": layer_net_cfg, "synapse": syn_cfg}

            topology = ConvNetworkTopology(layer_config, self.device)
            out_shape = topology.get_output_shape(current_in_shape)

            # Neurons
            neuron_kwargs = dict(
                shape=out_shape,
                v_rest=n_cfg["v_rest"],
                v_reset=n_cfg["v_reset"],
                v_thresh=n_cfg["v_thresh"],
                tau_membrane=n_cfg["tau_membrane"],
                refractory_period=n_cfg["refractory_period"],
                dt=dt,
                device=self.device,
            )
            model = n_cfg.get("model", "lif")
            if model == "adaptive_lif":
                at_cfg = n_cfg.get("adaptive_threshold", {})
                neuron_kwargs["tau_theta"] = float(at_cfg.get("tau_theta", 1e7))
                neuron_kwargs["theta_increment"] = float(at_cfg.get("theta_increment", 0.05))
                neurons = AdaptiveLIFNeuron(**neuron_kwargs)
            else:
                neurons = LIFNeuron(**neuron_kwargs)

            # Synapse
            synapse = ConvConductanceSynapse(
                shape=out_shape,
                stride=stride,
                padding=padding,
                tau_excitatory=syn_cfg.get("tau_excitatory", 1.0),
                tau_inhibitory=syn_cfg.get("tau_inhibitory", 2.0),
                dt=dt,
                device=self.device,
            )

            # Learning rule
            learning_rule = ConvTripletSTDP(
                in_shape=current_in_shape,
                out_shape=out_shape,
                stride=stride,
                padding=padding,
                tau_plus=lr_cfg["tau_plus"],
                tau_minus=lr_cfg["tau_minus"],
                tau_x=lr_cfg.get("tau_x", 101.0),
                tau_y=lr_cfg.get("tau_y", 125.0),
                A2_plus=lr_cfg["A2_plus"],
                A2_minus=lr_cfg["A2_minus"],
                A3_plus=lr_cfg.get("A3_plus", 0.0091),
                A3_minus=lr_cfg.get("A3_minus", 0.001),
                interaction=lr_cfg.get("interaction", "all_to_all"),
                dt=dt,
                w_min=syn_cfg.get("w_min", 0.0),
                w_max=syn_cfg.get("w_max", 1.0),
                weight_dependence=lr_cfg.get("weight_dependence", "linear"),
                mu_plus=lr_cfg.get("mu_plus", 0.0),
                mu_minus=lr_cfg.get("mu_minus", 0.0),
                device=self.device,
            )

            # Inhibition (channel-wise)
            inhibition = LateralInhibition(
                n_excitatory=out_channels,
                n_inhibitory=out_channels,
                inhibition_strength=inh_strength,
                device=self.device,
            )

            # Spatial WTA
            spatial_wta = SpatialWTA(device=self.device)

            # Pooling
            pooling = SpikeMaxPool(pool_size=pool_size, device=self.device)

            # Compute pooled shape
            C, H, W = out_shape
            pooled_shape = (C, H // pool_size, W // pool_size)

            layer = LayerComponents(
                name=name,
                topology=topology,
                neurons=neurons,
                synapse=synapse,
                learning_rule=learning_rule,
                inhibition=inhibition,
                spatial_wta=spatial_wta,
                pooling=pooling,
                in_shape=current_in_shape,
                out_shape=out_shape,
                pooled_shape=pooled_shape,
            )
            layers.append(layer)

            self.logger.info(
                f"Layer {name}: {current_in_shape} → conv({in_channels}→{out_channels}, "
                f"{kernel_size}x{kernel_size}) → {out_shape} → pool({pool_size}) → {pooled_shape}"
            )

            # Next layer's input is this layer's pooled output
            current_in_shape = pooled_shape

        return layers

    # ------------------------------------------------------------------ #
    #                          TRAINING LOOP                              #
    # ------------------------------------------------------------------ #

    def train(self, train_data: list):
        """Train layers sequentially with STDP.

        Layer-by-layer: train Conv1 for its epochs, freeze it,
        then train Conv2 for its epochs.
        """
        train_cfg = self.config["training"]
        progress_interval = train_cfg.get("progress_interval", 100)
        shuffle = train_cfg.get("shuffle", True)

        silence_cfg = train_cfg.get("silence_handling", {})
        silence_enabled = silence_cfg.get("enabled", True)
        min_spikes = silence_cfg.get("min_spikes", 5)
        rate_boost = silence_cfg.get("rate_boost", 32.0)

        # Mid-epoch checkpoint interval (default: every 10000 samples)
        checkpoint_interval = train_cfg.get("checkpoint_interval", 10000)

        # Epochs per layer (can be specified per-layer or globally)
        layers_cfg = self.config.get("layers", [self.config.get("network", {})])
        global_epochs = train_cfg["n_epochs"]

        for layer_idx, layer in enumerate(self.layers):
            n_epochs = layers_cfg[layer_idx].get("n_epochs", global_epochs)
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Training layer: {layer.name} for {n_epochs} epochs")
            self.logger.info(f"{'='*60}")

            for epoch in range(n_epochs):
                self.epoch = epoch
                self.logger.info(f"Layer {layer.name} — Epoch {epoch + 1}/{n_epochs}")

                indices = list(range(len(train_data)))
                if shuffle:
                    np.random.shuffle(indices)

                epoch_spike_counts = []
                epoch_start = time.time()

                for sample_idx, data_idx in enumerate(tqdm(indices, desc=f"{layer.name} E{epoch+1}")):
                    image, label = train_data[data_idx]

                    # DoG preprocessing
                    if self.use_dog:
                        image = self.dog_filter(image)

                    # Encode
                    spike_train = self.encoder.encode(image)

                    # Simulate through all layers, STDP only on active layer
                    total_spikes = self._simulate_sample(
                        spike_train, train_layer_idx=layer_idx
                    )

                    # Silence handling
                    if silence_enabled and total_spikes < min_spikes:
                        total_spikes = self._handle_silence(
                            image, rate_boost, min_spikes, train_layer_idx=layer_idx
                        )

                    epoch_spike_counts.append(total_spikes)

                    # Normalize weights for the training layer
                    layer.topology.normalize_weights()

                    # Wandb per-sample logging
                    if self.use_wandb and (self.global_step + 1) % self.wandb_log_interval == 0:
                        self._wandb_log_sample(layer, layer_idx, epoch_spike_counts)

                    # Progress logging
                    if (sample_idx + 1) % progress_interval == 0:
                        avg_spikes = np.mean(epoch_spike_counts[-progress_interval:])
                        self.logger.info(
                            f"  Sample {sample_idx + 1}/{len(indices)} | "
                            f"Avg spikes: {avg_spikes:.1f}"
                        )

                    # Mid-epoch checkpoint
                    if (sample_idx + 1) % checkpoint_interval == 0:
                        self.logger.info(
                            f"  Mid-epoch checkpoint at sample {sample_idx + 1}"
                        )
                        self._save_checkpoint(
                            layer_idx=layer_idx,
                            tag=f"e{epoch+1}_s{sample_idx+1}",
                        )

                    self.global_step += 1

                elapsed = time.time() - epoch_start
                avg_spikes = np.mean(epoch_spike_counts)
                self.logger.info(
                    f"  {layer.name} Epoch {epoch + 1} done in {elapsed:.1f}s | "
                    f"Avg spikes/sample: {avg_spikes:.1f}"
                )

                # Wandb epoch logging
                if self.use_wandb:
                    self._wandb_log_epoch(layer, layer_idx, epoch, epoch_spike_counts, elapsed)

                self._save_checkpoint(layer_idx=layer_idx)

            # Freeze this layer before moving to the next
            self.logger.info(f"Freezing layer {layer.name}")

        self.logger.info("STDP training complete for all layers.")

    def _simulate_sample(
        self, spike_train: torch.Tensor, train_layer_idx: int = -1
    ) -> int:
        """Run one sample through the full 2-layer SDNN.

        Args:
            spike_train: Encoded spike train [T, C_in, H, W].
            train_layer_idx: Which layer to apply STDP to (-1 = none).

        Returns:
            Total post-synaptic spikes in the last layer.
        """
        n_timesteps = spike_train.shape[0]

        # Reset all layers
        for layer in self.layers:
            layer.neurons.reset()
            layer.synapse.reset()
            layer.learning_rule.reset_traces()
            layer.pooling.reset(layer.out_shape)

        # Pre-cache weight refs and bounds
        layer_weights = [l.topology.weights for l in self.layers]
        w_mins = [l.topology.w_min for l in self.layers]
        w_maxs = [l.topology.w_max for l in self.layers]

        total_spikes = 0
        had_spikes = [False] * len(self.layers)

        for t in range(n_timesteps):
            current_input = spike_train[t]

            for li, layer in enumerate(self.layers):
                # Excitatory current from conv
                exc_current = layer.synapse.compute_current(
                    current_input, layer_weights[li]
                )

                # Lateral inhibition (channel-wise)
                if had_spikes[li]:
                    channel_activity = layer.neurons.spikes.sum(dim=(1, 2))
                    inh_current = layer.inhibition.compute_inhibition(channel_activity)
                    inh_current = inh_current.view(-1, 1, 1).expand_as(exc_current)
                    inh_current = layer.synapse.compute_inhibitory_current(inh_current)
                    exc_current = exc_current + inh_current

                # Neuron step
                post_spikes = layer.neurons.step(exc_current)

                # Spatial WTA: at each (h,w), only the winning channel spikes
                post_spikes = layer.spatial_wta(post_spikes, layer.neurons.v)
                # Update neuron spike record after WTA filtering
                layer.neurons.spikes = post_spikes

                # STDP update (only for the layer being trained)
                if li == train_layer_idx:
                    dw, diag = layer.learning_rule.update(
                        current_input, post_spikes, layer_weights[li]
                    )
                    layer_weights[li] = (layer_weights[li] + dw).clamp(
                        w_mins[li], w_maxs[li]
                    )
                    self._last_stdp_diag[layer.name] = diag

                # Track if any spikes happened (for inhibition gating)
                spike_sum = post_spikes.sum()
                if not had_spikes[li] and spike_sum > 0:
                    had_spikes[li] = True

                # Pool spikes for next layer's input
                pooled = layer.pooling.step(post_spikes)

                # Count spikes in last layer
                if li == len(self.layers) - 1:
                    total_spikes += int(spike_sum.item())

                # Next layer sees pooled spikes
                current_input = pooled

        # Write weights back
        for li, layer in enumerate(self.layers):
            layer.topology.weights = layer_weights[li]

        return total_spikes

    def _handle_silence(
        self, image: torch.Tensor, rate_boost: float, min_spikes: int,
        train_layer_idx: int = -1
    ) -> int:
        """Re-encode with boosted rate if network is silent."""
        if not hasattr(self.encoder, "max_firing_rate"):
            return 0

        original_max_rate = self.encoder.max_firing_rate
        total_spikes = 0

        for attempt in range(5):
            self.encoder.max_firing_rate = original_max_rate + rate_boost * (attempt + 1)
            spike_train = self.encoder.encode(image)
            total_spikes = self._simulate_sample(spike_train, train_layer_idx=train_layer_idx)
            if total_spikes >= min_spikes:
                break

        self.encoder.max_firing_rate = original_max_rate
        return total_spikes

    # ------------------------------------------------------------------ #
    #                        FEATURE EXTRACTION                           #
    # ------------------------------------------------------------------ #

    def extract_features(self, data: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features from frozen SDNN for readout training.

        Runs each sample through all layers, collects spike counts
        after final pooling, applies global average pooling.

        Args:
            data: List of (image, label) tuples.

        Returns:
            features: [N, n_channels] feature matrix.
            labels: [N] label tensor.
        """
        n_channels = self.layers[-1].pooled_shape[0]
        features = []
        labels = []

        for image, label in tqdm(data, desc="Extracting features"):
            if self.use_dog:
                image = self.dog_filter(image)

            spike_train = self.encoder.encode(image)
            n_timesteps = spike_train.shape[0]

            # Reset all layers
            for layer in self.layers:
                layer.neurons.reset()
                layer.synapse.reset()
                layer.pooling.reset(layer.out_shape)

            # Accumulate spike counts after final pooling
            spike_counts = torch.zeros(self.layers[-1].pooled_shape, device=self.device)

            for t in range(n_timesteps):
                current_input = spike_train[t]

                for li, layer in enumerate(self.layers):
                    exc_current = layer.synapse.compute_current(
                        current_input, layer.topology.weights
                    )
                    post_spikes = layer.neurons.step(exc_current)
                    post_spikes = layer.spatial_wta(post_spikes, layer.neurons.v)
                    layer.neurons.spikes = post_spikes

                    pooled = layer.pooling.step(post_spikes)

                    if li == len(self.layers) - 1:
                        # Accumulate spike counts for final layer
                        spike_counts += SpikeMaxPool.pool_spike_counts(
                            post_spikes, layer.pooling.pool_size
                        )

                    current_input = pooled

            # Global average pooling: (C, H, W) → (C,)
            feat = spike_counts.mean(dim=(1, 2))  # [n_channels]
            features.append(feat.cpu())
            labels.append(label)

        return torch.stack(features), torch.tensor(labels)

    # ------------------------------------------------------------------ #
    #                           EVALUATION                                #
    # ------------------------------------------------------------------ #

    def evaluate(self, train_data: list, test_data: list, n_presentations: int = 1) -> float:
        """Evaluate using channel-level voting (baseline, no readout).

        For proper evaluation, use the SNN readout via ReadoutTrainer.
        """
        self.logger.info("Evaluating with channel voting (baseline)...")

        n_classes = 10
        final_layer = self.layers[-1]
        n_channels = final_layer.pooled_shape[0]
        channel_votes = torch.zeros((n_channels, n_classes), device=self.device)

        # Label channels by training data response
        for image, label in tqdm(train_data, desc="Labeling channels"):
            if self.use_dog:
                image = self.dog_filter(image)
            spike_train = self.encoder.encode(image)
            n_timesteps = spike_train.shape[0]

            for layer in self.layers:
                layer.neurons.reset()
                layer.synapse.reset()
                layer.pooling.reset(layer.out_shape)

            channel_spikes = torch.zeros(n_channels, device=self.device)

            for t in range(n_timesteps):
                current_input = spike_train[t]
                for li, layer in enumerate(self.layers):
                    exc = layer.synapse.compute_current(current_input, layer.topology.weights)
                    post = layer.neurons.step(exc)
                    post = layer.spatial_wta(post, layer.neurons.v)
                    layer.neurons.spikes = post
                    pooled = layer.pooling.step(post)
                    if li == len(self.layers) - 1:
                        channel_spikes += post.sum(dim=(1, 2))
                    current_input = pooled

            channel_votes[:, label] += channel_spikes

        channel_labels = channel_votes.argmax(dim=1)

        # Test
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        for image, label in tqdm(test_data, desc="Testing"):
            if self.use_dog:
                image = self.dog_filter(image)
            spike_train = self.encoder.encode(image)
            n_timesteps = spike_train.shape[0]

            for layer in self.layers:
                layer.neurons.reset()
                layer.synapse.reset()
                layer.pooling.reset(layer.out_shape)

            votes = torch.zeros(n_classes, device=self.device)

            for t in range(n_timesteps):
                current_input = spike_train[t]
                for li, layer in enumerate(self.layers):
                    exc = layer.synapse.compute_current(current_input, layer.topology.weights)
                    post = layer.neurons.step(exc)
                    post = layer.spatial_wta(post, layer.neurons.v)
                    layer.neurons.spikes = post
                    pooled = layer.pooling.step(post)
                    if li == len(self.layers) - 1:
                        ch_spikes = post.sum(dim=(1, 2))
                        for c in range(n_channels):
                            votes[channel_labels[c]] += ch_spikes[c]
                    current_input = pooled

            pred = votes.argmax().item()
            all_preds.append(pred)
            all_labels.append(label)
            if pred == label:
                correct += 1
            total += 1

        accuracy = correct / max(1, total)
        self.logger.info(f"  Channel voting accuracy: {accuracy * 100:.2f}%")

        if self.use_wandb:
            log_dict = {
                "eval/channel_voting_accuracy": accuracy,
            }
            # Per-class neuron counts
            label_dist = {}
            for c in range(n_classes):
                count = (channel_labels == c).sum().item()
                label_dist[c] = count
                log_dict[f"eval/channels_class_{c}"] = count

            log_dict["eval/confusion_matrix"] = wandb.plot.confusion_matrix(
                y_true=all_labels,
                preds=all_preds,
                class_names=[str(i) for i in range(n_classes)],
            )
            wandb.log(log_dict)

        return accuracy

    # ------------------------------------------------------------------ #
    #                           WANDB LOGGING                             #
    # ------------------------------------------------------------------ #

    def _wandb_log_sample(self, layer: LayerComponents, layer_idx: int, spike_counts: list):
        """Log per-sample metrics to wandb."""
        w = layer.topology.weights
        prefix = f"train/{layer.name}"

        log_dict = {
            "sample_step": self.global_step,
            f"{prefix}/spikes_per_sample": spike_counts[-1] if spike_counts else 0,
            f"{prefix}/avg_spikes_recent": np.mean(
                spike_counts[-min(len(spike_counts), self.wandb_log_interval):]
            ),
            f"{prefix}/weight_mean": w.mean().item(),
            f"{prefix}/weight_std": w.std().item(),
            f"{prefix}/weight_max": w.max().item(),
            f"{prefix}/weight_min": w.min().item(),
            f"{prefix}/weight_sparsity": (w < 0.01).float().mean().item(),
        }

        # LTP/LTD diagnostics
        diag = self._last_stdp_diag.get(layer.name)
        if diag is not None:
            ltp_mag = diag["dw_plus"].abs().mean().item()
            ltd_mag = diag["dw_minus"].abs().mean().item()
            log_dict[f"{prefix}/ltp_magnitude"] = ltp_mag
            log_dict[f"{prefix}/ltd_magnitude"] = ltd_mag
            log_dict[f"{prefix}/ltp_ltd_ratio"] = ltp_mag / max(ltd_mag, 1e-10)

        wandb.log(log_dict)

    def _wandb_log_epoch(
        self, layer: LayerComponents, layer_idx: int,
        epoch: int, spike_counts: list, elapsed: float
    ):
        """Log per-epoch metrics and visualizations to wandb."""
        w = layer.topology.weights
        prefix = f"epoch/{layer.name}"

        log_dict = {
            "epoch": epoch + 1,
            f"{prefix}/avg_spikes": np.mean(spike_counts),
            f"{prefix}/std_spikes": np.std(spike_counts),
            f"{prefix}/duration_s": elapsed,
            f"{prefix}/weight_mean": w.mean().item(),
            f"{prefix}/weight_std": w.std().item(),
            f"{prefix}/weight_sparsity": (w < 0.01).float().mean().item(),
        }

        # Weight histogram
        log_dict[f"{prefix}/weight_histogram"] = wandb.Histogram(
            w.detach().cpu().numpy().flatten()
        )

        # Adaptive threshold stats
        if hasattr(layer.neurons, "theta"):
            theta = layer.neurons.theta
            log_dict[f"{prefix}/theta_mean"] = theta.mean().item()
            log_dict[f"{prefix}/theta_max"] = theta.max().item()

        # Weight filter visualization
        self._wandb_log_filters(layer, prefix, log_dict)

        wandb.log(log_dict)

    def _wandb_log_filters(self, layer: LayerComponents, prefix: str, log_dict: dict):
        """Log conv filters as wandb images."""
        w = layer.topology.weights.detach().cpu().numpy()
        # w shape: (out_channels, in_channels, kH, kW)
        n_filters = w.shape[0]
        in_ch = w.shape[1]
        kH, kW = w.shape[2], w.shape[3]

        n_cols = min(n_filters, 15)
        n_rows = int(np.ceil(n_filters / n_cols))

        # Show first input channel
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.0, n_rows * 1.0))
        if n_rows == 1:
            axes = axes[np.newaxis, :]
        for idx in range(n_rows * n_cols):
            r, c = divmod(idx, n_cols)
            ax = axes[r, c]
            if idx < n_filters:
                # Average across input channels for display
                filt = w[idx].mean(axis=0)
                ax.imshow(filt, cmap="hot", interpolation="nearest")
            ax.axis("off")
        plt.tight_layout()
        log_dict[f"{prefix}/weight_filters"] = wandb.Image(fig)
        plt.close(fig)

    # ------------------------------------------------------------------ #
    #                         CHECKPOINTING                               #
    # ------------------------------------------------------------------ #

    def _save_checkpoint(self, layer_idx: int = -1, tag: str = None):
        ckpt_dir = self.config["experiment"]["checkpoint_dir"]
        os.makedirs(ckpt_dir, exist_ok=True)

        state = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "config": self.config,
        }

        # Save all layer weights
        for i, layer in enumerate(self.layers):
            state[f"weights_{layer.name}"] = layer.topology.weights.cpu()
            if hasattr(layer.neurons, "theta"):
                state[f"theta_{layer.name}"] = layer.neurons.theta.cpu()

        suffix = f"_layer{layer_idx}" if layer_idx >= 0 else ""
        if tag:
            suffix += f"_{tag}"
        path = os.path.join(ckpt_dir, f"sdnn_epoch_{self.epoch + 1}{suffix}.pt")
        save_checkpoint(state, path)
        self.logger.info(f"  Checkpoint saved: {path}")

    def finish(self):
        """Clean up wandb."""
        if self.use_wandb:
            wandb.finish()

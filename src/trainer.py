import io
import os
import time
import logging

import torch
import numpy as np
from tqdm import tqdm

from .neurons.lif import LIFNeuron
from .neurons.adaptive_lif import AdaptiveLIFNeuron
from .synapses.conductance import ConductanceSynapse
from .synapses.current import CurrentSynapse
from .encoding.poisson import PoissonEncoder
from .encoding.latency import LatencyEncoder
from .encoding.rank_order import RankOrderEncoder
from .learning_rules.pair_stdp import PairSTDP
from .learning_rules.triplet_stdp import TripletSTDP
from .learning_rules.voltage_stdp import VoltageSTDP
from .learning_rules.probabilistic_stdp import ProbabilisticSTDP
from .network.topology import NetworkTopology
from .network.inhibition import LateralInhibition
from .network.homeostasis import Homeostasis
from .datasets.loader import DatasetLoader
from .evaluation.labeling import NeuronLabeler
from .evaluation.metrics import compute_accuracy
from .visualization.weights import plot_weight_maps
from .utils import save_checkpoint

try:
    import wandb
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class Trainer:
    """Main training loop for STDP-based spiking neural networks.

    Orchestrates:
        1. Data loading and encoding
        2. Network creation (neurons, synapses, topology)
        3. Per-sample, per-timestep simulation
        4. STDP weight updates
        5. Evaluation and checkpointing
    """

    def __init__(self, config: dict, logger: logging.Logger = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.device = self._get_device()

        # MPS (Apple Silicon) support
        if self.device.type == "mps":
            # MPS doesn't support all ops; fall back gracefully
            self.logger.info("Using Apple Silicon MPS backend")

        # Build components
        self.encoder = self._build_encoder()
        self.topology = NetworkTopology(config, self.device)
        self.neurons = self._build_neurons()
        self.synapse = self._build_synapse()
        self.learning_rule = self._build_learning_rule()
        self.inhibition = LateralInhibition(
            n_excitatory=config["network"]["n_excitatory"],
            n_inhibitory=config["network"]["n_inhibitory"],
            inhibition_strength=config["network"].get("inhibition_strength", 17.0),
            device=self.device,
        )
        self.homeostasis = Homeostasis(
            n_neurons=config["network"]["n_excitatory"],
            tau_theta=config["neuron"]["adaptive_threshold"].get("tau_theta", 1e7),
            theta_increment=config["neuron"]["adaptive_threshold"].get("theta_increment", 0.05),
            dt=config["encoding"]["dt"],
            device=self.device,
        )

        # Cache flags for inner loop (avoid repeated dict/hasattr lookups)
        self._use_adaptive_threshold = config["neuron"]["adaptive_threshold"].get("enabled", False)
        self._use_voltage_stdp = hasattr(self.learning_rule, "update_with_voltage")
        self._has_inh_current = hasattr(self.synapse, "compute_inhibitory_current")

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.spike_counts_per_sample = []

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
            wandb.define_metric("sample_step")
            wandb.define_metric("train/*", step_metric="sample_step")
            wandb.define_metric("epoch")
            wandb.define_metric("epoch/*", step_metric="epoch")
            wandb.define_metric("eval/*", step_metric="epoch")

    def _get_device(self) -> torch.device:
        device_str = self.config["experiment"].get("device", "cpu")
        if device_str == "cuda" and not torch.cuda.is_available():
            self.logger.warning("CUDA not available, using CPU")
            device_str = "cpu"
        if device_str == "mps" and not torch.backends.mps.is_available():
            self.logger.warning("MPS not available, using CPU")
            device_str = "cpu"
        return torch.device(device_str)

    def _build_encoder(self):
        enc_cfg = self.config["encoding"]
        method = enc_cfg["method"]
        kwargs = dict(
            time_window=enc_cfg["time_window"],
            dt=enc_cfg["dt"],
            device=self.device,
        )
        if method == "poisson":
            kwargs["max_firing_rate"] = enc_cfg.get("max_firing_rate", 63.75)
            kwargs["min_firing_rate"] = enc_cfg.get("min_firing_rate", 0.0)
            return PoissonEncoder(**kwargs)
        elif method == "latency":
            return LatencyEncoder(**kwargs)
        elif method == "rank_order":
            return RankOrderEncoder(**kwargs)
        else:
            raise ValueError(f"Unknown encoding method: {method}")

    def _build_neurons(self):
        n_cfg = self.config["neuron"]
        n_exc = self.config["network"]["n_excitatory"]
        dt = self.config["encoding"]["dt"]
        model = n_cfg.get("model", "lif")

        base_kwargs = dict(
            n_neurons=n_exc,
            v_rest=n_cfg["v_rest"],
            v_reset=n_cfg["v_reset"],
            v_thresh=n_cfg["v_thresh"],
            tau_membrane=n_cfg["tau_membrane"],
            refractory_period=n_cfg["refractory_period"],
            dt=dt,
            device=self.device,
        )

        if model == "adaptive_lif":
            adaptive_cfg = n_cfg.get("adaptive_threshold", {})
            base_kwargs["tau_theta"] = adaptive_cfg.get("tau_theta", 1e7)
            base_kwargs["theta_increment"] = adaptive_cfg.get("theta_increment", 0.05)
            return AdaptiveLIFNeuron(**base_kwargs)
        else:
            return LIFNeuron(**base_kwargs)

    def _build_synapse(self):
        syn_cfg = self.config["synapse"]
        model = syn_cfg.get("model", "conductance")
        dt = self.config["encoding"]["dt"]
        n_exc = self.config["network"]["n_excitatory"]

        if model == "conductance":
            return ConductanceSynapse(
                n_post=n_exc,
                tau_excitatory=syn_cfg.get("tau_excitatory", 1.0),
                tau_inhibitory=syn_cfg.get("tau_inhibitory", 2.0),
                dt=dt,
                device=self.device,
            )
        elif model == "current":
            return CurrentSynapse(dt=dt, device=self.device)
        else:
            raise ValueError(f"Unknown synapse model: {model}")

    def _build_learning_rule(self):
        lr_cfg = self.config["learning_rule"]
        rule_type = lr_cfg["type"]
        n_input = self.config["network"]["n_input"]
        n_exc = self.config["network"]["n_excitatory"]
        dt = self.config["encoding"]["dt"]
        syn_cfg = self.config["synapse"]

        base_kwargs = dict(
            n_pre=n_input,
            n_post=n_exc,
            dt=dt,
            w_min=syn_cfg.get("w_min", 0.0),
            w_max=syn_cfg.get("w_max", 1.0),
            weight_dependence=lr_cfg.get("weight_dependence", "none"),
            mu_plus=lr_cfg.get("mu_plus", 0.0),
            mu_minus=lr_cfg.get("mu_minus", 0.0),
            device=self.device,
        )

        if rule_type == "pair_stdp":
            return PairSTDP(
                tau_plus=lr_cfg["tau_plus"],
                tau_minus=lr_cfg["tau_minus"],
                A2_plus=lr_cfg["A2_plus"],
                A2_minus=lr_cfg["A2_minus"],
                interaction=lr_cfg.get("interaction", "all_to_all"),
                **base_kwargs,
            )
        elif rule_type == "triplet_stdp":
            return TripletSTDP(
                tau_plus=lr_cfg["tau_plus"],
                tau_minus=lr_cfg["tau_minus"],
                tau_x=lr_cfg.get("tau_x", 101.0),
                tau_y=lr_cfg.get("tau_y", 125.0),
                A2_plus=lr_cfg["A2_plus"],
                A2_minus=lr_cfg["A2_minus"],
                A3_plus=lr_cfg.get("A3_plus", 0.0091),
                A3_minus=lr_cfg.get("A3_minus", 0.0),
                interaction=lr_cfg.get("interaction", "all_to_all"),
                **base_kwargs,
            )
        elif rule_type == "voltage_stdp":
            return VoltageSTDP(
                tau_lowpass=lr_cfg.get("tau_lowpass", 10.0),
                tau_v_minus=lr_cfg.get("tau_v_minus", 10.0),
                theta_minus=lr_cfg.get("theta_minus", -70.0),
                theta_plus=lr_cfg.get("theta_plus", -49.0),
                A_ltd=lr_cfg.get("A_ltd", 0.0001),
                A_ltp=lr_cfg.get("A_ltp", 0.0002),
                tau_plus=lr_cfg["tau_plus"],
                **base_kwargs,
            )
        elif rule_type == "probabilistic_stdp":
            return ProbabilisticSTDP(
                tau_plus=lr_cfg["tau_plus"],
                tau_minus=lr_cfg["tau_minus"],
                A2_plus=lr_cfg["A2_plus"],
                A2_minus=lr_cfg["A2_minus"],
                p_update=lr_cfg.get("p_update", 0.5),
                temperature=lr_cfg.get("temperature", 1.0),
                interaction=lr_cfg.get("interaction", "all_to_all"),
                **base_kwargs,
            )
        else:
            raise ValueError(f"Unknown learning rule: {rule_type}")

    def train(self, train_data: list):
        """Run the full training loop.

        Args:
            train_data: List of (image_tensor, label) tuples.
        """
        train_cfg = self.config["training"]
        n_epochs = train_cfg["n_epochs"]
        progress_interval = train_cfg.get("progress_interval", 100)
        shuffle = train_cfg.get("shuffle", True)
        silence_cfg = train_cfg.get("silence_handling", {})
        silence_enabled = silence_cfg.get("enabled", True)
        min_spikes = silence_cfg.get("min_spikes", 5)
        rate_boost = silence_cfg.get("rate_boost", 32.0)

        vis_cfg = self.config.get("visualization", {})
        weight_map_interval = vis_cfg.get("weight_map_interval", 5000)
        save_weight_maps = vis_cfg.get("save_weight_maps", True)

        # Learning rate decay schedule
        lr_decay_cfg = self.config["learning_rule"].get("lr_decay", {})
        lr_decay_enabled = lr_decay_cfg.get("enabled", False)
        decay_factor = lr_decay_cfg.get("decay_factor", 0.1)
        decay_epochs = lr_decay_cfg.get("decay_epochs", [])

        for epoch in range(n_epochs):
            self.epoch = epoch
            self.logger.info(f"Epoch {epoch + 1}/{n_epochs}")

            # Learning rate decay
            if lr_decay_enabled and epoch in decay_epochs:
                self._decay_learning_rate(decay_factor)
                self.logger.info(f"  Learning rate decayed by {decay_factor}")

            # Shuffle training data
            indices = list(range(len(train_data)))
            if shuffle:
                np.random.shuffle(indices)

            epoch_spike_counts = []
            silence_retries = 0
            epoch_start = time.time()

            for sample_idx, data_idx in enumerate(tqdm(indices, desc=f"Epoch {epoch+1}")):
                image, label = train_data[data_idx]

                # Encode image to spike train
                spike_train = self.encoder.encode(image)

                # Simulate network for this sample
                total_spikes = self._simulate_sample(spike_train, learn=True)

                # Handle silent network (no post-synaptic spikes)
                was_silent = False
                if silence_enabled and total_spikes < min_spikes:
                    was_silent = True
                    silence_retries += 1
                    total_spikes = self._handle_silence(
                        image, rate_boost, min_spikes
                    )

                epoch_spike_counts.append(total_spikes)

                # Normalize weights after each sample
                self.topology.normalize_weights()

                # Wandb per-sample logging
                if self.use_wandb and (self.global_step + 1) % self.wandb_log_interval == 0:
                    w = self.topology.weights
                    wandb.log({
                        "sample_step": self.global_step,
                        "train/spikes_per_sample": total_spikes,
                        "train/avg_spikes_recent": np.mean(epoch_spike_counts[-self.wandb_log_interval:]),
                        "train/silence_retry": int(was_silent),
                        "train/weight_mean": w.mean().item(),
                        "train/weight_std": w.std().item(),
                        "train/weight_sparsity": (w < 0.01).float().mean().item(),
                        "train/weight_max": w.max().item(),
                        "train/weight_min": w.min().item(),
                    })

                # Progress logging
                if (sample_idx + 1) % progress_interval == 0:
                    avg_spikes = np.mean(epoch_spike_counts[-progress_interval:])
                    self.logger.info(
                        f"  Sample {sample_idx + 1}/{len(indices)} | "
                        f"Avg spikes: {avg_spikes:.1f}"
                    )

                # Save weight maps periodically
                if save_weight_maps and (self.global_step + 1) % weight_map_interval == 0:
                    self._save_weight_snapshot()
                    if self.use_wandb:
                        self._wandb_log_weight_maps()

                self.global_step += 1

            elapsed = time.time() - epoch_start
            avg_epoch_spikes = np.mean(epoch_spike_counts)

            # Per-neuron activity stats
            active_neurons = (self.homeostasis.spike_counts > 0).sum().item()
            n_exc = self.config["network"]["n_excitatory"]
            neuron_utilization = active_neurons / n_exc

            self.logger.info(
                f"  Epoch {epoch + 1} complete in {elapsed:.1f}s | "
                f"Avg spikes/sample: {avg_epoch_spikes:.1f} | "
                f"Active neurons: {active_neurons}/{n_exc} ({neuron_utilization:.1%})"
            )

            # Wandb epoch-level logging
            if self.use_wandb:
                w = self.topology.weights
                wandb.log({
                    "epoch": epoch + 1,
                    "epoch/avg_spikes": avg_epoch_spikes,
                    "epoch/std_spikes": np.std(epoch_spike_counts),
                    "epoch/silence_retries": silence_retries,
                    "epoch/active_neurons": active_neurons,
                    "epoch/neuron_utilization": neuron_utilization,
                    "epoch/weight_mean": w.mean().item(),
                    "epoch/weight_std": w.std().item(),
                    "epoch/weight_sparsity": (w < 0.01).float().mean().item(),
                    "epoch/duration_s": elapsed,
                    "epoch/theta_mean": self.homeostasis.theta.mean().item(),
                    "epoch/theta_max": self.homeostasis.theta.max().item(),
                })
                # Log weight distribution histogram
                wandb.log({
                    "epoch": epoch + 1,
                    "epoch/weight_histogram": wandb.Histogram(w.cpu().numpy().flatten()),
                })
                # Log weight maps as images
                self._wandb_log_weight_maps()

            # Reset homeostasis counts for next epoch
            self.homeostasis.reset_counts()

            # Save checkpoint
            self._save_checkpoint()

    def _simulate_sample(self, spike_train: torch.Tensor, learn: bool = True) -> int:
        """Simulate the network for one input sample.

        Args:
            spike_train: Input spikes [n_timesteps, n_input].
            learn: Whether to apply STDP updates.

        Returns:
            Total number of post-synaptic spikes.
        """
        n_timesteps = spike_train.shape[0]

        # Reset neuron and synapse state
        self.neurons.reset()
        self.synapse.reset()
        self.learning_rule.reset_traces()

        # Cache config lookups outside the hot loop
        use_adaptive = self._use_adaptive_threshold
        use_voltage_stdp = self._use_voltage_stdp
        has_inh_current = self._has_inh_current
        weights = self.topology.weights
        w_min = self.topology.w_min
        w_max = self.topology.w_max

        # Accumulate spike count on-device (avoid .item() per timestep)
        spike_accumulator = torch.zeros(1, device=self.device)
        had_any_spikes = False

        for t in range(n_timesteps):
            pre_spikes = spike_train[t]

            # Compute excitatory synaptic current
            exc_current = self.synapse.compute_current(pre_spikes, weights)

            # Add inhibitory current (skip until first spike)
            if had_any_spikes:
                inh_current = self.inhibition.compute_inhibition(self.neurons.spikes)
                if has_inh_current:
                    inh_current = self.synapse.compute_inhibitory_current(inh_current)
                    exc_current = exc_current + inh_current
                else:
                    exc_current = exc_current - inh_current

            # Neuron update
            post_spikes = self.neurons.step(exc_current)

            # Homeostatic threshold adaptation
            if use_adaptive:
                self.homeostasis.update_threshold(post_spikes)

            # STDP weight update
            if learn:
                if use_voltage_stdp:
                    dw = self.learning_rule.update_with_voltage(
                        pre_spikes, post_spikes, weights, self.neurons.v
                    )
                else:
                    dw = self.learning_rule.update(
                        pre_spikes, post_spikes, weights
                    )
                weights = (weights + dw).clamp(w_min, w_max)

            # Accumulate on-device
            spike_sum = post_spikes.sum()
            spike_accumulator += spike_sum
            if not had_any_spikes and spike_sum > 0:
                had_any_spikes = True

        # Write back weights
        self.topology.weights = weights

        return int(spike_accumulator.item())

    def _handle_silence(self, image: torch.Tensor, rate_boost: float, min_spikes: int) -> int:
        """Re-encode and simulate with boosted firing rate when network is silent."""
        if not isinstance(self.encoder, PoissonEncoder):
            return 0

        original_max_rate = self.encoder.max_firing_rate
        total_spikes = 0
        max_retries = 5

        for attempt in range(max_retries):
            self.encoder.max_firing_rate = original_max_rate + rate_boost * (attempt + 1)
            spike_train = self.encoder.encode(image)
            total_spikes = self._simulate_sample(spike_train, learn=True)

            if total_spikes >= min_spikes:
                break

        self.encoder.max_firing_rate = original_max_rate
        return total_spikes

    def _decay_learning_rate(self, factor: float):
        """Scale STDP amplitude parameters by decay factor."""
        if hasattr(self.learning_rule, "A2_plus"):
            self.learning_rule.A2_plus *= factor
        if hasattr(self.learning_rule, "A2_minus"):
            self.learning_rule.A2_minus *= factor
        if hasattr(self.learning_rule, "A3_plus"):
            self.learning_rule.A3_plus *= factor
        if hasattr(self.learning_rule, "A3_minus"):
            self.learning_rule.A3_minus *= factor

    def _save_weight_snapshot(self):
        """Save weight map visualization."""
        vis_dir = os.path.join(
            self.config["experiment"]["log_dir"], "weight_maps"
        )
        os.makedirs(vis_dir, exist_ok=True)
        path = os.path.join(vis_dir, f"weights_step{self.global_step + 1}.png")
        n_input = self.config["network"]["n_input"]
        side = int(np.sqrt(n_input))
        plot_weight_maps(
            self.topology.weights, side, side, save_path=path, show=False
        )

    def _wandb_log_weight_maps(self):
        """Log weight receptive fields as a wandb Image."""
        if not self.use_wandb:
            return
        w = self.topology.weights.detach().cpu().numpy()
        n_input = self.config["network"]["n_input"]
        side = int(np.sqrt(n_input))
        n_neurons = w.shape[1]
        n_cols = 20
        n_rows = int(np.ceil(n_neurons / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 0.8, n_rows * 0.8))
        for idx in range(n_rows * n_cols):
            row, col = divmod(idx, n_cols)
            ax = axes[row, col] if n_rows > 1 else axes[col]
            if idx < n_neurons:
                rf = w[:, idx].reshape(side, side)
                ax.imshow(rf, cmap="hot", interpolation="nearest", vmin=0, vmax=w.max())
            ax.axis("off")
        plt.tight_layout()
        wandb.log({"weight_maps": wandb.Image(fig), "sample_step": self.global_step})
        plt.close(fig)

    def finish(self):
        """Clean up wandb run."""
        if self.use_wandb:
            wandb.finish()

    def _save_checkpoint(self):
        """Save training checkpoint."""
        ckpt_dir = self.config["experiment"]["checkpoint_dir"]
        path = os.path.join(ckpt_dir, f"epoch_{self.epoch + 1}.pt")
        state = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "weights": self.topology.weights.cpu(),
            "homeostasis_theta": self.homeostasis.theta.cpu(),
            "config": self.config,
        }
        if hasattr(self.neurons, "theta"):
            state["neuron_theta"] = self.neurons.theta.cpu()
        save_checkpoint(state, path)
        self.logger.info(f"  Checkpoint saved: {path}")

    def evaluate(self, train_data: list, test_data: list, n_presentations: int = 1) -> float:
        """Evaluate the network using neuron labeling.

        Args:
            train_data: Training data for labeling neurons.
            test_data: Test data for computing accuracy.
            n_presentations: Number of repeated test presentations.

        Returns:
            Test accuracy.
        """
        self.logger.info("Evaluating...")

        # Label neurons using training data spike responses
        labeler = NeuronLabeler(
            n_neurons=self.config["network"]["n_excitatory"],
            n_classes=10,
        )

        self.logger.info("  Labeling neurons from training data...")
        for image, label in tqdm(train_data, desc="Labeling"):
            spike_train = self.encoder.encode(image)
            self.neurons.reset()
            self.synapse.reset()

            spike_count = torch.zeros(self.config["network"]["n_excitatory"], device=self.device)
            for t in range(spike_train.shape[0]):
                exc_current = self.synapse.compute_current(
                    spike_train[t], self.topology.weights
                )
                post_spikes = self.neurons.step(exc_current)
                spike_count += post_spikes

            labeler.record(spike_count, label)

        labeler.assign_labels()
        self.logger.info(f"  Neuron labels: {labeler.labels[:20].tolist()}...")

        # Evaluate on test data
        self.logger.info("  Computing test accuracy...")
        correct = 0
        total = 0

        for image, label in tqdm(test_data, desc="Testing"):
            votes = torch.zeros(10, device=self.device)

            for _ in range(n_presentations):
                spike_train = self.encoder.encode(image)
                self.neurons.reset()
                self.synapse.reset()

                spike_count = torch.zeros(
                    self.config["network"]["n_excitatory"], device=self.device
                )
                for t in range(spike_train.shape[0]):
                    exc_current = self.synapse.compute_current(
                        spike_train[t], self.topology.weights
                    )
                    post_spikes = self.neurons.step(exc_current)
                    spike_count += post_spikes

                # Vote: each neuron's spikes count toward its assigned class
                for c in range(10):
                    mask = labeler.labels == c
                    votes[c] += spike_count[mask].sum()

            predicted = votes.argmax().item()
            if predicted == label:
                correct += 1
            total += 1

        accuracy = correct / total
        self.logger.info(f"  Test accuracy: {accuracy * 100:.2f}%")

        # Wandb evaluation logging
        if self.use_wandb:
            label_dist = labeler.get_label_distribution()
            log_dict = {
                "epoch": self.epoch + 1,
                "eval/accuracy": accuracy,
            }
            # Per-class neuron counts
            for c, count in label_dist.items():
                log_dict[f"eval/neurons_class_{c}"] = count

            # Confusion matrix via predictions
            all_preds = []
            all_labels = []
            for image, label in test_data:
                spike_train = self.encoder.encode(image)
                self.neurons.reset()
                self.synapse.reset()
                spike_count = torch.zeros(self.config["network"]["n_excitatory"], device=self.device)
                for t in range(spike_train.shape[0]):
                    exc_current = self.synapse.compute_current(spike_train[t], self.topology.weights)
                    post_spikes = self.neurons.step(exc_current)
                    spike_count += post_spikes
                pred = labeler.predict(spike_count)
                all_preds.append(pred)
                all_labels.append(label)

            log_dict["eval/confusion_matrix"] = wandb.plot.confusion_matrix(
                y_true=all_labels, preds=all_preds,
                class_names=[str(i) for i in range(10)],
            )
            wandb.log(log_dict)

        return accuracy

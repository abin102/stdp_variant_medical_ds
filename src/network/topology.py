import torch
import numpy as np


class NetworkTopology:
    """Builds and manages the network architecture.

    Architecture:
        Input (n_input) → Excitatory (n_excitatory) → Inhibitory (n_inhibitory)
        Inhibitory → Excitatory (lateral inhibition, all-but-self)

    Weights:
        - Input→Excitatory: learned via STDP, shape [n_input, n_excitatory]
        - Excitatory→Inhibitory: fixed one-to-one
        - Inhibitory→Excitatory: fixed all-to-all-but-self
    """

    def __init__(self, config: dict, device: torch.device = None):
        self.device = device or torch.device("cpu")

        net_cfg = config["network"]
        syn_cfg = config["synapse"]

        self.n_input = net_cfg["n_input"]
        self.n_excitatory = net_cfg["n_excitatory"]
        self.n_inhibitory = net_cfg["n_inhibitory"]
        self.inhibition_strength = net_cfg.get("inhibition_strength", 17.0)

        # Initialize input→excitatory weights
        w_init = syn_cfg.get("w_init", "random_uniform")
        self.w_min = syn_cfg.get("w_min", 0.0)
        self.w_max = syn_cfg.get("w_max", 1.0)

        self.weights = self._init_weights(w_init)

    def _init_weights(self, method: str) -> torch.Tensor:
        """Initialize input→excitatory weight matrix."""
        if method == "random_uniform":
            w = torch.rand(self.n_input, self.n_excitatory, device=self.device)
            w = w * (self.w_max - self.w_min) + self.w_min
        elif method == "random_normal":
            w = torch.randn(self.n_input, self.n_excitatory, device=self.device)
            w = w * 0.1 + 0.5  # mean 0.5, std 0.1
            w = w.clamp(self.w_min, self.w_max)
        else:
            raise ValueError(f"Unknown weight init method: {method}")

        # Normalize: each post-synaptic neuron receives same total weight
        col_sums = w.sum(dim=0, keepdim=True)
        w = w / col_sums * (self.n_input * 0.5)  # target sum
        w = w.clamp(self.w_min, self.w_max)

        return w

    def normalize_weights(self):
        """Normalize weights so each column sums to a target value."""
        target_sum = self.n_input * 0.5
        col_sums = self.weights.sum(dim=0, keepdim=True)
        col_sums = col_sums.clamp(min=1e-8)
        self.weights = self.weights / col_sums * target_sum
        self.weights = self.weights.clamp(self.w_min, self.w_max)

    def get_excitatory_to_inhibitory(self) -> torch.Tensor:
        """One-to-one connection from excitatory to inhibitory neurons."""
        return torch.eye(
            self.n_excitatory, self.n_inhibitory, device=self.device
        )

    def get_inhibitory_to_excitatory(self) -> torch.Tensor:
        """All-to-all-but-self inhibitory connections.

        Returns:
            Connection matrix [n_inhibitory, n_excitatory] where
            diagonal is 0 (no self-inhibition).
        """
        w = torch.ones(
            self.n_inhibitory, self.n_excitatory, device=self.device
        ) * self.inhibition_strength
        # Remove self-connections (assumes n_inhibitory == n_excitatory)
        n = min(self.n_inhibitory, self.n_excitatory)
        w[:n, :n].fill_diagonal_(0.0)
        return w

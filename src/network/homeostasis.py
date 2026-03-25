import torch


class Homeostasis:
    """Homeostatic mechanisms for stable learning.

    Implements:
    1. Adaptive threshold: neurons that fire too much get higher thresholds.
    2. Weight normalization: keeps total synaptic input bounded.
    """

    def __init__(
        self,
        shape,
        tau_theta: float = 1e7,
        theta_increment: float = 0.05,
        target_rate: float = None,
        dt: float = 1.0,
        device: torch.device = None,
    ):
        self.shape = (shape,) if isinstance(shape, int) else tuple(shape)
        self.n_neurons = shape if isinstance(shape, int) else torch.prod(torch.tensor(shape)).item()
        self.tau_theta = float(tau_theta)
        self.theta_increment = float(theta_increment)
        self.target_rate = target_rate
        self.dt = dt
        self.device = device or torch.device("cpu")

        self.theta = torch.zeros(self.shape, device=self.device)
        self.spike_counts = torch.zeros(self.shape, device=self.device)
        self.total_steps = 0

    def update_threshold(self, spikes: torch.Tensor):
        """Update adaptive threshold based on spiking activity.

        Args:
            spikes: Binary spike vector [n_neurons].
        """
        # Increase threshold for spiking neurons
        self.theta = self.theta + spikes * self.theta_increment

        # Exponential decay
        self.theta = self.theta * (1.0 - self.dt / self.tau_theta)

        # Track activity
        self.spike_counts = self.spike_counts + spikes
        self.total_steps += 1

    def get_effective_threshold(self, base_threshold: float) -> torch.Tensor:
        """Return per-neuron effective threshold.

        Args:
            base_threshold: Base threshold voltage.

        Returns:
            Effective threshold [n_neurons].
        """
        return base_threshold + self.theta

    def reset_counts(self):
        """Reset spike counts (e.g., between epochs)."""
        self.spike_counts.zero_()
        self.total_steps = 0

    @staticmethod
    def normalize_weights(
        weights: torch.Tensor, target_sum: float = None, w_min: float = 0.0, w_max: float = 1.0
    ) -> torch.Tensor:
        """Normalize weight columns to a target sum.

        Args:
            weights: Weight matrix [n_pre, n_post].
            target_sum: Target column sum. If None, uses n_pre * 0.5.
            w_min, w_max: Weight bounds.

        Returns:
            Normalized weight matrix.
        """
        if target_sum is None:
            target_sum = weights.shape[0] * 0.5

        col_sums = weights.sum(dim=0, keepdim=True).clamp(min=1e-8)
        weights = weights / col_sums * target_sum
        return weights.clamp(w_min, w_max)

import torch

from .base import BaseSynapse


class CurrentSynapse(BaseSynapse):
    """Current-based synapse (instantaneous).

    The post-synaptic current at each timestep is simply the weighted
    sum of pre-synaptic spikes — no temporal filtering.
    """

    def __init__(self, dt: float = 1.0, device: torch.device = None):
        super().__init__(dt, device)

    def reset(self):
        pass  # No state to reset

    def compute_current(
        self, pre_spikes: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """Compute instantaneous post-synaptic current.

        Args:
            pre_spikes: [n_pre] binary spikes.
            weights: [n_pre, n_post] weight matrix.

        Returns:
            Post-synaptic current [n_post].
        """
        return pre_spikes @ weights

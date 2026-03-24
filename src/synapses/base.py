from abc import ABC, abstractmethod

import torch


class BaseSynapse(ABC):
    """Abstract base class for synapse models.

    Converts pre-synaptic spikes + weights into post-synaptic current.
    """

    def __init__(self, dt: float = 1.0, device: torch.device = None):
        self.dt = dt
        self.device = device or torch.device("cpu")

    @abstractmethod
    def reset(self):
        """Reset synaptic state."""
        pass

    @abstractmethod
    def compute_current(
        self, pre_spikes: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """Compute post-synaptic current from pre-synaptic spikes.

        Args:
            pre_spikes: Binary spike vector [n_pre].
            weights: Weight matrix [n_pre, n_post].

        Returns:
            Post-synaptic current [n_post].
        """
        pass

from abc import ABC, abstractmethod

import torch


class BaseNeuron(ABC):
    """Abstract base class for spiking neuron models.

    All neuron models must implement reset() and step() methods.
    The state is maintained as tensors for vectorized simulation.
    """

    def __init__(self, n_neurons: int, dt: float = 1.0, device: torch.device = None):
        self.n_neurons = n_neurons
        self.dt = dt
        self.device = device or torch.device("cpu")

    @abstractmethod
    def reset(self):
        """Reset neuron state to initial conditions."""
        pass

    @abstractmethod
    def step(self, input_current: torch.Tensor) -> torch.Tensor:
        """Advance one timestep given input current.

        Args:
            input_current: Input current to each neuron [n_neurons].

        Returns:
            Binary spike tensor [n_neurons] (1.0 where neuron fired).
        """
        pass

    @abstractmethod
    def get_state(self) -> dict:
        """Return current neuron state as a dictionary."""
        pass

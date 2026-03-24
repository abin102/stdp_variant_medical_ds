import torch

from .base import BaseSynapse


class ConductanceSynapse(BaseSynapse):
    """Conductance-based synapse with exponential decay.

    Excitatory conductance:
        dg_e/dt = -g_e / tau_e + sum(w_ij * spike_j)

    The total synaptic current is modeled as instantaneous weighted input
    passed through an exponential filter (synaptic conductance trace).
    """

    def __init__(
        self,
        n_post: int,
        tau_excitatory: float = 1.0,
        tau_inhibitory: float = 2.0,
        dt: float = 1.0,
        device: torch.device = None,
    ):
        super().__init__(dt, device)
        self.n_post = n_post
        self.tau_excitatory = tau_excitatory
        self.tau_inhibitory = tau_inhibitory

        self.decay_e = 1.0 - dt / tau_excitatory
        self.decay_i = 1.0 - dt / tau_inhibitory

        self.g_e = None  # excitatory conductance [n_post]
        self.g_i = None  # inhibitory conductance [n_post]
        self.reset()

    def reset(self):
        self.g_e = torch.zeros(self.n_post, device=self.device)
        self.g_i = torch.zeros(self.n_post, device=self.device)

    def compute_current(
        self, pre_spikes: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """Compute excitatory post-synaptic current.

        Args:
            pre_spikes: [n_pre] binary spikes.
            weights: [n_pre, n_post] weight matrix.

        Returns:
            Post-synaptic current [n_post].
        """
        # Decay conductance
        self.g_e = self.g_e * self.decay_e

        # Add new input: weighted sum of pre-synaptic spikes
        self.g_e = self.g_e + pre_spikes @ weights

        return self.g_e

    def compute_inhibitory_current(
        self, inhibitory_input: torch.Tensor
    ) -> torch.Tensor:
        """Compute inhibitory current from lateral inhibition.

        Args:
            inhibitory_input: Inhibitory input [n_post].

        Returns:
            Inhibitory current [n_post].
        """
        self.g_i = self.g_i * self.decay_i
        self.g_i = self.g_i + inhibitory_input
        return -self.g_i

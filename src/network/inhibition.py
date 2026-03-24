import torch


class LateralInhibition:
    """Lateral inhibition among excitatory neurons via inhibitory population.

    When an excitatory neuron spikes, the corresponding inhibitory neuron
    fires (one-to-one), which then inhibits all other excitatory neurons
    (all-to-all-but-self). This implements competitive learning (WTA-like).
    """

    def __init__(
        self,
        n_excitatory: int,
        n_inhibitory: int,
        inhibition_strength: float = 17.0,
        device: torch.device = None,
    ):
        self.n_excitatory = n_excitatory
        self.n_inhibitory = n_inhibitory
        self.inhibition_strength = inhibition_strength
        self.device = device or torch.device("cpu")

        # Inhibitory connection matrix: all-to-all-but-self
        self.w_inh = torch.ones(
            n_inhibitory, n_excitatory, device=self.device
        ) * inhibition_strength
        n = min(n_inhibitory, n_excitatory)
        self.w_inh[:n, :n].fill_diagonal_(0.0)

    def compute_inhibition(self, excitatory_spikes: torch.Tensor) -> torch.Tensor:
        """Compute inhibitory input to excitatory neurons.

        The inhibitory population mirrors excitatory spikes (one-to-one),
        then broadcasts inhibition to all others.

        Args:
            excitatory_spikes: Binary spikes [n_excitatory].

        Returns:
            Inhibitory current to excitatory neurons [n_excitatory].
        """
        # Inhibitory neurons fire when their paired excitatory neuron fires
        inhibitory_spikes = excitatory_spikes[:self.n_inhibitory]

        # Compute inhibitory current to excitatory population
        inhibitory_current = inhibitory_spikes @ self.w_inh

        return inhibitory_current

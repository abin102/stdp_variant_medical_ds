import torch

from .base import BaseNeuron


class LIFNeuron(BaseNeuron):
    """Leaky Integrate-and-Fire neuron model.

    Membrane dynamics:
        tau_m * dV/dt = -(V - V_rest) + R * I(t)

    Discretized (Euler):
        V[t+1] = V[t] + dt/tau_m * (-(V[t] - V_rest) + I[t])

    When V >= V_thresh: spike, then V = V_reset for refractory_period ms.
    """

    def __init__(
        self,
        n_neurons: int,
        v_rest: float = -65.0,
        v_reset: float = -65.0,
        v_thresh: float = -52.0,
        tau_membrane: float = 100.0,
        refractory_period: float = 5.0,
        dt: float = 1.0,
        device: torch.device = None,
    ):
        super().__init__(n_neurons, dt, device)
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.v_thresh = v_thresh
        self.tau_membrane = tau_membrane
        self.refractory_period = refractory_period

        # Pre-allocate scalar tensors to avoid per-step allocation
        self._v_reset_t = torch.tensor(self.v_reset, dtype=torch.float32, device=self.device)
        self._refrac_t = torch.tensor(self.refractory_period, dtype=torch.float32, device=self.device)

        # State tensors
        self.v = None
        self.refractory_timer = None
        self.spikes = None
        self.reset()

    def reset(self):
        """Reset all neurons to resting state."""
        self.v = torch.full(
            (self.n_neurons,), self.v_rest, dtype=torch.float32, device=self.device
        )
        self.refractory_timer = torch.zeros(
            self.n_neurons, dtype=torch.float32, device=self.device
        )
        self.spikes = torch.zeros(
            self.n_neurons, dtype=torch.float32, device=self.device
        )

    def step(self, input_current: torch.Tensor) -> torch.Tensor:
        """Advance one timestep.

        Args:
            input_current: Synaptic + external current [n_neurons].

        Returns:
            Binary spike tensor [n_neurons].
        """
        # Determine which neurons are not in refractory period
        not_refractory = self.refractory_timer <= 0

        # Update membrane potential (only for non-refractory neurons)
        dv = (self.dt / self.tau_membrane) * (
            -(self.v - self.v_rest) + input_current
        )
        self.v = torch.where(not_refractory, self.v + dv, self.v)

        # Check for spikes
        self.spikes = (self.v >= self.v_thresh).float()

        # Reset spiking neurons
        spiked = self.spikes.bool()
        self.v = torch.where(spiked, self._v_reset_t, self.v)

        # Set refractory timer for spiking neurons
        self.refractory_timer = torch.where(spiked, self._refrac_t, self.refractory_timer)

        # Decrement refractory timer
        self.refractory_timer = (self.refractory_timer - self.dt).clamp(min=0)

        return self.spikes

    def get_state(self) -> dict:
        return {
            "v": self.v.clone(),
            "refractory_timer": self.refractory_timer.clone(),
            "spikes": self.spikes.clone(),
        }

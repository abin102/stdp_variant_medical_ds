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
        shape,
        v_rest: float = -65.0,
        v_reset: float = -65.0,
        v_thresh: float = -52.0,
        tau_membrane: float = 100.0,
        refractory_period: float = 5.0,
        dt: float = 1.0,
        device: torch.device = None,
    ):
        super().__init__(shape, dt, device)
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.v_thresh = v_thresh
        self.tau_membrane = tau_membrane
        self.refractory_period = refractory_period

        # Pre-compute constant
        self._dt_over_tau = dt / tau_membrane

        # State tensors
        self.v = None
        self.refractory_timer = None
        self.spikes = None
        self.reset()

    def reset(self):
        """Reset all neurons to resting state."""
        self.v = torch.full(
            self.shape, self.v_rest, dtype=torch.float32, device=self.device
        )
        self.refractory_timer = torch.zeros(
            self.shape, dtype=torch.float32, device=self.device
        )
        self.spikes = torch.zeros(
            self.shape, dtype=torch.float32, device=self.device
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
        dv = (self._dt_over_tau) * (
            input_current - self.v + self.v_rest
        )
        self.v.add_(dv * not_refractory)

        # Check for spikes
        spiked = self.v >= self.v_thresh
        self.spikes = spiked.float()

        # Reset spiking neurons and set refractory timer
        self.v[spiked] = self.v_reset
        self.refractory_timer[spiked] = self.refractory_period

        # Decrement refractory timer
        self.refractory_timer.sub_(self.dt).clamp_(min=0)

        return self.spikes

    def get_state(self) -> dict:
        return {
            "v": self.v.clone(),
            "refractory_timer": self.refractory_timer.clone(),
            "spikes": self.spikes.clone(),
        }

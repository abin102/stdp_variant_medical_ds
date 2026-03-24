import torch

from .base import BaseNeuron


class AdaptiveLIFNeuron(BaseNeuron):
    """LIF neuron with adaptive (activity-dependent) threshold.

    The effective threshold for each neuron is:
        V_thresh_eff = V_thresh + theta

    After each spike, theta increases by theta_increment.
    Between spikes, theta decays exponentially:
        d(theta)/dt = -theta / tau_theta
    """

    def __init__(
        self,
        n_neurons: int,
        v_rest: float = -65.0,
        v_reset: float = -65.0,
        v_thresh: float = -52.0,
        tau_membrane: float = 100.0,
        refractory_period: float = 5.0,
        tau_theta: float = 1e7,
        theta_increment: float = 0.05,
        dt: float = 1.0,
        device: torch.device = None,
    ):
        super().__init__(n_neurons, dt, device)
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.v_thresh = v_thresh
        self.tau_membrane = tau_membrane
        self.refractory_period = refractory_period
        self.tau_theta = tau_theta
        self.theta_increment = theta_increment

        # Pre-allocate scalar tensors
        self._v_reset_t = torch.tensor(self.v_reset, dtype=torch.float32, device=self.device)
        self._refrac_t = torch.tensor(self.refractory_period, dtype=torch.float32, device=self.device)

        self.v = None
        self.theta = None
        self.refractory_timer = None
        self.spikes = None
        self.reset()

    def reset(self):
        self.v = torch.full(
            (self.n_neurons,), self.v_rest, dtype=torch.float32, device=self.device
        )
        self.theta = torch.zeros(
            self.n_neurons, dtype=torch.float32, device=self.device
        )
        self.refractory_timer = torch.zeros(
            self.n_neurons, dtype=torch.float32, device=self.device
        )
        self.spikes = torch.zeros(
            self.n_neurons, dtype=torch.float32, device=self.device
        )

    def step(self, input_current: torch.Tensor) -> torch.Tensor:
        not_refractory = self.refractory_timer <= 0

        # Membrane dynamics
        dv = (self.dt / self.tau_membrane) * (
            -(self.v - self.v_rest) + input_current
        )
        self.v = torch.where(not_refractory, self.v + dv, self.v)

        # Adaptive threshold
        effective_thresh = self.v_thresh + self.theta

        # Spike detection
        self.spikes = (self.v >= effective_thresh).float()

        # Reset spiking neurons
        spiked = self.spikes.bool()
        self.v = torch.where(spiked, self._v_reset_t, self.v)

        # Increase theta for spiking neurons
        self.theta = self.theta + self.spikes * self.theta_increment

        # Decay theta
        self.theta = self.theta * (1.0 - self.dt / self.tau_theta)

        # Refractory period
        self.refractory_timer = torch.where(spiked, self._refrac_t, self.refractory_timer)
        self.refractory_timer = (self.refractory_timer - self.dt).clamp(min=0)

        return self.spikes

    def get_state(self) -> dict:
        return {
            "v": self.v.clone(),
            "theta": self.theta.clone(),
            "refractory_timer": self.refractory_timer.clone(),
            "spikes": self.spikes.clone(),
        }

import torch

from .base import BaseSTDP


class VoltageSTDP(BaseSTDP):
    """Clopath et al. voltage-based STDP rule.

    Uses the post-synaptic membrane voltage instead of post-synaptic spikes
    for computing weight changes.

    LTD: dW- = -A_ltd * x(t) * (v_minus(t) - theta_minus)_+
    LTP: dW+ = A_ltp * x(t) * (v(t) - theta_plus)_+ * (v_minus(t) - theta_minus)_+

    where:
        x(t): pre-synaptic trace (decays with tau_plus)
        v_minus(t): low-pass filtered voltage (decays with tau_v_minus)
        (.)_+ : rectification (max(0, .))
    """

    def __init__(
        self,
        n_pre: int,
        n_post: int,
        tau_lowpass: float = 10.0,
        tau_v_minus: float = 10.0,
        theta_minus: float = -70.0,
        theta_plus: float = -49.0,
        A_ltd: float = 0.0001,
        A_ltp: float = 0.0002,
        tau_plus: float = 16.8,
        dt: float = 1.0,
        w_min: float = 0.0,
        w_max: float = 1.0,
        weight_dependence: str = "none",
        mu_plus: float = 0.0,
        mu_minus: float = 0.0,
        device: torch.device = None,
    ):
        super().__init__(n_pre, n_post, w_min, w_max, weight_dependence, mu_plus, mu_minus, device)
        self.tau_lowpass = tau_lowpass
        self.tau_v_minus = tau_v_minus
        self.theta_minus = theta_minus
        self.theta_plus = theta_plus
        self.A_ltd = A_ltd
        self.A_ltp = A_ltp
        self.tau_plus = tau_plus
        self.dt = dt

        self.decay_pre = 1.0 - dt / tau_plus
        self.decay_v_minus = 1.0 - dt / tau_v_minus

        self.trace_pre = None    # [n_pre]
        self.v_minus = None      # [n_post] low-pass filtered voltage
        self.reset_traces()

    def reset_traces(self):
        self.trace_pre = torch.zeros(self.n_pre, device=self.device)
        self.v_minus = torch.full((self.n_post,), -65.0, device=self.device)

    def update(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Fallback: use post spikes as proxy for voltage threshold crossing."""
        # Approximate voltage from spikes for compatibility
        v_proxy = torch.where(
            post_spikes.bool(),
            torch.tensor(self.theta_plus + 10.0, device=self.device),
            torch.tensor(-65.0, device=self.device),
        )
        return self.update_with_voltage(pre_spikes, post_spikes, weights, v_proxy)

    def update_with_voltage(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        weights: torch.Tensor,
        v_post: torch.Tensor,
    ) -> torch.Tensor:
        """Compute voltage-based STDP weight update.

        Args:
            pre_spikes: [n_pre] binary.
            post_spikes: [n_post] binary (used for LTP gating).
            weights: [n_pre, n_post].
            v_post: [n_post] current membrane voltage.

        Returns:
            dW [n_pre, n_post].
        """
        # Decay traces
        self.trace_pre = self.trace_pre * self.decay_pre
        self.trace_pre = self.trace_pre + pre_spikes

        # Update low-pass filtered voltage
        self.v_minus = self.v_minus * self.decay_v_minus + v_post * (1.0 - self.decay_v_minus)

        # Rectified terms
        v_minus_rect = (self.v_minus - self.theta_minus).clamp(min=0)  # [n_post]
        v_plus_rect = (v_post - self.theta_plus).clamp(min=0)          # [n_post]

        # LTD: -A_ltd * x_pre * (v_minus - theta_minus)_+
        dw_minus = self.A_ltd * torch.outer(self.trace_pre, v_minus_rect)

        # LTP: A_ltp * x_pre * (v - theta_plus)_+ * (v_minus - theta_minus)_+
        ltp_post = v_plus_rect * v_minus_rect  # [n_post]
        dw_plus = self.A_ltp * torch.outer(self.trace_pre, ltp_post)

        # Apply weight dependence
        dw_plus = self.apply_weight_dependence(dw_plus, weights, is_ltp=True)
        dw_minus = self.apply_weight_dependence(dw_minus, weights, is_ltp=False)

        dw = dw_plus - dw_minus
        return dw

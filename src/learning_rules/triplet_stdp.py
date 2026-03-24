import torch

from .base import BaseSTDP


class TripletSTDP(BaseSTDP):
    """Pfister-Gerstner triplet STDP rule.

    Extends pair-based STDP with slow traces that capture higher-order
    spike correlations:

    Traces (4 total):
        x1: pre fast trace  (tau_plus)  — pair term
        x2: pre slow trace  (tau_x)     — triplet term
        y1: post fast trace (tau_minus)  — pair term
        y2: post slow trace (tau_y)      — triplet term

    Weight updates:
        On post spike: dW += (A2_plus + A3_plus * x2(t-eps)) * x1(t)
        On pre spike:  dW -= (A2_minus + A3_minus * y2(t-eps)) * y1(t)

    The (t-eps) notation means the slow trace value just before the spike update.
    """

    def __init__(
        self,
        n_pre: int,
        n_post: int,
        tau_plus: float = 16.8,
        tau_minus: float = 33.7,
        tau_x: float = 101.0,
        tau_y: float = 125.0,
        A2_plus: float = 0.0046,
        A2_minus: float = 0.003,
        A3_plus: float = 0.0091,
        A3_minus: float = 0.0,
        dt: float = 1.0,
        interaction: str = "all_to_all",
        w_min: float = 0.0,
        w_max: float = 1.0,
        weight_dependence: str = "none",
        mu_plus: float = 0.0,
        mu_minus: float = 0.0,
        device: torch.device = None,
    ):
        super().__init__(n_pre, n_post, w_min, w_max, weight_dependence, mu_plus, mu_minus, device)
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.tau_x = tau_x
        self.tau_y = tau_y
        self.A2_plus = A2_plus
        self.A2_minus = A2_minus
        self.A3_plus = A3_plus
        self.A3_minus = A3_minus
        self.dt = dt
        self.interaction = interaction

        # Decay factors
        self.decay_pre_fast = 1.0 - dt / tau_plus
        self.decay_post_fast = 1.0 - dt / tau_minus
        self.decay_pre_slow = 1.0 - dt / tau_x
        self.decay_post_slow = 1.0 - dt / tau_y

        self.trace_pre_fast = None   # x1 [n_pre]
        self.trace_pre_slow = None   # x2 [n_pre]
        self.trace_post_fast = None  # y1 [n_post]
        self.trace_post_slow = None  # y2 [n_post]
        self.reset_traces()

    def reset_traces(self):
        self.trace_pre_fast = torch.zeros(self.n_pre, device=self.device)
        self.trace_pre_slow = torch.zeros(self.n_pre, device=self.device)
        self.trace_post_fast = torch.zeros(self.n_post, device=self.device)
        self.trace_post_slow = torch.zeros(self.n_post, device=self.device)

    def update(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Compute triplet STDP weight update.

        Returns:
            dW [n_pre, n_post].
        """
        # Decay all traces
        self.trace_pre_fast = self.trace_pre_fast * self.decay_pre_fast
        self.trace_post_fast = self.trace_post_fast * self.decay_post_fast
        self.trace_pre_slow = self.trace_pre_slow * self.decay_pre_slow
        self.trace_post_slow = self.trace_post_slow * self.decay_post_slow

        # Save slow traces before spike update (t-epsilon)
        x2_before = self.trace_pre_slow.clone()
        y2_before = self.trace_post_slow.clone()

        # Update traces with current spikes
        if self.interaction == "nearest_spike":
            self.trace_pre_fast = torch.where(
                pre_spikes.bool(), torch.ones_like(self.trace_pre_fast), self.trace_pre_fast
            )
            self.trace_post_fast = torch.where(
                post_spikes.bool(), torch.ones_like(self.trace_post_fast), self.trace_post_fast
            )
            self.trace_pre_slow = torch.where(
                pre_spikes.bool(), torch.ones_like(self.trace_pre_slow), self.trace_pre_slow
            )
            self.trace_post_slow = torch.where(
                post_spikes.bool(), torch.ones_like(self.trace_post_slow), self.trace_post_slow
            )
        else:
            self.trace_pre_fast = self.trace_pre_fast + pre_spikes
            self.trace_post_fast = self.trace_post_fast + post_spikes
            self.trace_pre_slow = self.trace_pre_slow + pre_spikes
            self.trace_post_slow = self.trace_post_slow + post_spikes

        # LTP: on post spike
        # dW += (A2_plus + A3_plus * x2_before[i]) * x1[i] * post_spike[j]
        ltp_amplitude = self.A2_plus + self.A3_plus * x2_before  # [n_pre]
        dw_plus = torch.outer(ltp_amplitude * self.trace_pre_fast, post_spikes)

        # LTD: on pre spike
        # dW -= (A2_minus + A3_minus * y2_before[j]) * y1[j] * pre_spike[i]
        ltd_amplitude = self.A2_minus + self.A3_minus * y2_before  # [n_post]
        dw_minus = torch.outer(pre_spikes, ltd_amplitude * self.trace_post_fast)

        # Apply weight dependence
        dw_plus = self.apply_weight_dependence(dw_plus, weights, is_ltp=True)
        dw_minus = self.apply_weight_dependence(dw_minus, weights, is_ltp=False)

        dw = dw_plus - dw_minus
        return dw

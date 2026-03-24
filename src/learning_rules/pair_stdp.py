import torch

from .base import BaseSTDP


class PairSTDP(BaseSTDP):
    """Classic pair-based STDP rule.

    Maintains pre and post traces that are updated on each spike:
        x(t) = x(t) * exp(-dt/tau_plus) + pre_spike(t)
        y(t) = y(t) * exp(-dt/tau_minus) + post_spike(t)

    Weight updates:
        On post spike: dW += A2_plus * x (LTP)
        On pre spike:  dW -= A2_minus * y (LTD)

    Supports "nearest_spike" or "all_to_all" interaction modes.
    """

    def __init__(
        self,
        n_pre: int,
        n_post: int,
        tau_plus: float = 16.8,
        tau_minus: float = 33.7,
        A2_plus: float = 0.0046,
        A2_minus: float = 0.003,
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
        self.A2_plus = A2_plus
        self.A2_minus = A2_minus
        self.dt = dt
        self.interaction = interaction

        # Decay factors
        self.decay_pre = 1.0 - dt / tau_plus
        self.decay_post = 1.0 - dt / tau_minus

        # Traces
        self.trace_pre = None   # [n_pre]
        self.trace_post = None  # [n_post]
        self.reset_traces()

    def reset_traces(self):
        self.trace_pre = torch.zeros(self.n_pre, device=self.device)
        self.trace_post = torch.zeros(self.n_post, device=self.device)

    def update(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Compute pair-based STDP weight update.

        Returns:
            dW [n_pre, n_post].
        """
        # Decay traces
        self.trace_pre = self.trace_pre * self.decay_pre
        self.trace_post = self.trace_post * self.decay_post

        # Update traces with current spikes
        if self.interaction == "nearest_spike":
            # Reset trace on spike (nearest-spike pairing)
            self.trace_pre = torch.where(pre_spikes.bool(), torch.ones_like(self.trace_pre), self.trace_pre)
            self.trace_post = torch.where(post_spikes.bool(), torch.ones_like(self.trace_post), self.trace_post)
        else:
            # all_to_all: accumulate
            self.trace_pre = self.trace_pre + pre_spikes
            self.trace_post = self.trace_post + post_spikes

        # Compute weight changes
        # LTP: post spike × pre trace -> potentiation
        # dw_plus[i,j] = A2_plus * trace_pre[i] * post_spikes[j]
        dw_plus = self.A2_plus * torch.outer(self.trace_pre, post_spikes)

        # LTD: pre spike × post trace -> depression
        # dw_minus[i,j] = A2_minus * pre_spikes[i] * trace_post[j]
        dw_minus = self.A2_minus * torch.outer(pre_spikes, self.trace_post)

        # Apply weight dependence
        dw_plus = self.apply_weight_dependence(dw_plus, weights, is_ltp=True)
        dw_minus = self.apply_weight_dependence(dw_minus, weights, is_ltp=False)

        dw = dw_plus - dw_minus
        return dw

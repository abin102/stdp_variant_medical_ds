import torch

from .base import BaseSTDP


class ProbabilisticSTDP(BaseSTDP):
    """Probabilistic STDP rule.

    Like pair-based STDP but weight updates are applied stochastically:
    each synapse is updated with probability p_update, and the update
    magnitude is scaled by a temperature parameter.

    This models biological variability in synaptic plasticity and can
    act as a regularizer.
    """

    def __init__(
        self,
        n_pre: int,
        n_post: int,
        tau_plus: float = 16.8,
        tau_minus: float = 33.7,
        A2_plus: float = 0.0046,
        A2_minus: float = 0.003,
        p_update: float = 0.5,
        temperature: float = 1.0,
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
        self.p_update = p_update
        self.temperature = temperature
        self.dt = dt
        self.interaction = interaction

        self.decay_pre = 1.0 - dt / tau_plus
        self.decay_post = 1.0 - dt / tau_minus

        self.trace_pre = None
        self.trace_post = None
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
        """Compute probabilistic STDP weight update.

        Returns:
            dW [n_pre, n_post].
        """
        # Decay traces
        self.trace_pre = self.trace_pre * self.decay_pre
        self.trace_post = self.trace_post * self.decay_post

        # Update traces
        if self.interaction == "nearest_spike":
            self.trace_pre = torch.where(pre_spikes.bool(), torch.ones_like(self.trace_pre), self.trace_pre)
            self.trace_post = torch.where(post_spikes.bool(), torch.ones_like(self.trace_post), self.trace_post)
        else:
            self.trace_pre = self.trace_pre + pre_spikes
            self.trace_post = self.trace_post + post_spikes

        # Deterministic weight changes (same as pair STDP)
        dw_plus = self.A2_plus * torch.outer(self.trace_pre, post_spikes)
        dw_minus = self.A2_minus * torch.outer(pre_spikes, self.trace_post)

        # Apply weight dependence
        dw_plus = self.apply_weight_dependence(dw_plus, weights, is_ltp=True)
        dw_minus = self.apply_weight_dependence(dw_minus, weights, is_ltp=False)

        dw = (dw_plus - dw_minus) * self.temperature

        # Stochastic mask: each synapse updated with probability p_update
        mask = torch.bernoulli(
            torch.full((self.n_pre, self.n_post), self.p_update, device=self.device)
        )

        return dw * mask

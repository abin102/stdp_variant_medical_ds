from abc import ABC, abstractmethod

import torch


class BaseSTDP(ABC):
    """Abstract base class for STDP learning rules.

    Maintains eligibility traces and provides the interface for weight updates.
    Subclasses implement specific STDP variants.
    """

    def __init__(
        self,
        n_pre: int,
        n_post: int,
        w_min: float = 0.0,
        w_max: float = 1.0,
        weight_dependence: str = "none",
        mu_plus: float = 0.0,
        mu_minus: float = 0.0,
        device: torch.device = None,
    ):
        self.n_pre = n_pre
        self.n_post = n_post
        self.w_min = w_min
        self.w_max = w_max
        self.weight_dependence = weight_dependence
        self.mu_plus = mu_plus
        self.mu_minus = mu_minus
        self.device = device or torch.device("cpu")

    @abstractmethod
    def reset_traces(self):
        """Reset all eligibility traces to zero."""
        pass

    @abstractmethod
    def update(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Compute weight change for one timestep.

        Args:
            pre_spikes: Binary pre-synaptic spikes [n_pre].
            post_spikes: Binary post-synaptic spikes [n_post].
            weights: Current weight matrix [n_pre, n_post].

        Returns:
            Weight change matrix dW [n_pre, n_post].
        """
        pass

    def apply_weight_dependence(
        self, dw: torch.Tensor, weights: torch.Tensor, is_ltp: bool
    ) -> torch.Tensor:
        """Apply weight-dependent scaling to weight changes.

        Args:
            dw: Raw weight change.
            weights: Current weights.
            is_ltp: True for potentiation, False for depression.

        Returns:
            Scaled weight change.
        """
        if self.weight_dependence == "none":
            return dw
        elif self.weight_dependence == "linear":
            if is_ltp:
                return dw * (self.w_max - weights)
            else:
                return dw * (weights - self.w_min)
        elif self.weight_dependence == "exponential":
            if is_ltp:
                return dw * ((self.w_max - weights) / (self.w_max - self.w_min)) ** self.mu_plus
            else:
                return dw * ((weights - self.w_min) / (self.w_max - self.w_min)) ** self.mu_minus
        else:
            raise ValueError(f"Unknown weight dependence: {self.weight_dependence}")

    def clamp_weights(self, weights: torch.Tensor) -> torch.Tensor:
        """Clamp weights to [w_min, w_max]."""
        return weights.clamp(self.w_min, self.w_max)

"""Spatial Winner-Take-All (WTA) inhibition for convolutional SNN layers.

At each spatial position (h, w), only the channel with the highest
membrane potential is allowed to spike. This enforces that each
spatial location activates at most one feature detector, creating
a competitive learning regime where filters specialize.
"""

import torch


class SpatialWTA:
    """Per-location winner-take-all across channels.

    Given post-synaptic spikes and membrane voltages of shape (C, H, W),
    keeps only the spike from the channel with the highest voltage at
    each spatial position. All other spikes at that position are suppressed.
    """

    def __init__(self, device: torch.device = None):
        self.device = device or torch.device("cpu")

    def __call__(
        self,
        spikes: torch.Tensor,
        membrane_voltages: torch.Tensor,
    ) -> torch.Tensor:
        """Apply spatial WTA filtering.

        Args:
            spikes: Binary spike tensor (C, H, W).
            membrane_voltages: Membrane voltage tensor (C, H, W).

        Returns:
            Filtered spikes (C, H, W) with at most 1 spike per (h, w).
        """
        # Mask voltages to only consider spiking neurons
        # Non-spiking neurons get -inf so they can't win
        masked_v = torch.where(
            spikes.bool(),
            membrane_voltages,
            torch.tensor(float("-inf"), device=self.device),
        )

        # Winner channel at each spatial position
        winners = masked_v.argmax(dim=0)  # (H, W)

        # Build output: only the winning channel keeps its spike
        C, H, W = spikes.shape
        channel_idx = torch.arange(C, device=self.device).view(C, 1, 1).expand(C, H, W)
        winner_mask = channel_idx == winners.unsqueeze(0)

        return spikes * winner_mask.float()

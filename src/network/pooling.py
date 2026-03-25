"""Spike-based max pooling for spiking convolutional networks.

Implements temporal first-spike pooling: within each pooling window,
the neuron that spikes first propagates its spike to the pooled output.
For accumulated spike counts (evaluation), standard max pooling applies.
"""

import torch
import torch.nn.functional as F


class SpikeMaxPool:
    """Max pooling layer for spike maps.

    During simulation: tracks first-spike-time per position and propagates
    the earliest spike within each pooling window.

    For evaluation on spike counts: applies standard F.max_pool2d.
    """

    def __init__(self, pool_size: int = 2, device: torch.device = None):
        self.pool_size = pool_size
        self.device = device or torch.device("cpu")

        # Track first spike times: large value = no spike yet
        self._first_spike_time = None
        self._timestep = 0
        self._output_shape = None

    def reset(self, input_shape: tuple):
        """Reset pooling state for a new sample.

        Args:
            input_shape: (C, H, W) shape of pre-pooling feature maps.
        """
        self._first_spike_time = torch.full(
            input_shape, float("inf"), device=self.device
        )
        self._timestep = 0
        C, H, W = input_shape
        H_out = H // self.pool_size
        W_out = W // self.pool_size
        self._output_shape = (C, H_out, W_out)

    def step(self, spikes: torch.Tensor) -> torch.Tensor:
        """Process one timestep of spikes through pooling.

        For each pool_size x pool_size window, propagate a spike only if
        the spiking neuron was the first to spike in that window.

        Args:
            spikes: Binary spike tensor (C, H, W).

        Returns:
            Pooled spikes (C, H_out, W_out).
        """
        # Record first spike times
        spiked = spikes.bool()
        newly_spiked = spiked & (self._first_spike_time == float("inf"))
        self._first_spike_time[newly_spiked] = float(self._timestep)
        self._timestep += 1

        # For each pooling window, find the position with earliest spike
        # If that position just spiked this timestep, propagate it
        C, H, W = spikes.shape
        ps = self.pool_size
        H_out, W_out = H // ps, W // ps

        pooled = torch.zeros(C, H_out, W_out, device=self.device)

        # Reshape to windows: (C, H_out, ps, W_out, ps)
        spike_windows = spikes[:, :H_out * ps, :W_out * ps].reshape(
            C, H_out, ps, W_out, ps
        )
        time_windows = self._first_spike_time[:, :H_out * ps, :W_out * ps].reshape(
            C, H_out, ps, W_out, ps
        )

        # For each window, check if any neuron spiked at this timestep
        # and it was the first spike in the window
        current_spikes = spike_windows.sum(dim=(2, 4))  # (C, H_out, W_out)

        # Min first-spike-time in each window
        min_times = time_windows.amin(dim=(2, 4))  # (C, H_out, W_out)

        # Propagate spike if the earliest spiker in the window fired this step
        pooled = ((min_times == float(self._timestep - 1)) & (current_spikes > 0)).float()

        return pooled

    @staticmethod
    def pool_spike_counts(spike_counts: torch.Tensor, pool_size: int = 2) -> torch.Tensor:
        """Apply standard max pooling to accumulated spike count maps.

        Used for feature extraction during evaluation.

        Args:
            spike_counts: Spike count tensor (C, H, W).
            pool_size: Pooling window size.

        Returns:
            Pooled spike counts (C, H_out, W_out).
        """
        return F.max_pool2d(
            spike_counts.unsqueeze(0), kernel_size=pool_size
        ).squeeze(0)

import torch


class LatencyEncoder:
    """Temporal/latency coding: higher intensity fires earlier.

    Each pixel fires exactly once. The spike time is inversely proportional
    to the pixel intensity. Pixels with intensity 0 do not fire.

    Supports both 1D (flattened) and spatial (C, H, W) inputs.
    """

    def __init__(
        self,
        time_window: int = 350,
        dt: float = 1.0,
        device: torch.device = None,
    ):
        self.time_window = time_window
        self.dt = dt
        self.n_timesteps = int(time_window / dt)
        self.device = device or torch.device("cpu")

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """Convert image to latency-coded spike train.

        Args:
            image: Pixel intensities with values in [0, 1].
                   Shape can be [n_pixels], [C, H, W], or [H, W].

        Returns:
            Spike train, binary.
                If input is [n_pixels]: returns [n_timesteps, n_pixels].
                If input is [C, H, W]: returns [n_timesteps, C, H, W].
                If input is [H, W]: returns [n_timesteps, H, W].
        """
        image = image.to(self.device)
        original_shape = image.shape

        # Flatten for vectorized processing
        flat = image.reshape(-1)
        n_pixels = flat.shape[0]

        # Only encode pixels with nonzero intensity
        active = flat > 0

        # Spike time: high intensity -> early spike
        spike_times = ((1.0 - flat) * (self.n_timesteps - 1)).long()
        spike_times = spike_times.clamp(0, self.n_timesteps - 1)

        # Vectorized spike placement (no for-loop)
        spikes = torch.zeros(self.n_timesteps, n_pixels, device=self.device)
        active_indices = torch.where(active)[0]
        if active_indices.numel() > 0:
            spikes[spike_times[active_indices], active_indices] = 1.0

        # Reshape back to original spatial dimensions
        return spikes.reshape(self.n_timesteps, *original_shape)

import torch


class LatencyEncoder:
    """Temporal/latency coding: higher intensity fires earlier.

    Each pixel fires exactly once. The spike time is inversely proportional
    to the pixel intensity. Pixels with intensity 0 do not fire.
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
        """Convert flattened image to latency-coded spike train.

        Args:
            image: Pixel intensities [n_pixels], values in [0, 1].

        Returns:
            Spike train [n_timesteps, n_pixels], binary.
        """
        image = image.to(self.device)
        n_pixels = image.shape[0]

        spikes = torch.zeros(self.n_timesteps, n_pixels, device=self.device)

        # Only encode pixels with nonzero intensity
        active = image > 0
        # Spike time: high intensity -> early spike
        # t_spike = (1 - intensity) * (n_timesteps - 1)
        spike_times = ((1.0 - image) * (self.n_timesteps - 1)).long()
        spike_times = spike_times.clamp(0, self.n_timesteps - 1)

        # Place spikes
        for i in range(n_pixels):
            if active[i]:
                spikes[spike_times[i], i] = 1.0

        return spikes

import torch


class RankOrderEncoder:
    """Rank order coding: neurons fire in order of decreasing intensity.

    At each timestep, the next-highest-intensity pixel fires. Only the
    top-k pixels fire (where k = n_timesteps or n_pixels, whichever is smaller).
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
        """Convert flattened image to rank-order spike train.

        Args:
            image: Pixel intensities [n_pixels], values in [0, 1].

        Returns:
            Spike train [n_timesteps, n_pixels], binary.
        """
        image = image.to(self.device)
        n_pixels = image.shape[0]

        spikes = torch.zeros(self.n_timesteps, n_pixels, device=self.device)

        # Sort pixels by intensity (descending)
        sorted_indices = torch.argsort(image, descending=True)

        # Only fire pixels with nonzero intensity
        n_active = min(self.n_timesteps, (image > 0).sum().item())

        for t in range(n_active):
            pixel_idx = sorted_indices[t]
            spikes[t, pixel_idx] = 1.0

        return spikes

import torch


class PoissonEncoder:
    """Generate Poisson spike trains from pixel intensities.

    Each pixel intensity (in [0, 1]) is mapped to a firing rate between
    min_firing_rate and max_firing_rate Hz. At each timestep, a spike is
    emitted with probability rate * dt / 1000.
    """

    def __init__(
        self,
        time_window: int = 350,
        dt: float = 1.0,
        max_firing_rate: float = 63.75,
        min_firing_rate: float = 0.0,
        device: torch.device = None,
    ):
        self.time_window = time_window
        self.dt = dt
        self.max_firing_rate = max_firing_rate
        self.min_firing_rate = min_firing_rate
        self.n_timesteps = int(time_window / dt)
        self.device = device or torch.device("cpu")

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """Convert a flattened image to a Poisson spike train.

        Args:
            image: Pixel intensities [n_pixels], values in [0, 1].

        Returns:
            Spike train [n_timesteps, n_pixels], binary (0/1).
        """
        image = image.to(self.device)

        # Map intensity to firing rate (Hz)
        rates = self.min_firing_rate + image * (
            self.max_firing_rate - self.min_firing_rate
        )

        # Probability of spiking in each dt-ms bin
        spike_prob = rates * self.dt / 1000.0

        # Generate spikes via Bernoulli sampling
        spike_prob = spike_prob.unsqueeze(0).expand(self.n_timesteps, *spike_prob.shape)
        spikes = torch.bernoulli(spike_prob)

        return spikes

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_spike_raster(
    spike_trains: torch.Tensor,
    dt: float = 1.0,
    neuron_indices: list = None,
    save_path: str = None,
    show: bool = False,
    title: str = "Spike Raster Plot",
):
    """Plot spike raster for a set of neurons.

    Args:
        spike_trains: Spike data [n_timesteps, n_neurons].
        dt: Timestep in ms.
        neuron_indices: Which neurons to plot (None = all, up to 100).
        save_path: If provided, save figure.
        show: If True, display.
        title: Figure title.
    """
    spikes_np = spike_trains.detach().cpu().numpy()
    n_timesteps, n_neurons = spikes_np.shape

    if neuron_indices is None:
        neuron_indices = list(range(min(n_neurons, 100)))

    fig, ax = plt.subplots(figsize=(12, 6))

    for plot_idx, neuron_idx in enumerate(neuron_indices):
        spike_times = np.where(spikes_np[:, neuron_idx] > 0)[0] * dt
        ax.scatter(
            spike_times,
            np.full_like(spike_times, plot_idx),
            s=1,
            c="black",
            marker="|",
        )

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Neuron index")
    ax.set_title(title)
    ax.set_ylim(-0.5, len(neuron_indices) - 0.5)
    ax.set_xlim(0, n_timesteps * dt)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_spike_counts(
    spike_counts: torch.Tensor,
    save_path: str = None,
    show: bool = False,
    title: str = "Spike Count Distribution",
):
    """Plot histogram of spike counts across neurons.

    Args:
        spike_counts: Total spike count per neuron [n_neurons].
    """
    counts = spike_counts.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(counts)), counts)
    ax.set_xlabel("Neuron index")
    ax.set_ylabel("Spike count")
    ax.set_title(title)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

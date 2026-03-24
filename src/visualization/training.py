import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_learning_curves(
    accuracies: list,
    save_path: str = None,
    show: bool = False,
    title: str = "Learning Curve",
):
    """Plot accuracy over epochs.

    Args:
        accuracies: List of accuracy values per epoch.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = list(range(1, len(accuracies) + 1))
    ax.plot(epochs, [a * 100 for a in accuracies], "b-o", markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_spike_activity(
    spike_counts_per_epoch: list,
    save_path: str = None,
    show: bool = False,
    title: str = "Average Spike Activity Over Training",
):
    """Plot average spike count per sample over training.

    Args:
        spike_counts_per_epoch: List of lists (per epoch, per sample counts).
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    means = [np.mean(counts) for counts in spike_counts_per_epoch]
    stds = [np.std(counts) for counts in spike_counts_per_epoch]
    epochs = list(range(1, len(means) + 1))

    ax.errorbar(epochs, means, yerr=stds, fmt="g-o", markersize=4, capsize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Avg Spikes / Sample")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

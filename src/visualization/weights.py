import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_weight_maps(
    weights: torch.Tensor,
    img_height: int = 28,
    img_width: int = 28,
    n_cols: int = 20,
    save_path: str = None,
    show: bool = False,
    title: str = "Learned Weight Maps",
):
    """Visualize weight vectors as receptive field images.

    Each column of the weight matrix [n_input, n_post] represents one
    neuron's receptive field, reshaped to (img_height, img_width).

    Args:
        weights: Weight matrix [n_input, n_post].
        img_height, img_width: Image dimensions for reshaping.
        n_cols: Number of columns in the grid.
        save_path: If provided, save figure to this path.
        show: If True, display the figure.
        title: Figure title.
    """
    weights_np = weights.detach().cpu().numpy()
    n_neurons = weights_np.shape[1]
    n_rows = int(np.ceil(n_neurons / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 0.8, n_rows * 0.8))
    fig.suptitle(title, fontsize=12)

    for idx in range(n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col] if n_rows > 1 else axes[col]

        if idx < n_neurons:
            rf = weights_np[:, idx].reshape(img_height, img_width)
            ax.imshow(rf, cmap="hot", interpolation="nearest", vmin=0, vmax=weights_np.max())
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

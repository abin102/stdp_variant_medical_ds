"""Difference-of-Gaussians (DoG) preprocessing for spike-based vision.

Produces ON-center and OFF-center channels from grayscale images,
mimicking retinal ganglion cell processing. This converts raw pixel
intensities into edge-like features that are far more suitable for
STDP-based learning.

Reference: Kheradpisheh et al. (2018) — STDP-based spiking deep
convolutional neural network for object recognition.
"""

import math
import torch
import torch.nn.functional as F


class DoGFilter:
    """Difference-of-Gaussians filter producing ON/OFF edge channels.

    Given a grayscale image (1, H, W), computes:
        DoG = G(sigma1) * image - G(sigma2) * image
        ON  = ReLU(DoG)          (bright-on-dark edges)
        OFF = ReLU(-DoG)         (dark-on-bright edges)

    Output: (2, H, W) normalized to [0, 1].
    """

    def __init__(
        self,
        sigma1: float = 1.0,
        sigma2: float = 2.0,
        kernel_size: int = 7,
        device: torch.device = None,
    ):
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.kernel_size = kernel_size
        self.device = device or torch.device("cpu")

        # Precompute Gaussian kernels as conv2d weights
        self.kernel1 = self._gaussian_kernel(sigma1, kernel_size).to(self.device)
        self.kernel2 = self._gaussian_kernel(sigma2, kernel_size).to(self.device)

    @staticmethod
    def _gaussian_kernel(sigma: float, size: int) -> torch.Tensor:
        """Create a 2D Gaussian kernel for use with F.conv2d.

        Returns:
            Tensor of shape (1, 1, size, size).
        """
        coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
        g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
        kernel_2d = g.unsqueeze(1) * g.unsqueeze(0)
        kernel_2d = kernel_2d / kernel_2d.sum()
        return kernel_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, size, size)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Apply DoG filtering.

        Args:
            image: Grayscale image tensor of shape (1, H, W) or (H, W).
                   Values in [0, 1].

        Returns:
            Tensor of shape (2, H, W) with ON and OFF channels, normalized to [0, 1].
        """
        if image.dim() == 2:
            image = image.unsqueeze(0)

        image = image.to(self.device)
        pad = self.kernel_size // 2

        # (1, 1, H, W) for conv2d
        x = image.unsqueeze(0)

        blurred1 = F.conv2d(x, self.kernel1, padding=pad)
        blurred2 = F.conv2d(x, self.kernel2, padding=pad)

        dog = (blurred1 - blurred2).squeeze(0)  # (1, H, W)

        on_channel = F.relu(dog)
        off_channel = F.relu(-dog)

        # Concatenate: (2, H, W)
        result = torch.cat([on_channel, off_channel], dim=0)

        # Normalize to [0, 1]
        rmax = result.max()
        if rmax > 0:
            result = result / rmax

        return result

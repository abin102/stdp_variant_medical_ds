import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


class DatasetLoader:
    """Unified loader for image classification datasets.

    Loads MNIST, FashionMNIST, or CIFAR-10, normalizes pixel values to [0, 1],
    and flattens images for input to the spiking network.
    """

    DATASETS = {
        "mnist": datasets.MNIST,
        "fashionmnist": datasets.FashionMNIST,
        "cifar10": datasets.CIFAR10,
    }

    def __init__(self, config: dict, data_root: str = "./data", flatten: bool = True):
        self.name = config["dataset"]["name"].lower()
        self.train_samples = config["dataset"].get("train_samples", -1)
        self.test_samples = config["dataset"].get("test_samples", -1)
        self.data_root = data_root
        self.flatten = flatten

        if self.name not in self.DATASETS:
            raise ValueError(
                f"Unknown dataset: {self.name}. "
                f"Available: {list(self.DATASETS.keys())}"
            )

        transform_list = [transforms.ToTensor()]
        if self.name == "cifar10":
            transform_list.insert(0, transforms.Grayscale())
        self.transform = transforms.Compose(transform_list)

    def load(self):
        """Load train and test datasets.

        Returns:
            train_data: list of (image_tensor, label) — image is flattened [n_pixels]
            test_data: list of (image_tensor, label)
        """
        dataset_cls = self.DATASETS[self.name]

        train_dataset = dataset_cls(
            root=self.data_root, train=True, download=True, transform=self.transform
        )
        test_dataset = dataset_cls(
            root=self.data_root, train=False, download=True, transform=self.transform
        )

        train_data = self._extract(train_dataset, self.train_samples)
        test_data = self._extract(test_dataset, self.test_samples)

        return train_data, test_data

    def _extract(self, dataset, max_samples: int):
        """Extract and optionally flatten images from dataset."""
        n = len(dataset) if max_samples == -1 else min(max_samples, len(dataset))
        data = []
        for i in range(n):
            img, label = dataset[i]
            # img is [C, H, W] tensor with values in [0, 1]
            if self.flatten:
                flat = img.view(-1)  # flatten to [n_pixels]
                data.append((flat, label))
            else:
                data.append((img, label))
        return data

    def get_input_size(self) -> int:
        """Return the flattened input dimension."""
        if self.name == "cifar10":
            return 32 * 32  # grayscale
        return 28 * 28

    def get_input_shape(self) -> tuple:
        """Return the spatial input shape [C, H, W]."""
        if self.name == "cifar10":
            return (1, 32, 32)
        return (1, 28, 28)

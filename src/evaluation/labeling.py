import torch


class NeuronLabeler:
    """Assign class labels to neurons based on their spiking responses.

    For each neuron, accumulates spike counts per class across all training
    samples, then assigns the class that elicited the most spikes.
    """

    def __init__(self, n_neurons: int, n_classes: int = 10):
        self.n_neurons = n_neurons
        self.n_classes = n_classes

        # Accumulate spike counts: [n_neurons, n_classes]
        self.spike_counts = torch.zeros(n_neurons, n_classes)
        self.labels = torch.zeros(n_neurons, dtype=torch.long)

    def record(self, spike_count: torch.Tensor, label: int):
        """Record spike counts for one sample.

        Args:
            spike_count: Spike count per neuron [n_neurons].
            label: Class label for this sample.
        """
        self.spike_counts[:, label] += spike_count.cpu()

    def assign_labels(self):
        """Assign each neuron to the class that maximally activates it."""
        self.labels = self.spike_counts.argmax(dim=1)

    def predict(self, spike_count: torch.Tensor) -> int:
        """Predict class from spike count using majority vote.

        Args:
            spike_count: Spike count per neuron [n_neurons].

        Returns:
            Predicted class label.
        """
        votes = torch.zeros(self.n_classes)
        for c in range(self.n_classes):
            mask = self.labels == c
            votes[c] = spike_count.cpu()[mask].sum()
        return votes.argmax().item()

    def get_label_distribution(self) -> dict:
        """Return count of neurons assigned to each class."""
        dist = {}
        for c in range(self.n_classes):
            dist[c] = (self.labels == c).sum().item()
        return dist

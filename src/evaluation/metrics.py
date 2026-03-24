import torch
import numpy as np


def compute_accuracy(predictions: list, labels: list) -> float:
    """Compute classification accuracy.

    Args:
        predictions: List of predicted class labels.
        labels: List of true class labels.

    Returns:
        Accuracy as a float in [0, 1].
    """
    correct = sum(p == l for p, l in zip(predictions, labels))
    return correct / len(labels) if labels else 0.0


def compute_confusion_matrix(
    predictions: list, labels: list, n_classes: int = 10
) -> np.ndarray:
    """Compute confusion matrix.

    Args:
        predictions: List of predicted class labels.
        labels: List of true class labels.
        n_classes: Number of classes.

    Returns:
        Confusion matrix [n_classes, n_classes] where [i, j] = count of
        true class i predicted as class j.
    """
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(labels, predictions):
        cm[true][pred] += 1
    return cm


def per_class_accuracy(
    predictions: list, labels: list, n_classes: int = 10
) -> dict:
    """Compute per-class accuracy.

    Returns:
        Dictionary mapping class -> accuracy.
    """
    cm = compute_confusion_matrix(predictions, labels, n_classes)
    result = {}
    for c in range(n_classes):
        total = cm[c].sum()
        result[c] = cm[c][c] / total if total > 0 else 0.0
    return result

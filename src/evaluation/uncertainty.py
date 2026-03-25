import torch
import numpy as np
import torch.nn.functional as F

class UncertaintyEvaluator:
    """Evaluates Bayesian approximation metrics across MC sample batches."""
    
    @staticmethod
    def expected_calibration_error(confidences, predictions, true_labels, n_bins=10):
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin.item() > 0:
                accuracy_in_bin = (predictions[in_bin] == true_labels[in_bin]).float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
        return ece.item()
        
    @staticmethod
    def compute_metrics(probs_mean, probs_variance, true_labels, n_classes=10):
        """
        Calculate metrics over ensemble outputs mapping to true target shapes.
        
        Args:
            probs_mean: [batch_size, n_classes] -> Average outputs of N MC Dropout Runs
            probs_variance: [batch_size, n_classes] -> Variance in prediction outputs.
            true_labels: [batch_size]
        """
        predictions = probs_mean.argmax(dim=-1)
        confidences = probs_mean.max(dim=-1).values
        
        acc = (predictions == true_labels).float().mean().item()
        
        # Predictive Entropy: -sum(p * log(p))
        entropy = -torch.sum(probs_mean * torch.log(probs_mean + 1e-8), dim=-1).mean().item()
        
        # Brier Score: MSE between predicted probabilities and one-hot labels
        one_hot = F.one_hot(true_labels, n_classes).float()
        brier = torch.mean(torch.sum((probs_mean - one_hot)**2, dim=-1)).item()
        
        # Negative Log Likelihood (NLL)
        nll_tensor = -torch.log(probs_mean.gather(1, true_labels.unsqueeze(-1)) + 1e-8)
        nll = torch.mean(nll_tensor).item()
        
        ece = UncertaintyEvaluator.expected_calibration_error(confidences, predictions, true_labels)
        
        return {
            "accuracy": acc,
            "entropy": entropy,
            "brier_score": brier,
            "nll": nll,
            "ece": ece
        }

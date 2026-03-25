import torch
import torch.nn as nn
from src.neurons.surrogate import ATanSurrogate

class SNNReadout(nn.Module):
    """
    Supervised Readout network that maps high-dimensional STDP spatio-temporal spikes
    into class probabilities. Contains MC Dropout to evaluate Neural Network uncertainty.
    """
    def __init__(self, in_features, num_classes, dropout_p=0.5, alpha=2.0, tau_m=10.0, dt=1.0, v_thresh=1.0):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)
        self.dropout = nn.Dropout(p=dropout_p)
        
        self.alpha = alpha
        self.tau_m = tau_m
        self.dt = dt
        self.v_thresh = v_thresh
        self.beta = 1.0 - dt / tau_m
        
        # We manually register the backward pass function.
        self.surrogate = ATanSurrogate.apply

    def forward(self, spike_seq):
        """
        Calculates readout activities iteratively applying dropout and LIF dynamics.
        
        Args:
            spike_seq: Tensor of spatial spikes from STDP feature extraction.
                       Shape: [sequence_length, batch_size, in_features]
                       
        Returns:
            out_spikes: Tensor sequence of classification votes 
                        Shape: [sequence_length, batch_size, num_classes]
        """
        seq_len, batch_size, _ = spike_seq.shape
        
        # Hidden State V initialization
        v = torch.zeros(batch_size, self.linear.out_features, device=spike_seq.device)
        out_spikes = []
        
        for t in range(seq_len):
            x = spike_seq[t]
            
            # Apply Dropout in MC fashion over time (different dropout masks vs consistent mask)
            # Standard dropout calculates unique masks per timestep in default PyTorch.
            x = self.dropout(x)
            
            current = self.linear(x)
            
            # Standard Discrete LIF Equation w/ Soft reset
            v = v * self.beta + current
            spike = self.surrogate(v - self.v_thresh, self.alpha)
            v = v - spike * self.v_thresh
            
            out_spikes.append(spike)
            
        return torch.stack(out_spikes)

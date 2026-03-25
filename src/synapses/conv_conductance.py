import torch
import torch.nn.functional as F
from .base import BaseSynapse

class ConvConductanceSynapse(BaseSynapse):
    def __init__(self, shape, stride=1, padding=0, tau_excitatory=1.0, tau_inhibitory=2.0, dt=1.0, device=None):
        super().__init__(dt, device)
        self.shape = shape
        self.stride = stride
        self.padding = padding
        self.tau_excitatory = tau_excitatory
        self.tau_inhibitory = tau_inhibitory
        
        self.decay_e = 1.0 - dt / tau_excitatory
        self.decay_i = 1.0 - dt / tau_inhibitory
        
        self.g_e = None
        self.g_i = None
        self.reset()
        
    def reset(self):
        self.g_e = torch.zeros(self.shape, device=self.device)
        self.g_i = torch.zeros(self.shape, device=self.device)
        
    def compute_current(self, pre_spikes: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        self.g_e = self.g_e * self.decay_e
        # pre_spikes: [C_in, H, W]
        # weights: [C_out, C_in, kH, kW]
        out = F.conv2d(pre_spikes.unsqueeze(0), weights, stride=self.stride, padding=self.padding)
        self.g_e = self.g_e + out.squeeze(0)
        return self.g_e
        
    def compute_inhibitory_current(self, inhibitory_input: torch.Tensor) -> torch.Tensor:
        self.g_i = self.g_i * self.decay_i
        self.g_i = self.g_i + inhibitory_input
        return -self.g_i

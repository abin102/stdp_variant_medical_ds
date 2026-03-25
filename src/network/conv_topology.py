import torch

class ConvNetworkTopology:
    def __init__(self, config: dict, device: torch.device = None):
        self.device = device or torch.device("cpu")
        net_cfg = config["network"]
        syn_cfg = config["synapse"]
        
        self.in_channels = net_cfg["in_channels"]
        self.out_channels = net_cfg["out_channels"]
        self.kernel_size = net_cfg["kernel_size"]
        self.stride = net_cfg.get("stride", 1)
        self.padding = net_cfg.get("padding", 0)
        self.inhibition_strength = net_cfg.get("inhibition_strength", 17.0)
        
        self.w_min = syn_cfg.get("w_min", 0.0)
        self.w_max = syn_cfg.get("w_max", 1.0)
        
        self.weights = self._init_weights(syn_cfg.get("w_init", "random_uniform"))
        
    def _init_weights(self, method: str) -> torch.Tensor:
        shape = (self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        if method == "random_uniform":
            w = torch.rand(shape, device=self.device)
            w = w * (self.w_max - self.w_min) + self.w_min
        elif method == "random_normal":
            w = torch.randn(shape, device=self.device)
            w = w * 0.1 + 0.5
            w = w.clamp(self.w_min, self.w_max)
        else:
            raise ValueError(f"Unknown weight init method: {method}")
        
        target_sum = self.in_channels * self.kernel_size * self.kernel_size * 0.5
        return self._normalize(w, target_sum)

    def _normalize(self, w: torch.Tensor, target_sum: float) -> torch.Tensor:
        # Normalize over in_channels, kH, kW for each out_channel
        sums = w.sum(dim=(1, 2, 3), keepdim=True).clamp(min=1e-8)
        w = w / sums * target_sum
        return w.clamp(self.w_min, self.w_max)
        
    def normalize_weights(self):
        target_sum = self.in_channels * self.kernel_size * self.kernel_size * 0.5
        self.weights = self._normalize(self.weights, target_sum)

    def get_output_shape(self, input_shape: tuple) -> tuple:
        H_in, W_in = input_shape[-2:]
        H_out = (H_in + 2*self.padding - self.kernel_size) // self.stride + 1
        W_out = (W_in + 2*self.padding - self.kernel_size) // self.stride + 1
        return (self.out_channels, H_out, W_out)

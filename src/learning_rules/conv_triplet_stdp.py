import torch
from torch.nn.grad import conv2d_weight
from .base import BaseSTDP

class ConvTripletSTDP(BaseSTDP):
    def __init__(
        self,
        in_shape,
        out_shape,
        stride=1,
        padding=0,
        tau_plus=16.8,
        tau_minus=33.7,
        tau_x=101.0,
        tau_y=125.0,
        A2_plus=0.0046,
        A2_minus=0.003,
        A3_plus=0.0091,
        A3_minus=0.0,
        dt=1.0,
        interaction="all_to_all",
        w_min=0.0,
        w_max=1.0,
        weight_dependence="none",
        mu_plus=0.0,
        mu_minus=0.0,
        device=None,
    ):
        n_pre = in_shape[0] * in_shape[1] * in_shape[2]
        n_post = out_shape[0] * out_shape[1] * out_shape[2]
        super().__init__(n_pre, n_post, w_min, w_max, weight_dependence, mu_plus, mu_minus, device)
        
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.stride = stride
        self.padding = padding
        
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.tau_x = tau_x
        self.tau_y = tau_y
        self.A2_plus = A2_plus
        self.A2_minus = A2_minus
        self.A3_plus = A3_plus
        self.A3_minus = A3_minus
        self.dt = dt
        self.interaction = interaction

        self.decay_pre_fast = 1.0 - dt / tau_plus
        self.decay_post_fast = 1.0 - dt / tau_minus
        self.decay_pre_slow = 1.0 - dt / tau_x
        self.decay_post_slow = 1.0 - dt / tau_y

        self.trace_pre_fast = None
        self.trace_pre_slow = None
        self.trace_post_fast = None
        self.trace_post_slow = None

        self._x2_before = torch.zeros(in_shape, device=self.device)
        self._y2_before = torch.zeros(out_shape, device=self.device)

        self.reset_traces()

    def reset_traces(self):
        self.trace_pre_fast = torch.zeros(self.in_shape, device=self.device)
        self.trace_pre_slow = torch.zeros(self.in_shape, device=self.device)
        self.trace_post_fast = torch.zeros(self.out_shape, device=self.device)
        self.trace_post_slow = torch.zeros(self.out_shape, device=self.device)

    def update(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        # Decay traces
        self.trace_pre_fast.mul_(self.decay_pre_fast)
        self.trace_post_fast.mul_(self.decay_post_fast)
        self.trace_pre_slow.mul_(self.decay_pre_slow)
        self.trace_post_slow.mul_(self.decay_post_slow)

        self._x2_before.copy_(self.trace_pre_slow)
        self._y2_before.copy_(self.trace_post_slow)

        if self.interaction == "nearest_spike":
            self.trace_pre_fast = torch.where(pre_spikes.bool(), torch.ones_like(self.trace_pre_fast), self.trace_pre_fast)
            self.trace_post_fast = torch.where(post_spikes.bool(), torch.ones_like(self.trace_post_fast), self.trace_post_fast)
            self.trace_pre_slow = torch.where(pre_spikes.bool(), torch.ones_like(self.trace_pre_slow), self.trace_pre_slow)
            self.trace_post_slow = torch.where(post_spikes.bool(), torch.ones_like(self.trace_post_slow), self.trace_post_slow)
        else:
            self.trace_pre_fast.add_(pre_spikes)
            self.trace_post_fast.add_(post_spikes)
            self.trace_pre_slow.add_(pre_spikes)
            self.trace_post_slow.add_(post_spikes)

        ltp_amplitude = self.A2_plus + self.A3_plus * self._x2_before
        ltd_amplitude = self.A2_minus + self.A3_minus * self._y2_before

        ltp_trace = ltp_amplitude * self.trace_pre_fast
        ltd_trace = ltd_amplitude * self.trace_post_fast

        # Compute conv2d weight gradients for DW
        # conv2d_weight signature: input, weight_size, grad_output, stride, padding, dilation, groups
        dw_plus = conv2d_weight(
            ltp_trace.unsqueeze(0),
            weights.shape,
            post_spikes.unsqueeze(0),
            stride=self.stride,
            padding=self.padding,
            dilation=1,
            groups=1
        )

        dw_minus = conv2d_weight(
            pre_spikes.unsqueeze(0),
            weights.shape,
            ltd_trace.unsqueeze(0),
            stride=self.stride,
            padding=self.padding,
            dilation=1,
            groups=1
        )

        # Apply weight dependence
        dw_plus = self.apply_weight_dependence(dw_plus, weights, is_ltp=True)
        dw_minus = self.apply_weight_dependence(dw_minus, weights, is_ltp=False)

        dw = dw_plus - dw_minus
        diagnostics = {
            "dw_plus": dw_plus.detach(),
            "dw_minus": dw_minus.detach(),
        }
        return dw, diagnostics

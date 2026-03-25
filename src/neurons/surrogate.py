import torch

class ATanSurrogate(torch.autograd.Function):
    """
    ArcTan surrogate gradient for standard LIF neurons.
    Forward pass acts as a strict step function, while backward
    pass acts as the derivative of an ArcTan smoothed variant.
    """
    @staticmethod
    def forward(ctx, input, alpha):
        ctx.save_for_backward(input)
        ctx.alpha = alpha
        return (input >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        alpha = ctx.alpha
        grad_input = grad_output / (1 + (alpha * torch.pi / 2 * input).pow(2))
        return grad_input, None

class FastSigmoidSurrogate(torch.autograd.Function):
    """
    Fast Sigmoid surrogate gradient.
    """
    @staticmethod
    def forward(ctx, input, alpha):
        ctx.save_for_backward(input)
        ctx.alpha = alpha
        return (input >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        alpha = ctx.alpha
        grad_input = grad_output / (alpha * torch.abs(input) + 1.0)**2
        return grad_input, None

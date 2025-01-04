import torch


class Clamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=0, max=1)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()

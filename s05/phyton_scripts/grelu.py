"""
Implementation of Guided ReLU activation function in PyTorch.
"""

import torch
import torch.nn as nn

class GuidedReLUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.relu(x)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        # standard ReLU mask on forward
        positive_mask = (x > 0).float()
        grad_output_positive = (grad_output > 0).float()
        # guided backprop: both forward>0 and grad>0
        grad_input = grad_output * positive_mask * grad_output_positive
        return grad_input

class GuidedReLU(nn.Module):
    def forward(self, x):
        return GuidedReLUFn.apply(x)

def replace_relu_with_guided(module):
    """
    Recursively replaces all nn.ReLU layers with GuidedReLU.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, name, GuidedReLU())
        else:
            replace_relu_with_guided(child)
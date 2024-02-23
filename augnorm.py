"""
AugNorm layer for PyTorch version. This supports both BatchNorm2D and LayerNorm. When using BatchAugNorm, just specify layer='batch'
when initializing AugNorm layer.

Note: there's some slight code modification when trying to combine the `dim` argument. When copying code to outside of this file, please do so with caution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-12

def calc(y, X, phi, eps=EPS):
    "Returns deviation [x_ij - y_j]_ij and [|x_ij - y_j|^(phi-2)]"
    dev = torch.sub(y, X)  # why y - x and not x - y here?
    dev = torch.where(dev < 0, dev - eps, dev + eps)
    abs_dev = torch.abs(dev)
    abs_dev_phi_2 = torch.pow(abs_dev, phi - 2)
    return dev, abs_dev_phi_2

def sum2d(T, dim):
    return torch.sum(T, dim=dim, keepdim=True)


def generalized_median_newtons(X, phi, dim, iter=4):
    """
    Calculates generalized median via Newton-Raphson method.
        `X`: 4D batch tensor -> dims [batches, channels, height, width] for batch norm,
          or 3D batch tensor -> dims [batches, token_length, num_features] for layer norm
        `phi`: exponent of generalized median. `phi > 1`.
        `iter`: number of iterations of Newton's
    """
    # initial [y_j]_j
    y = torch.mean(X, dim=dim, keepdim=True)

    # minima computed via Newton-Raphson
    for i in range(iter):
        # [x_ij - y_j]_ij and [|x_ij - y_j|^(phi-2)]
        dev, abs_dev_phi_2 = calc(y, X, phi)

        # f'(x_n) sum divided by phi
        F_x = sum2d(abs_dev_phi_2 * dev, dim)

        # f''(x_n) sum divided by phi
        F_xx = (phi - 1) * sum2d(abs_dev_phi_2, dim)

        # step
        y -= F_x / F_xx

    return y


# f(X, Φ): R^n, R -> R
class AugNormFunction(torch.autograd.Function):
    """Calculates the generalized median using newton's method."""


    @staticmethod
    def forward(ctx, X, phi, dim):
        # F_xy \in \mathbb{R}^{N \times M}   N=datapoints, M=number of nodes in batch
        # F_Φy \in \mathbb{R}^M     M=number of nodes in batch
        # F_yy \in \mathbb{R}^M     M=number of nodes in batch
        # To do this we stuff Φ into x to create a \mathbb{R}^{N+1 \times M} tensor
        # X = X.type(torch.Double)
        # print("type:", X.dtype)
        y = generalized_median_newtons(X, phi, dim)
        ctx.save_for_backward(X, y)
        ctx.phi = phi
        ctx.dim = dim
        return y

    @staticmethod
    def backward(ctx, grad_output):
        # backpropagation is computed via differentiating through argmin
        # https://arxiv.org/pdf/1607.05447.pdf
        # dMdx : \mathbb{R}^M \to \mathbb{R}^{N \times M}
        # represented as F_{xy}/F_{yy}
        # dMdΦ : \mathbb{R}^M \to \mathbb{R}^{M}
        # represented as F_{Φy}/F_{yy}
        X, y = ctx.saved_tensors

        # X = X.type(torch.Double)
        # y = y.type(torch.Double)

        # [|x_ij-y_j|^(phi-2)]_ij
        _, F_xy = calc(y, X, ctx.phi)

        # [SUM_i |x_ij-y_j|^(phi-2)]_j
        F_yy = sum2d(F_xy, ctx.dim)

        # chain rule
        return torch.div(F_xy, F_yy) * grad_output, None, None

class AugNorm(nn.Module):
    
    def __init__(self, phi, type='layer', shape=(768,)):
        super(AugNorm, self).__init__()

        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        
        self.type = type

        if type == 'batch':
            self.weight = nn.Parameter(torch.ones((1, shape, 1, 1)))
            self.bias = nn.Parameter(torch.zeros((1, shape, 1, 1)))
            self.running_mean = torch.zeros((1, shape, 1, 1))
            self.running_var = torch.ones((1, shape, 1, 1))
            self.dim = (0, 2, 3)
        else:
            self.weight = nn.Parameter(torch.ones(shape))
            self.bias = nn.Parameter(torch.zeros(shape))
            self.dim = (2)

        print(self.weight.shape)

        # Exponent: constant choice for phi
        self.phi = phi

        self.eps = 1e-16
        self.momentum = 0.1

    def forward(self, X):

        if self.type == 'batch' and self.running_mean.device != X.device:
            self.running_mean = self.running_mean.to(X.device)
            self.running_var = self.running_var.to(X.device)
        
        mean = AugNormFunction.apply(X, self.phi, self.dim)
        
        var = (torch.pow(X - mean, 2)).mean(dim=self.dim, keepdim=True)
        
        X_hat = (X - mean) / torch.sqrt(var + self.eps)

        if self.type == 'batch':
            with torch.no_grad():
                self.running_mean = (1.0 - self.momentum) * self.running_mean + (self.momentum) * mean
                self.running_var = (1.0 - self.momentum) * self.running_var + (self.momentum) * var
            
            # self.running_mean = self.running_mean.flatten()
            # self.running_var = self.running_var.flatten()

            if not torch.is_grad_enabled():
                X_hat = (X - self.running_mean[None, :, None, None]) / torch.sqrt(self.running_var[None, :, None, None] + self.eps)

        Y = self.weight[None, :, None, None] * X_hat + self.bias[None, :, None, None]  # Scale and shift
        
        return Y
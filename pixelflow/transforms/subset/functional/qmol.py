import torch
import torch.nn as nn
import torch.nn.functional as F
from pixelflow.transforms.functional._utils_bisection_inverse import bisection_inverse
from pixelflow.transforms.functional._utils_cmol import cmol_cdf


def qmol_transform_forward(x_lower, x_upper, logit_weights, means, log_scales, K=256):
    '''
    Quantized univariate mixture of logistics forward transform.

    Args:
        x_lower: torch.Tensor, shape (shape,)
        x_upper: torch.Tensor, shape (shape,)
        logit_weights: torch.Tensor, shape (shape, num_mixtures)
        means: torch.Tensor, shape (shape, num_mixtures)
        log_scales: torch.Tensor, shape (shape, num_mixtures)
        K: int, the number of bins
    '''

    log_weights = F.log_softmax(logit_weights, dim=-1)
    log_scales = log_scales.clamp(min=-7.0)

    def mix_cdf(x):
        return torch.sum(log_weights.exp() * cmol_cdf(x, means, log_scales, K), dim=-1)

    z_lower = mix_cdf(x_lower)
    z_upper = mix_cdf(x_upper)

    return z_lower, z_upper


def qmol_transform_inverse(z, logit_weights, means, log_scales, K=256, eps=1e-10, max_iters=100):
    '''
    Quantized univariate mixture of logistics inverse transform.

    Args:
        z: torch.Tensor, shape (shape,)
        logit_weights: torch.Tensor, shape (shape, num_mixtures)
        means: torch.Tensor, shape (shape, num_mixtures)
        log_scales: torch.Tensor, shape (shape, num_mixtures)
        K: int, the number of bins
        eps: float, tolerance for bisection |f(x) - z_est| < eps
        max_iters: int, maximum iterations for bisection
    '''

    log_weights = F.log_softmax(logit_weights, dim=-1)
    log_scales = log_scales.clamp(min=-7.0)

    def mix_cdf(x):
        return torch.sum(log_weights.exp() * cmol_cdf(x, means, log_scales, K), dim=-1)

    return bisection_inverse(fn=lambda x: mix_cdf(x),
                             z=z,
                             init_x=torch.ones_like(z) * 0.5,
                             init_lower=torch.zeros_like(z),
                             init_upper=torch.ones_like(z),
                             eps=eps,
                             max_iters=max_iters)

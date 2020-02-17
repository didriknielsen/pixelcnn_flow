import math
import torch
from torch import nn
import torch.nn.functional as F
from pixelflow.transforms.functional._utils_bisection_inverse import bisection_inverse
from pixelflow.transforms.functional._utils_cmol import cmol_cdf, cmol_log_pdf
from pixelflow.transforms.functional._utils_multivariate import adjust_log_weights, adjust_means


def multivariate_qmol_transform_elementwise_params(x_lower, x_upper, logit_weights, means, log_scales, unnormalized_corr, K=256, mean_lambd=lambda x: 2*x-1):
    '''
    Quantized multivariate mixture of logistics forward transform.

    Args:
        x_lower: torch.Tensor, shape (shape,)
        x_upper: torch.Tensor, shape (shape,)
        logit_weights: torch.Tensor, shape (shape, num_mixtures)
        means: torch.Tensor, shape (shape, num_mixtures)
        log_scales: torch.Tensor, shape (shape, num_mixtures)
        unnormalized_corr: torch.Tensor, shape (shape, num_mixtures*c*(c-1)//2)
        K: int, the number of bins
    '''

    log_weights = F.log_softmax(logit_weights, dim=-1)
    log_scales = log_scales.clamp(min=-7.0)
    corr = torch.tanh(unnormalized_corr)

    adjusted_means = adjust_means(means, corr, x_lower, in_lambd=mean_lambd) # Correlate means for each component
    component_cdf_lower = cmol_cdf(x_lower, adjusted_means, log_scales, K)
    component_cdf_upper = cmol_cdf(x_upper, adjusted_means, log_scales, K)
    component_log_probs = torch.log((component_cdf_upper - component_cdf_lower).clamp(1e-12))
    adjusted_log_weights = adjust_log_weights(log_weights, component_log_probs) # Adjust log_weights for autoregressive transformation
    return adjusted_log_weights, adjusted_means, log_scales, component_cdf_lower, component_cdf_upper

import math
import torch
from torch import nn
import torch.nn.functional as F
from pixelflow.transforms.functional._utils_bisection_inverse import bisection_inverse
from pixelflow.transforms.functional._utils_cmol import cmol_cdf, cmol_log_pdf
from pixelflow.transforms.functional._utils_multivariate import adjust_log_weights, adjust_means


def multivariate_cmol_transform(inputs, logit_weights, means, log_scales, unnormalized_corr, K=256, eps=1e-10, max_iters=100, mean_lambd=lambda x: 2*x-1, inverse=False):
    '''
    Censored multivariate mixture of logistics transform.

    Args:
        inputs: torch.Tensor, shape (shape,)
        logit_weights: torch.Tensor, shape (shape, num_mixtures)
        means: torch.Tensor, shape (shape, num_mixtures)
        log_scales: torch.Tensor, shape (shape, num_mixtures)
        unnormalized_corr: torch.Tensor, shape (shape, num_mixtures*c*(c-1)//2)
        K: int, the number of bins
        eps: float, tolerance for bisection |f(x) - z_est| < eps
        max_iters: int, maximum iterations for bisection
        inverse: bool, if True, return inverse
    '''

    log_weights = F.log_softmax(logit_weights, dim=-1)
    log_scales = log_scales.clamp(min=-7.0)
    corr = torch.tanh(unnormalized_corr)

    def mix_cdf(x, adjusted_log_weights, adjusted_means, log_scales):
        return torch.sum(adjusted_log_weights.exp() * cmol_cdf(x, adjusted_means, log_scales, K), dim=-1)

    def mix_log_pdf(x, adjusted_log_weights, component_log_probs):
        return torch.logsumexp(adjusted_log_weights + component_log_probs, dim=-1)

    def univariate_inverse(z, adjusted_log_weights, adjusted_means, log_scales):
        return bisection_inverse(fn=lambda x: mix_cdf(x, adjusted_log_weights, adjusted_means, log_scales),
                                 z=z,
                                 init_x=torch.ones_like(z) * 0.5,
                                 init_lower=torch.zeros_like(z),
                                 init_upper=torch.ones_like(z),
                                 eps=eps,
                                 max_iters=max_iters)

    if inverse:
        z = inputs
        x = torch.zeros_like(z)
        for c in range(x.shape[1]):
            adjusted_means = adjust_means(means, corr, x, in_lambd=mean_lambd) # Correlate means for each component
            component_log_probs = cmol_log_pdf(x[:,:c+1], adjusted_means[:,:c+1], log_scales[:,:c+1], K) # Compute log_probs per component
            adjusted_log_weights = adjust_log_weights(log_weights, component_log_probs) # Adjust log_weights for autoregressive transformation
            x[:,c:c+1] = univariate_inverse(z[:,c:c+1], adjusted_log_weights[:,c:c+1], adjusted_means[:,c:c+1], log_scales[:,c:c+1])
        return x
    else:
        x = inputs
        adjusted_means = adjust_means(means, corr, x, in_lambd=mean_lambd) # Correlate means for each component
        component_log_probs = cmol_log_pdf(x, adjusted_means, log_scales, K) # Compute log_probs per component
        adjusted_log_weights = adjust_log_weights(log_weights, component_log_probs) # Adjust log_weights for autoregressive transformation
        z = mix_cdf(x, adjusted_log_weights, adjusted_means, log_scales)
        ldj = mix_log_pdf(x, adjusted_log_weights, component_log_probs)
        return z, ldj

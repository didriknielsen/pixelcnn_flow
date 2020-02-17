from collections.abc import Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F
from pixelflow.utils import sum_except_batch
from pixelflow.transforms.functional import get_mixture_params, get_matching_multivariate_mixture_params_2d
from pixelflow.transforms.functional import splines, cmol_transform, multivariate_cmol_transform
from pixelflow.transforms.subset.functional import qmol_transform_forward, qmol_transform_inverse
from pixelflow.transforms.subset.functional import multivariate_qmol_transform_elementwise_params
from pixelflow.transforms.functional._utils_bisection_inverse import bisection_inverse
from pixelflow.transforms.functional._utils_cmol import cmol_cdf


class AutoregressiveSubsetTransform2d(nn.Module):

    def __init__(self, autoregressive_net, condition_on_cube=False):
        super(AutoregressiveSubsetTransform2d, self).__init__()
        self.autoregressive_net = autoregressive_net
        self.condition_on_cube = condition_on_cube

    def forward(self, x_lower, x_upper):
        elementwise_params = self.elementwise_params(x_lower, x_upper)
        z_lower, z_upper = self._elementwise_forward(x_lower, x_upper, elementwise_params)
        return z_lower, z_upper

    def elementwise_params(self, x_lower, x_upper):
        if self.condition_on_cube:  x = torch.cat([x_lower, x_upper], dim=1)
        else:                       x = x_lower
        return self.autoregressive_net(x)

    def elementwise_forward(self, x_lower, x_upper, elementwise_params):
        return self._elementwise_forward(x_lower, x_upper, elementwise_params)

    def elementwise_inverse(self, z, elementwise_params):
        return self._elementwise_inverse(z, elementwise_params)

    def _output_dim_multiplier(self):
        raise NotImplementedError()

    def _elementwise_forward(self, x_lower, x_upper, elementwise_params):
        raise NotImplementedError()

    def _elementwise_inverse(self, z_lower, z_upper, elementwise_params):
        raise NotImplementedError()


class LinearSplineAutoregressiveSubsetTransform2d(AutoregressiveSubsetTransform2d):

    def __init__(self, autoregressive_net, num_bins, condition_on_cube=False):
        super(LinearSplineAutoregressiveSubsetTransform2d, self).__init__(autoregressive_net=autoregressive_net, condition_on_cube=condition_on_cube)
        self.num_bins = num_bins

    def _output_dim_multiplier(self):
        return self.num_bins

    def _elementwise_forward(self, x_lower, x_upper, elementwise_params):
        assert elementwise_params.shape[-1] == self._output_dim_multiplier()
        z_lower, _ = splines.linear_spline(x_lower, elementwise_params, inverse=False)
        z_upper, _ = splines.linear_spline(x_upper, elementwise_params, inverse=False)
        return z_lower, z_upper

    def _elementwise_inverse(self, z, elementwise_params):
        assert elementwise_params.shape[-1] == self._output_dim_multiplier()
        x, _ = splines.linear_spline(z, elementwise_params, inverse=True)
        return x


class QuadraticSplineAutoregressiveSubsetTransform2d(AutoregressiveSubsetTransform2d):

    def __init__(self, autoregressive_net, num_bins, condition_on_cube=False):
        super(QuadraticSplineAutoregressiveSubsetTransform2d, self).__init__(autoregressive_net=autoregressive_net, condition_on_cube=condition_on_cube)
        self.num_bins = num_bins

    def _output_dim_multiplier(self):
        return 2 * self.num_bins + 1

    def _elementwise_forward(self, x_lower, x_upper, elementwise_params):
        assert elementwise_params.shape[-1] == self._output_dim_multiplier()
        unnormalized_widths, unnormalized_heights = elementwise_params[..., :self.num_bins], elementwise_params[..., self.num_bins:]
        z_lower, _ = splines.quadratic_spline(x_lower, unnormalized_widths=unnormalized_widths, unnormalized_heights=unnormalized_heights, inverse=False)
        z_upper, _ = splines.quadratic_spline(x_upper, unnormalized_widths=unnormalized_widths, unnormalized_heights=unnormalized_heights, inverse=False)
        return z_lower, z_upper

    def _elementwise_inverse(self, z, elementwise_params):
        assert elementwise_params.shape[-1] == self._output_dim_multiplier()
        unnormalized_widths, unnormalized_heights = elementwise_params[..., :self.num_bins], elementwise_params[..., self.num_bins:]
        x, _ = splines.quadratic_spline(z, unnormalized_widths=unnormalized_widths, unnormalized_heights=unnormalized_heights, inverse=True)
        return x


class CubicSplineAutoregressiveSubsetTransform2d(AutoregressiveSubsetTransform2d):

    def __init__(self, autoregressive_net, num_bins, condition_on_cube=False):
        super(CubicSplineAutoregressiveSubsetTransform2d, self).__init__(autoregressive_net=autoregressive_net, condition_on_cube=condition_on_cube)
        self.num_bins = num_bins

    def _output_dim_multiplier(self):
        return 2 * self.num_bins + 2

    def _elementwise_forward(self, x_lower, x_upper, elementwise_params):
        assert elementwise_params.shape[-1] == self._output_dim_multiplier()
        unnormalized_widths = elementwise_params[..., :self.num_bins]
        unnormalized_heights = elementwise_params[..., self.num_bins:2*self.num_bins]
        unnorm_derivatives_left = elementwise_params[..., 2*self.num_bins:2*self.num_bins+1]
        unnorm_derivatives_right = elementwise_params[..., 2*self.num_bins+1:]
        z_lower, _ = splines.cubic_spline(x_lower,
                                          unnormalized_widths=unnormalized_widths,
                                          unnormalized_heights=unnormalized_heights,
                                          unnorm_derivatives_left=unnorm_derivatives_left,
                                          unnorm_derivatives_right=unnorm_derivatives_right,
                                          inverse=False)
        z_upper, _ = splines.cubic_spline(x_upper,
                                          unnormalized_widths=unnormalized_widths,
                                          unnormalized_heights=unnormalized_heights,
                                          unnorm_derivatives_left=unnorm_derivatives_left,
                                          unnorm_derivatives_right=unnorm_derivatives_right,
                                          inverse=False)
        return z_lower, z_upper

    def _elementwise_inverse(self, z, elementwise_params):
        assert elementwise_params.shape[-1] == self._output_dim_multiplier()
        unnormalized_widths = elementwise_params[..., :self.num_bins]
        unnormalized_heights = elementwise_params[..., self.num_bins:2*self.num_bins]
        unnorm_derivatives_left = elementwise_params[..., 2*self.num_bins:2*self.num_bins+1]
        unnorm_derivatives_right = elementwise_params[..., 2*self.num_bins+1:]
        x, _ = splines.cubic_spline(z,
                                    unnormalized_widths=unnormalized_widths,
                                    unnormalized_heights=unnormalized_heights,
                                    unnorm_derivatives_left=unnorm_derivatives_left,
                                    unnorm_derivatives_right=unnorm_derivatives_right,
                                    inverse=True)
        return x


class RationalQuadraticSplineAutoregressiveSubsetTransform2d(AutoregressiveSubsetTransform2d):

    def __init__(self, autoregressive_net, num_bins, condition_on_cube=False):
        super(RationalQuadraticSplineAutoregressiveSubsetTransform2d, self).__init__(autoregressive_net=autoregressive_net, condition_on_cube=condition_on_cube)
        self.num_bins = num_bins

    def _output_dim_multiplier(self):
        return 3 * self.num_bins + 1

    def _elementwise_forward(self, x_lower, x_upper, elementwise_params):
        assert elementwise_params.shape[-1] == self._output_dim_multiplier()
        unnormalized_widths = elementwise_params[..., :self.num_bins]
        unnormalized_heights = elementwise_params[..., self.num_bins:2*self.num_bins]
        unnormalized_derivatives = elementwise_params[..., 2*self.num_bins:]
        z_lower, _ = splines.rational_quadratic_spline(x_lower,
                                                       unnormalized_widths=unnormalized_widths,
                                                       unnormalized_heights=unnormalized_heights,
                                                       unnormalized_derivatives=unnormalized_derivatives,
                                                       inverse=False)
        z_upper, _ = splines.rational_quadratic_spline(x_upper,
                                                       unnormalized_widths=unnormalized_widths,
                                                       unnormalized_heights=unnormalized_heights,
                                                       unnormalized_derivatives=unnormalized_derivatives,
                                                       inverse=False)
        return z_lower, z_upper

    def _elementwise_inverse(self, z, elementwise_params):
        assert elementwise_params.shape[-1] == self._output_dim_multiplier()
        unnormalized_widths = elementwise_params[..., :self.num_bins]
        unnormalized_heights = elementwise_params[..., self.num_bins:2*self.num_bins]
        unnormalized_derivatives = elementwise_params[..., 2*self.num_bins:]
        x, _ = splines.rational_quadratic_spline(z,
                                                 unnormalized_widths=unnormalized_widths,
                                                 unnormalized_heights=unnormalized_heights,
                                                 unnormalized_derivatives=unnormalized_derivatives,
                                                 inverse=True)
        return x


class MOLAutoregressiveSubsetTransform2d(AutoregressiveSubsetTransform2d):

    def __init__(self, autoregressive_net, num_mixtures, num_bins, eps=1e-10, max_iters=100, condition_on_cube=False):
        super(MOLAutoregressiveSubsetTransform2d, self).__init__(autoregressive_net=autoregressive_net, condition_on_cube=condition_on_cube)
        self.num_bins = num_bins
        self.num_mixtures = num_mixtures
        self.eps = eps
        self.max_iters = max_iters

    def _output_dim_multiplier(self):
        return 3 * self.num_mixtures

    def _elementwise_forward(self, x_lower, x_upper, elementwise_params):
        assert elementwise_params.shape[-1] == self._output_dim_multiplier()

        logit_weights, means, log_scales = get_mixture_params(elementwise_params, num_mixtures=self.num_mixtures)

        z_lower, z_upper = qmol_transform_forward(x_lower=x_lower,
                                                  x_upper=x_upper,
                                                  logit_weights=logit_weights,
                                                  means=means,
                                                  log_scales=log_scales)
        return z_lower, z_upper

    def _elementwise_inverse(self, z, elementwise_params):
        assert elementwise_params.shape[-1] == self._output_dim_multiplier()

        logit_weights, means, log_scales = get_mixture_params(elementwise_params, num_mixtures=self.num_mixtures)

        x = qmol_transform_inverse(z=z,
                                   logit_weights=logit_weights,
                                   means=means,
                                   log_scales=log_scales,
                                   eps=self.eps,
                                   max_iters=self.max_iters)

        return x


class MultivariateMOLAutoregressiveSubsetTransform2d(AutoregressiveSubsetTransform2d):

    def __init__(self, autoregressive_net, channels, num_mixtures, num_bins, eps=1e-10, max_iters=100, mean_lambd=lambda x: 2*x-1, condition_on_cube=False):
        assert channels==3
        super(MultivariateMOLAutoregressiveSubsetTransform2d, self).__init__(autoregressive_net=autoregressive_net, condition_on_cube=condition_on_cube)
        self.num_bins = num_bins
        self.num_mixtures = num_mixtures
        self.channels = channels
        self.eps = eps
        self.max_iters = max_iters
        self.mean_lambd = mean_lambd

    def _params(self):
        return self.num_mixtures * (1 + 2 * self.channels + self.channels * (self.channels - 1) // 2)

    def _mix_cdf(self, x, adjusted_means, log_scales, adjusted_log_weights):
        return torch.sum(adjusted_log_weights.exp() * cmol_cdf(x, adjusted_means, log_scales, self.num_bins), dim=-1)

    def _compute_params(self, x_lower, x_upper):
        if self.condition_on_cube:  x = torch.cat([x_lower, x_upper], dim=1)
        else:                       x = x_lower
        params = self.autoregressive_net(x)
        assert params.shape[1] == self._params()
        logit_weights, means, log_scales, unnormalized_corr = get_matching_multivariate_mixture_params_2d(params, num_mixtures=self.num_mixtures)
        return multivariate_qmol_transform_elementwise_params(x_lower, x_upper, logit_weights, means, log_scales, unnormalized_corr, K=self.num_bins, mean_lambd=self.mean_lambd)

    def forward(self, x_lower, x_upper):
        adjusted_log_weights, adjusted_means, log_scales, component_cdf_lower, component_cdf_upper = self._compute_params(x_lower, x_upper)
        z_lower = torch.sum(adjusted_log_weights.exp() * component_cdf_lower, dim=-1).clamp(min=0.0, max=1.0)
        z_upper = torch.sum(adjusted_log_weights.exp() * component_cdf_upper, dim=-1).clamp(min=0.0, max=1.0)
        return z_lower, z_upper

    def elementwise_params(self, x_lower, x_upper):
        adjusted_log_weights, adjusted_means, log_scales, _, _ = self._compute_params(x_lower, x_upper)
        elementwise_params = torch.cat([adjusted_log_weights, adjusted_means, log_scales], dim=0)
        return elementwise_params

    def elementwise_forward(self, x_lower, x_upper, elementwise_params):
        adjusted_log_weights, adjusted_means, log_scales = torch.chunk(elementwise_params, 3, dim=0)
        z_lower = self._mix_cdf(x_lower, adjusted_means, log_scales, adjusted_log_weights).clamp(min=0.0, max=1.0)
        z_upper = self._mix_cdf(x_upper, adjusted_means, log_scales, adjusted_log_weights).clamp(min=0.0, max=1.0)
        return z_lower, z_upper

    def elementwise_inverse(self, z, elementwise_params):
        adjusted_log_weights, adjusted_means, log_scales = torch.chunk(elementwise_params, 3, dim=0)
        return bisection_inverse(fn=lambda x: self._mix_cdf(x, adjusted_means, log_scales, adjusted_log_weights).clamp(min=0.0, max=1.0),
                                 z=z,
                                 init_x=torch.ones_like(z) * 0.5,
                                 init_lower=torch.zeros_like(z),
                                 init_upper=torch.ones_like(z),
                                 eps=self.eps,
                                 max_iters=self.max_iters)

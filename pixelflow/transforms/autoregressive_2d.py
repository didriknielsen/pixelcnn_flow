from collections.abc import Iterable
import torch
from pixelflow.transforms import Transform
from pixelflow.utils import sum_except_batch
from pixelflow.transforms.functional import get_mixture_params, get_matching_multivariate_mixture_params_2d
from pixelflow.transforms.functional import splines, cmol_transform, multivariate_cmol_transform


class AutoregressiveTransform2d(Transform):
    """Transforms each input variable with an invertible elementwise transformation.

    The parameters of each invertible elementwise transformation can be functions of previous input
    variables, but they must not depend on the current or any following input variables.

    NOTE: Calculating the inverse transform is D times slower than calculating the
    forward transform, where D is the dimensionality of the input to the transform.

    Args:
        autoregressive_net: nn.Module, an autoregressive network such that
            elementwise_params = autoregressive_net(x)
        autoregressive_order: str or Iterable, the order in which to sample.
            One of `{'raster_scan', 'raster_scan_spatial'}`
    """
    def __init__(self, autoregressive_net, autoregressive_order='raster_scan'):
        super(AutoregressiveTransform2d, self).__init__()
        assert isinstance(autoregressive_order, str) or isinstance(autoregressive_order, Iterable)
        assert autoregressive_order in {'raster_scan', 'raster_scan_spatial'}
        self.autoregressive_net = autoregressive_net
        self.autoregressive_order = autoregressive_order

    def forward(self, x):
        elementwise_params = self.autoregressive_net(x)
        z, ldj = self._elementwise_forward(x, elementwise_params)
        return z, ldj

    def inverse(self, z):
        with torch.no_grad():
            if self.autoregressive_order == 'raster_scan': return self._inverse_raster_scan(z)
            if self.autoregressive_order == 'raster_scan_spatial': return self._inverse_raster_scan_spatial(z)

    def _inverse_raster_scan(self, z):
        x = torch.zeros_like(z)
        for h in range(x.shape[2]):
            for w in range(x.shape[3]):
                for c in range(x.shape[1]):
                    elementwise_params = self.autoregressive_net(x)
                    x[:,c,h,w] = self._elementwise_inverse(z[:,c,h,w], elementwise_params[:,c,h,w])
        return x

    def _inverse_raster_scan_spatial(self, z):
        x = torch.zeros_like(z)
        for h in range(x.shape[2]):
            for w in range(x.shape[3]):
                elementwise_params = self.autoregressive_net(x)
                x[:,:,h,w] = self._elementwise_inverse(z[:,:,h,w], elementwise_params[:,:,h,w])
        return x

    def _output_dim_multiplier(self):
        raise NotImplementedError()

    def _elementwise_forward(self, x, elementwise_params):
        raise NotImplementedError()

    def _elementwise_inverse(self, z, elementwise_params):
        raise NotImplementedError()


class AdditiveAutoregressiveTransform2d(AutoregressiveTransform2d):

    def _output_dim_multiplier(self):
        return 1

    def _elementwise_forward(self, x, elementwise_params):
        return x + elementwise_params, torch.zeros(x.shape[0], device=x.device)

    def _elementwise_inverse(self, z, elementwise_params):
        return z - elementwise_params


class AffineAutoregressiveTransform2d(AutoregressiveTransform2d):

    def _output_dim_multiplier(self):
        return 2

    def _elementwise_forward(self, x, elementwise_params):
        assert elementwise_params.shape[-1] == self._output_dim_multiplier()
        unconstrained_scale, shift = self._unconstrained_scale_and_shift(elementwise_params)
        scale = torch.sigmoid(unconstrained_scale + 2.) + 1e-3
        z = scale * x + shift
        ldj = sum_except_batch(torch.log(scale))
        return z, ldj

    def _elementwise_inverse(self, z, elementwise_params):
        assert elementwise_params.shape[-1] == self._output_dim_multiplier()
        unconstrained_scale, shift = self._unconstrained_scale_and_shift(elementwise_params)
        scale = torch.sigmoid(unconstrained_scale + 2.) + 1e-3
        x = (z - shift) / scale
        return x

    def _unconstrained_scale_and_shift(self, elementwise_params):
        unconstrained_scale = elementwise_params[..., 0]
        shift = elementwise_params[..., 1]
        return unconstrained_scale, shift


class LinearSplineAutoregressiveTransform2d(AutoregressiveTransform2d):

    def __init__(self, autoregressive_net, num_bins, autoregressive_order='raster_scan'):
        super(LinearSplineAutoregressiveTransform2d, self).__init__(autoregressive_net=autoregressive_net, autoregressive_order=autoregressive_order)
        self.num_bins = num_bins

    def _output_dim_multiplier(self):
        return self.num_bins

    def _elementwise_forward(self, x, elementwise_params):
        assert elementwise_params.shape[-1] == self._output_dim_multiplier()
        z, ldj_elementwise = splines.linear_spline(x, elementwise_params, inverse=False)
        ldj = sum_except_batch(ldj_elementwise)
        return z, ldj

    def _elementwise_inverse(self, z, elementwise_params):
        assert elementwise_params.shape[-1] == self._output_dim_multiplier()
        x, _ = splines.linear_spline(z, elementwise_params, inverse=True)
        return x


class QuadraticSplineAutoregressiveTransform2d(AutoregressiveTransform2d):

    def __init__(self, autoregressive_net, num_bins, autoregressive_order='raster_scan'):
        super(QuadraticSplineAutoregressiveTransform2d, self).__init__(autoregressive_net=autoregressive_net, autoregressive_order=autoregressive_order)
        self.num_bins = num_bins

    def _output_dim_multiplier(self):
        return 2 * self.num_bins + 1

    def _elementwise_forward(self, x, elementwise_params):
        assert elementwise_params.shape[-1] == self._output_dim_multiplier()
        unnormalized_widths, unnormalized_heights = elementwise_params[..., :self.num_bins], elementwise_params[..., self.num_bins:]
        z, ldj_elementwise = splines.quadratic_spline(x, unnormalized_widths=unnormalized_widths, unnormalized_heights=unnormalized_heights, inverse=False)
        ldj = sum_except_batch(ldj_elementwise)
        return z, ldj

    def _elementwise_inverse(self, z, elementwise_params):
        assert elementwise_params.shape[-1] == self._output_dim_multiplier()
        unnormalized_widths, unnormalized_heights = elementwise_params[..., :self.num_bins], elementwise_params[..., self.num_bins:]
        x, _ = splines.quadratic_spline(z, unnormalized_widths=unnormalized_widths, unnormalized_heights=unnormalized_heights, inverse=True)
        return x


class CubicSplineAutoregressiveTransform2d(AutoregressiveTransform2d):

    def __init__(self, autoregressive_net, num_bins, autoregressive_order='raster_scan'):
        super(CubicSplineAutoregressiveTransform2d, self).__init__(autoregressive_net=autoregressive_net, autoregressive_order=autoregressive_order)
        self.num_bins = num_bins

    def _output_dim_multiplier(self):
        return 2 * self.num_bins + 2

    def _elementwise_forward(self, x, elementwise_params):
        assert elementwise_params.shape[-1] == self._output_dim_multiplier()
        unnormalized_widths = elementwise_params[..., :self.num_bins]
        unnormalized_heights = elementwise_params[..., self.num_bins:2*self.num_bins]
        unnorm_derivatives_left = elementwise_params[..., 2*self.num_bins:2*self.num_bins+1]
        unnorm_derivatives_right = elementwise_params[..., 2*self.num_bins+1:]
        z, ldj_elementwise = splines.cubic_spline(x,
                                                  unnormalized_widths=unnormalized_widths,
                                                  unnormalized_heights=unnormalized_heights,
                                                  unnorm_derivatives_left=unnorm_derivatives_left,
                                                  unnorm_derivatives_right=unnorm_derivatives_right,
                                                  inverse=False)
        ldj = sum_except_batch(ldj_elementwise)
        return z, ldj

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


class RationalQuadraticSplineAutoregressiveTransform2d(AutoregressiveTransform2d):

    def __init__(self, autoregressive_net, num_bins, autoregressive_order='raster_scan'):
        super(RationalQuadraticSplineAutoregressiveTransform2d, self).__init__(autoregressive_net=autoregressive_net, autoregressive_order=autoregressive_order)
        self.num_bins = num_bins

    def _output_dim_multiplier(self):
        return 3 * self.num_bins + 1

    def _elementwise_forward(self, x, elementwise_params):
        assert elementwise_params.shape[-1] == self._output_dim_multiplier()
        unnormalized_widths = elementwise_params[..., :self.num_bins]
        unnormalized_heights = elementwise_params[..., self.num_bins:2*self.num_bins]
        unnormalized_derivatives = elementwise_params[..., 2*self.num_bins:]
        z, ldj_elementwise = splines.rational_quadratic_spline(x,
                                                               unnormalized_widths=unnormalized_widths,
                                                               unnormalized_heights=unnormalized_heights,
                                                               unnormalized_derivatives=unnormalized_derivatives,
                                                               inverse=False)
        ldj = sum_except_batch(ldj_elementwise)
        return z, ldj

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


class CMOLAutoregressiveTransform2d(AutoregressiveTransform2d):

    def __init__(self, autoregressive_net, num_mixtures, num_bins, autoregressive_order='raster_scan', eps=1e-10, max_iters=100):
        super(CMOLAutoregressiveTransform2d, self).__init__(autoregressive_net=autoregressive_net, autoregressive_order=autoregressive_order)
        self.num_bins = num_bins
        self.num_mixtures = num_mixtures
        self.eps = eps
        self.max_iters = max_iters

    def _output_dim_multiplier(self):
        return 3 * self.num_mixtures

    def _elementwise(self, inputs, elementwise_params, inverse):
        assert elementwise_params.shape[-1] == self._output_dim_multiplier()

        logit_weights, means, log_scales = get_mixture_params(elementwise_params, num_mixtures=self.num_mixtures)

        x = cmol_transform(inputs=inputs,
                           logit_weights=logit_weights,
                           means=means,
                           log_scales=log_scales,
                           K=self.num_bins,
                           eps=self.eps,
                           max_iters=self.max_iters,
                           inverse=inverse)

        if inverse:
            return x
        else:
            z, ldj_elementwise = x
            ldj = sum_except_batch(ldj_elementwise)
            return z, ldj

    def _elementwise_forward(self, x, elementwise_params):
        return self._elementwise(x, elementwise_params, inverse=False)

    def _elementwise_inverse(self, z, elementwise_params):
        return self._elementwise(z, elementwise_params, inverse=True)


class MultivariateCMOLAutoregressiveTransform2d(AutoregressiveTransform2d):

    def __init__(self, autoregressive_net, channels, num_mixtures, num_bins, autoregressive_order='raster_scan_spatial', eps=1e-10, max_iters=100, mean_lambd=lambda x: 2*x-1):
        assert autoregressive_order in {'raster_scan_spatial'}
        assert channels==3, 'Only 3 channels currently supported'
        super(MultivariateCMOLAutoregressiveTransform2d, self).__init__(autoregressive_net=autoregressive_net, autoregressive_order=autoregressive_order)
        self.num_bins = num_bins
        self.num_mixtures = num_mixtures
        self.channels = channels
        self.eps = eps
        self.max_iters = max_iters
        self.mean_lambd = mean_lambd

    def _params(self):
        return self.num_mixtures * (1 + 2 * self.channels + self.channels * (self.channels - 1) // 2)

    def _elementwise(self, inputs, elementwise_params, inverse):
        assert elementwise_params.shape[1] == self._params()
        batch_size = inputs.shape[0]

        single_pixel = False
        if inverse and inputs.dim()==2:
            inputs = inputs.reshape(inputs.shape + (1,1))
            elementwise_params = elementwise_params.reshape(elementwise_params.shape + (1,1))
            single_pixel = True

        logit_weights, means, log_scales, unnormalized_corr = get_matching_multivariate_mixture_params_2d(elementwise_params,
                                                                                                          num_mixtures=self.num_mixtures)

        x = multivariate_cmol_transform(inputs=inputs,
                                        logit_weights=logit_weights,
                                        means=means,
                                        log_scales=log_scales,
                                        unnormalized_corr=unnormalized_corr,
                                        K=self.num_bins,
                                        eps=self.eps,
                                        max_iters=self.max_iters,
                                        mean_lambd=self.mean_lambd,
                                        inverse=inverse)

        if inverse:
            if single_pixel:
                x = x.squeeze(-1).squeeze(-1)
            return x
        else:
            z, ldj_elementwise = x
            ldj = sum_except_batch(ldj_elementwise)
            return z, ldj

    def _elementwise_forward(self, x, elementwise_params):
        return self._elementwise(x, elementwise_params, inverse=False)

    def _elementwise_inverse(self, z, elementwise_params):
        return self._elementwise(z, elementwise_params, inverse=True)

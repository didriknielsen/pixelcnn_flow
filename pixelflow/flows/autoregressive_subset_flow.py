import math
import torch
from torch import nn
from collections.abc import Iterable
from pixelflow.distributions import Distribution, StandardUniform
from pixelflow.transforms.subset import AutoregressiveSubsetTransform2d
from pixelflow.utils import sum_except_batch


class AutoregressiveSubsetFlow2d(Distribution):

    mode = 'log_prob'
    allowed_modes = {'log_prob', 'sample'}

    def __init__(self, base_shape, transforms, autoregressive_order='raster_scan', bits=8):
        super(AutoregressiveSubsetFlow2d, self).__init__()
        if isinstance(transforms, AutoregressiveSubsetTransform2d): transforms = [transforms]
        assert isinstance(transforms, Iterable)
        assert all(isinstance(transform, AutoregressiveSubsetTransform2d) for transform in transforms)
        assert isinstance(autoregressive_order, str) or isinstance(autoregressive_order, Iterable)
        assert autoregressive_order in {'raster_scan'}
        self.base_dist = StandardUniform(base_shape)
        self.transforms = nn.ModuleList(transforms)
        self.autoregressive_order = autoregressive_order
        self.bits = bits

    def _x_to_volume(self, x):
        x_lower = x.type(self.base_dist.zero.dtype)/(2**self.bits)
        x_upper = (x.type(self.base_dist.zero.dtype)+1)/(2**self.bits)
        return x_lower, x_upper

    def _y_to_x(self, y):
        return torch.floor((2**self.bits)*y).clamp(max=2**self.bits-1).long()

    def forward_transform(self, x):
        z_lower, z_upper = self._x_to_volume(x)
        for transform in self.transforms:
            z_lower, z_upper = transform(z_lower, z_upper)
        return z_lower, z_upper

    def inverse_transform(self, z, clamp):
        with torch.no_grad():
            if self.autoregressive_order == 'raster_scan': return self._inverse_raster_scan(z, clamp=clamp)

    def _inverse_raster_scan(self, z, clamp):
        x_lower = [torch.zeros_like(z) for _ in range(len(self.transforms))]
        x_upper = [torch.zeros_like(z) for _ in range(len(self.transforms))]
        x = torch.zeros_like(z).long()
        elementwise_params = [None for _ in range(len(self.transforms))]
        for h in range(x.shape[2]):
            for w in range(x.shape[3]):
                for c in range(x.shape[1]):
                    for i, transform in enumerate(self.transforms):
                        elementwise_params[i] = transform.elementwise_params(x_lower[i], x_upper[i])
                    z_chw = z[:,c,h,w]
                    for transform, elementwise_param in zip(reversed(self.transforms), reversed(elementwise_params)):
                        z_chw = transform.elementwise_inverse(z_chw, elementwise_param[:,c,h,w])
                        if clamp: z_chw.clamp(0.0, 1.0)
                    x_chw = self._y_to_x(z_chw)
                    x[:,c,h,w] = x_chw
                    x_lower[0][:,c,h,w], x_upper[0][:,c,h,w] = self._x_to_volume(x_chw)
                    for i, transform in enumerate(self.transforms[:-1]):
                        x_lower[i+1][:,c,h,w], x_upper[i+1][:,c,h,w] = transform.elementwise_forward(x_lower[i][:,c,h,w], x_upper[i][:,c,h,w], elementwise_params[i][:,c,h,w])
        return x

    def log_prob(self, x):
        z_lower, z_upper = self.forward_transform(x)
        log_prob = sum_except_batch(torch.log((z_upper-z_lower).clamp(1e-12))) # Prob. in [0,2^bits]
        return log_prob

    def sample(self, num_samples, clamp=False):
        z = self.base_dist.sample(num_samples)
        return self.inverse_transform(z, clamp=clamp)

    def loglik_bpd(self, x):
        return - self.log_prob(x).sum() / (x.numel() * math.log(2))

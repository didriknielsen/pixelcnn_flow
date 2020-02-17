import torch
from torch import nn
from collections.abc import Iterable
from pixelflow.distributions import Distribution
from pixelflow.transforms import Transform


class Flow(Distribution):
    """
    Base class for Flow.
    Flows use the forward transforms to transform data to noise.
    The inverse transforms can subsequently be used for sampling.
    These are typically useful as generative models of data.
    """

    mode = 'log_prob'
    allowed_modes = {'log_prob', 'sample'}

    def __init__(self, base_dist, transforms):
        super(Flow, self).__init__()
        assert isinstance(base_dist, Distribution)
        if isinstance(transforms, Transform): transforms = [transforms]
        assert isinstance(transforms, Iterable)
        assert all(isinstance(transform, Transform) for transform in transforms)
        self.base_dist = base_dist
        self.transforms = nn.ModuleList(transforms)

    def log_prob(self, x):
        self.base_dist.set_mode('log_prob')
        log_prob = torch.zeros(x.shape[0], device=x.device)
        for transform in self.transforms:
            x, ldj = transform(x)
            log_prob += ldj
        log_prob += self.base_dist(x) # Mode set to log_prob
        return log_prob

    def sample(self, num_samples):
        self.base_dist.set_mode('sample')
        z = self.base_dist(num_samples)
        for transform in reversed(self.transforms):
            z = transform.inverse(z)
        return z

    def sample_with_log_prob(self, num_samples):
        raise RuntimeError("Flows do not support sample_with_log_prob")

    def check_shapes(self, x):
        print("Checking Shapes...")
        with torch.no_grad():
            shape = x.shape
            for n, transform in enumerate(self.transforms):
                expected_shape = transform.z_shape(shape)
                x, _ = transform.forward(x)
                shape = x.shape
                assert shape == expected_shape, 'Expected shape {}, but found shape {} in output of layer {}'.format(expected_shape, shape, n)
        print("Success!")

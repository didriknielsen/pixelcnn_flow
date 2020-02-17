import torch
from torch import nn
from collections.abc import Iterable
from pixelflow.distributions import Distribution
from pixelflow.transforms import Transform


class InverseFlow(Distribution):
    """
    Base class for InverseFlow.
    Inverse flows use the forward transforms to transform noise to samples.
    These are typically useful as variational distributions.
    Here, we are not interested in the log probability of novel samples.
    However, using .sample_with_log_prob(), samples can be obtained together
    with their log probability.
    """

    mode = 'sample_with_log_prob'
    allowed_modes = {'sample_with_log_prob', 'sample'}

    def __init__(self, base_dist, transforms):
        super(InverseFlow, self).__init__()
        assert isinstance(base_dist, Distribution)
        if isinstance(transforms, Transform): transforms = [transforms]
        assert isinstance(transforms, Iterable)
        assert all(isinstance(transform, Transform) for transform in transforms)
        self.base_dist = base_dist
        self.transforms = nn.ModuleList(transforms)

    def log_prob(self, x):
        raise RuntimeError("Inverse flows do not support log_prob")

    def sample(self, num_samples):
        self.base_dist.set_mode('sample')
        z = self.base_dist(num_samples)
        for transform in self.transforms:
            z, _ = transform(z)
        return z

    def sample_with_log_prob(self, num_samples):
        self.base_dist.set_mode('sample_with_log_prob')
        z, log_prob = self.base_dist(num_samples)
        for transform in self.transforms:
            z, ldj = transform(z)
            log_prob -= ldj
        return z, log_prob

    def check_shapes(self, num_samples):
        assert isinstance(num_samples, int)
        print("Checking Shapes...")
        with torch.no_grad():
            x = self.base_dist.sample(num_samples)
            shape = x.shape
            expected_shape = self.base_dist.sample_shape(num_samples)
            assert shape == expected_shape, 'Expected shape {}, but found shape {} in output of layer {}'.format(expected_shape, shape, n)
            for n, transform in enumerate(self.transforms):
                expected_shape = transform.z_shape(shape)
                x, _ = transform(x)
                shape = x.shape
                assert shape == expected_shape, 'Expected shape {}, but found shape {} in output of layer {}'.format(expected_shape, shape, n)
        print("Success!")

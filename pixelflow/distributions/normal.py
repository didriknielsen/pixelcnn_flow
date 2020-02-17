import math
import torch
from pixelflow.distributions import Distribution
from pixelflow.utils import sum_except_batch
from torch.distributions import Normal


class StandardNormal(Distribution):
    """A multivariate Normal with zero mean and unit covariance."""

    def __init__(self, shape):
        super(StandardNormal, self).__init__()
        self.shape = torch.Size(shape)
        self.register_buffer('base_measure', - 0.5 * self.shape.numel() * torch.log(torch.tensor(2 * math.pi)))

    def log_prob(self, x):
        return self.base_measure - 0.5 * sum_except_batch(x**2)

    def sample(self, num_samples):
        return torch.randn((num_samples,) + self.shape, device=self.base_measure.device, dtype=self.base_measure.dtype)

    def sample_shape(self, num_samples):
        return (num_samples,) + self.shape

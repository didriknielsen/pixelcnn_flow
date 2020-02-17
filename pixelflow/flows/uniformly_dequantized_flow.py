import math
import torch
from torch import nn
from collections.abc import Iterable
from pixelflow.distributions import Distribution
from pixelflow.transforms import Transform


class UniformlyDequantizedFlow(Distribution):
    '''
    Continuous Flow with Uniform Dequantization.

    Expects `x` to be categorical, `x \in {0,1,2,...,K-1}`.
    From this, `y = (x+u)/K` is computed.
    Expects `generative_flow` to operate on the dequantized `y \in [0,1]`.
    '''

    mode = 'log_prob'
    allowed_modes = {'log_prob', 'sample'}

    def __init__(self, base_dist, transforms, bits=8):
        super(UniformlyDequantizedFlow, self).__init__()
        assert isinstance(base_dist, Distribution)
        if isinstance(transforms, Transform): transforms = [transforms]
        assert isinstance(transforms, Iterable)
        assert all(isinstance(transform, Transform) for transform in transforms)
        self.base_dist = base_dist
        self.transforms = nn.ModuleList(transforms)
        self.bits = bits

    def continuous_log_prob(self, y):
        self.base_dist.set_mode('log_prob')
        log_prob = torch.zeros(y.shape[0], device=y.device, dtype=y.dtype)
        for transform in self.transforms:
            y, ldj = transform(y)
            log_prob += ldj
        log_prob += self.base_dist(y) # Mode set to log_prob
        return log_prob

    def continuous_sample(self, num_samples):
        self.base_dist.set_mode('sample')
        z = self.base_dist(num_samples)
        for transform in reversed(self.transforms):
            z = transform.inverse(z)
        return z

    def log_prob(self, x):
        self.base_dist.set_mode('log_prob')
        u = torch.rand(x.shape, device=x.device, dtype=next(iter(self.transforms.parameters())).dtype)
        y = (x.type(u.dtype)+u) / (2**self.bits)
        log_prob = self.continuous_log_prob(y) # Prob. in [0,1]
        log_prob -= math.log(2**self.bits) * x.shape[1:].numel() # Prob. in [0,2^bits]
        return log_prob

    def sample(self, num_samples):
        self.base_dist.set_mode('sample')
        y = self.continuous_sample(num_samples)
        x = y * (2**self.bits)
        x = x.floor().clamp(max=2**self.bits-1)
        return x.long()

    def elbo_bpd(self, x):
        '''ELBO in bits/dim'''
        return - self.log_prob(x).sum() / (x.numel() * math.log(2))

    def iwbo_bpd(self, x, k, batch_size=None):
        '''IWAE bound in bits/dim'''
        if batch_size:
            return self._iwbo_batched(x, k, batch_size)
        else:
            return self._iwbo(x, k)

    def _iwbo(self, x, k):
        x_stack = torch.cat([x for _ in range(k)], dim=0)
        ll_stack = self.log_prob(x_stack)
        ll = torch.stack(torch.chunk(ll_stack, k, dim=0))
        iwae = torch.logsumexp(ll, dim=0) - math.log(k)
        return - iwae.sum() / (x.numel() * math.log(2))

    def _iwbo_batched(self, x, k, batch_size):
        assert k % batch_size == 0
        passes = k // batch_size
        ll_batched = []
        for i in range(passes):
            x_stack = torch.cat([x for _ in range(batch_size)], dim=0)
            ll_stack = self.log_prob(x_stack)
            ll_batched.append(torch.stack(torch.chunk(ll_stack, k, dim=0)))
        ll = torch.cat(ll_batched, dim=0)
        iwae = torch.logsumexp(ll, dim=0) - math.log(k)
        return - iwae.sum() / (x.numel() * math.log(2))

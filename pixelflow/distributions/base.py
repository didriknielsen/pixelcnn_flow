import torch
from torch import nn


class DataParallelDistribution(nn.DataParallel):
    """
    A DataParallel wrapper for Distribution.
    To be used instead of nn.DataParallel for Distribution objects.
    """

    def set_mode(self, mode):
        self.module.set_mode(mode)

    def log_prob(self, *args, **kwargs):
        self.set_mode('log_prob')
        return self.forward(*args, **kwargs)

    def sample(self, *args, **kwargs):
        self.set_mode('sample')
        return self.forward(*args, **kwargs)

    def sample_with_log_prob(self, *args, **kwargs):
        self.set_mode('sample_with_log_prob')
        return self.forward(*args, **kwargs)


class Distribution(nn.Module):
    """Distribution base class"""

    mode = 'log_prob'
    allowed_modes = {'log_prob', 'sample', 'sample_with_log_prob'}

    def _assert_allowed_mode(self, mode):
        assert mode in self.allowed_modes, 'Got mode {}, but needs to be in {}'.format(mode, str(self.allowed_modes))

    def set_mode(self, mode):
        '''
        Set mode for .forward().
        '''
        self._assert_allowed_mode(mode)
        self.mode = mode

    def forward(self, *args, **kwargs):
        '''
        Calls either {.log_prob(), .sample(), .sample_with_log_prob()}
        depending on self.mode.
        To allow Distribution objects to be wrapped by DataParallelDistribution,
        which parallelizes .forward() of replicas on subsets of data.
        '''
        if self.mode == 'log_prob':
            return self.log_prob(*args, **kwargs)
        elif self.mode == 'sample':
            return self.sample(*args, **kwargs)
        elif self.mode == 'sample_with_log_prob':
            return self.sample_with_log_prob(*args, **kwargs)

    def log_prob(self, x):
        """Calculate log probability under the distribution.

        Args:
            x: Tensor, shape (batch_size, ...)

        Returns:
            log_prob: Tensor, shape (batch_size,)
        """
        raise NotImplementedError()

    def sample(self, num_samples):
        """Generates samples from the distribution.

        Args:
            num_samples: int, number of samples to generate.

        Returns:
            samples: Tensor, shape (num_samples, ...)
        """
        raise NotImplementedError()

    def sample_with_log_prob(self, num_samples):
        """Generates samples from the distribution together with their log probability.

        Args:
            num_samples: int, number of samples to generate.

        Returns:
            samples: Tensor, shape (num_samples, ...)
            log_prob: Tensor, shape (num_samples,)
        """
        samples = self.sample(num_samples)
        log_prob = self.log_prob(samples)
        return samples, log_prob

    def sample_shape(self, num_samples):
        """The shape of samples from the distribution.

        Args:
            num_samples: int, number of samples.

        Returns:
            sample_shape: torch.Size
        """
        raise NotImplementedError

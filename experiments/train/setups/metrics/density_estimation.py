import math
import torch
import torch.nn.functional as F

from .metric import Metric


class NegAvgLogLikelihood(Metric):

    def add_batch(self, log_probs, numel):
        '''
        Computes the negative average log-likelihood (nats per dimension).

        Input:
            log_probs: torch.FloatTensor, (batch_size)
                The log probabilities of the observations.
            numel: int
                The number of elements per observation (for averaging).
        '''
        self._batch_count += 1
        self._data_count += log_probs.shape[0]
        self._aggregated_value += - log_probs.sum().detach().cpu().item() / numel


class BitsPerDim(NegAvgLogLikelihood):
    '''Computes the bits per dimension.'''

    def __init__(self, range_scaling_factor=256):
        self.range_scaling_factor = range_scaling_factor
        super(BitsPerDim, self).__init__()

    def get_value(self):
        nats_per_dim_01 = super(BitsPerDim, self).get_value()
        return (nats_per_dim_01 + math.log(self.range_scaling_factor)) / math.log(2.0)

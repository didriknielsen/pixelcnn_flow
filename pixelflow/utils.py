import torch


def sum_except_batch(x):
    '''
    Sums all dimensions except the first.

    Args:
        x: Tensor, shape (batch_size, ...)

    Returns:
        x_sum: Tensor, shape (batch_size,)
    '''
    return x.reshape(x.shape[0], -1).sum(-1)


def mean_except_batch(x):
    '''
    Averages all dimensions except the first.

    Args:
        x: Tensor, shape (batch_size, ...)

    Returns:
        x_mean: Tensor, shape (batch_size,)
    '''
    return x.reshape(x.shape[0], -1).mean(-1)


def quantize(x, num_bins=256):
    return torch.floor(num_bins * x).clamp(max=num_bins-1) / (num_bins-1)

import torch


def gather_elementwise(tensor, idx_tensor):
    '''
    For `tensor.shape = tensor_shape + (K,)`
    and `idx_tensor.shape = tensor_shape` with elements in {0,1,...,K-1}
    '''
    return tensor.gather(-1, idx_tensor[..., None])[..., 0]

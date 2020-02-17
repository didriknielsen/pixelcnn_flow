import os
import random
import numpy as np
import torch


def set_env(gpu_ids, verbose=True):
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(id) for id in gpu_ids)
    if verbose:
        print("CUDA_VISIBLE_DEVICES =", os.environ["CUDA_VISIBLE_DEVICES"])


def set_seeds(seed, cuda_deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if cuda_deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

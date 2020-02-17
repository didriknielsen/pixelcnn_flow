import copy
import torch


def get_mixture_params(params, num_mixtures):
    '''Get parameters for mixture transforms.'''
    assert params.shape[-1] == 3 * num_mixtures

    unnormalized_weights = params[..., :num_mixtures]
    means = params[..., num_mixtures:2*num_mixtures]
    log_scales = params[..., 2*num_mixtures:3*num_mixtures]

    return unnormalized_weights, means, log_scales


def get_multivariate_mixture_params_2d(params, num_mixtures, channels=3):
    '''Get parameters for multivariate mixture 2d transforms.'''
    num_corr_params = channels * (channels - 1) // 2
    num_params_per_mix = 1 + 2 * channels + num_corr_params
    num_params_total = num_mixtures * num_params_per_mix
    assert params.dim() == 4
    assert params.shape[1] == num_params_total

    w_idx = torch.arange(num_mixtures, device=params.device)
    m_idx = torch.arange(num_mixtures, num_mixtures * (1 + channels), device=params.device)
    s_idx = torch.arange(num_mixtures * (1 + channels), num_mixtures * (1 + 2 * channels), device=params.device)
    r_idx = torch.arange(num_mixtures * (1 + 2 * channels), num_params_total, device=params.device)

    ## Extract different parameters
    r_unnormalized_weights = torch.index_select(params, 1, w_idx)
    r_unadjusted_means = torch.index_select(params, 1, m_idx)
    r_log_scales = torch.index_select(params, 1, s_idx)
    r_unnormalized_corr = torch.index_select(params, 1, r_idx)

    ## Reshape unnormalized_weights
    # From shape (bs,m,h,w)
    #         to (bs,m,h,w,1)
    #         to (bs,1,h,w,m)
    r_unnormalized_weights = r_unnormalized_weights.unsqueeze(-1)
    unnormalized_weights = r_unnormalized_weights.transpose(1,4)

    ## Reshape log_scales & unadjusted_means
    # From shape (bs,m*c,h w)
    #         to (bs,m,c,h w)
    #         to (bs,c,h w,m)
    tmp_shape = r_log_scales.shape[0:1] + (num_mixtures, channels) + r_log_scales.shape[2:]
    log_scales = r_log_scales.reshape(tmp_shape).permute(0,2,3,4,1)
    unadjusted_means = r_unadjusted_means.reshape(tmp_shape).permute(0,2,3,4,1)

    ## Reshape unnormalized_corr
    # For l = c*(c-1)//2,
    # From shape (bs,m*l,h w)
    #         to (bs,m,l,h w)
    #         to (bs,l,h w,m)
    tmp_shape = r_unnormalized_corr.shape[0:1] + (num_mixtures, num_corr_params) + r_unnormalized_corr.shape[2:]
    unnormalized_corr = r_unnormalized_corr.reshape(tmp_shape).permute(0,2,3,4,1)

    return unnormalized_weights, unadjusted_means, log_scales, unnormalized_corr


def get_matching_multivariate_mixture_params_2d(params, num_mixtures):
    '''
    Get parameters for multivariate mixture 2d transforms matching the ordering in the original PixelCNN++ code.
    This allows one to extract parameters for model pre-trained with the original code.
    '''
    assert params.dim() == 4
    assert params.shape[1] == 100

    # Pytorch ordering
    l = params.permute(0, 2, 3, 1)
    ls = [int(y) for y in l.size()]
    xs = copy.deepcopy(ls)
    xs[-1] = 3

    # here and below: unpacking the params of the mixture of logistics
    nr_mix = int(ls[-1] / 10)
    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 3]) # 3 for mean, scale, coef
    means = l[:, :, :, :, :nr_mix]
    log_scales = l[:, :, :, :, nr_mix:2 * nr_mix]

    coeffs = l[:, :, :, :, 2 * nr_mix:3 * nr_mix]

    return logit_probs.unsqueeze(1), means.permute(0,3,1,2,4), log_scales.permute(0,3,1,2,4), coeffs.permute(0,3,1,2,4)

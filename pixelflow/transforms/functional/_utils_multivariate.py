import torch


def adjust_log_weights(log_weights, component_log_probs):
    '''
    Adjust log_weights for multivariate mixture distributions.

    Uses that `sum_m w_m p_m1(x_1) p_m2(x_2|x_1) p_m3(x_3|x_2,x_1) = [sum_m w_m1 p_m1(x_1)] [sum_m w_m2 p_m2(x_2|x_1)] [sum_m w_m3 p_m3(x_3|x_2,x_1)]`.
    This computes `[w_m1, w_m2, w_m3]` given `w_m` and `[log p_m1(x_1), log p_m2(x_2|x_1), log p_m3(x_3|x_2,x_1)]`.
    Here,
    w_m1 = w_m
    w_m2 = w_m p_m1(x_1) / [sum_s w_s p_s1(x_1)]
    w_m3 = w_m p_m2(x_2|x_1) p_m1(x_1) / [sum_s w_s p_s2(x_2|x_1) p_s1(x_1)].

    With l=c*(c-1)//2,
    log_weights.shape = (b,1,h,w,m)
    component_log_probs.shape = (b,c,h,w,m)
    '''
    # Adjust log_weights (autoregressively from (b,1,h,w,m) to (b,c,h,w,m))
    log_weights = log_weights + torch.cat([torch.zeros_like(log_weights), torch.cumsum(component_log_probs, dim=1)[:,:-1]], dim=1)
    log_weights = log_weights - torch.logsumexp(log_weights, dim=-1, keepdim=True)
    return log_weights


def adjust_means(unadjusted_means, corr, x, in_lambd=lambda x: 2*x-1):
    '''
    Adjust means for multivariate mixture distributions.

    This computes
    m_1 = m_1
    m_2 = m_2 + r_1 x_1
    m_3 = m_3 + r_2 x_1 + r_3 x_2

    With l=c*(c-1)//2,
    unadjusted_means.shape = (b,c,h,w,m)
    corr.shape = (b,l,h,w,m)
    x.shape = (b,c,h,w)
    '''
    xp = in_lambd(x)
    adjustments = torch.empty_like(unadjusted_means)
    for c in range(x.shape[1]):
        start = c * (c - 1) // 2
        stop = (c + 1) * c // 2
        adjustments[:,c] = torch.sum(corr[:,start:stop] * xp.unsqueeze(-1)[:,:c], dim=1)
    return unadjusted_means + adjustments


## Vectorized implementation of adjust_means (not faster in practice, even when R is computed once outside):
# def adjust_means(unadjusted_means, corr, x):
#     '''
#     Adjust means for multivariate MOL.
#
#     This computes
#     m_1 = m_1
#     m_2 = m_2 + r_1 x_1
#     m_3 = m_3 + r_2 x_1 + r_3 x_2
#
#     With l=c*(c-1)//2,
#     unadjusted_means.shape = (b,c,h,w,m)
#     corr.shape = (b,l,h,w,m)
#     x.shape = (b,c,h,w)
#     '''
#     b, c = inputs.shape[0], inputs.shape[1]
#     idx = torch.tril(torch.ones(c, c), diagonal=-1) == 1
#     R = torch.zeros(b, c, c, *corr.shape[2:], dtype=corr.dtype, device=corr.device)
#     R_idx = idx.reshape(1,c,c,1,1,1).expand_as(R)
#     R[R_idx] = corr.view(-1)
#     return unadjusted_means + torch.einsum('bcdhwm,bdhw->bchwm', R, x)

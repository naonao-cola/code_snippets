'''
Copyright (C) 2023 TuringVision

Image filter function for pytorch.
'''

__all__ = ['ncc']


def ncc(x, y):
    ''' Normalized cross-correlation
    see cv2.TM_CCOEFF_NORMED
    In:
        x: (M, K) tensor
        y: (N, K) tensor
    Out:
        (M, N) tensor
    '''
    import torch
    x_mean = x.mean(dim=-1, keepdim=True)
    y_mean = y.mean(dim=-1, keepdim=True)
    x = x - x_mean
    y = y - y_mean
    ncc_score_num = torch.einsum('ij,kj->ik', x, y)
    x_2_sum = torch.sum(x ** 2, dim=-1, keepdim=False)
    y_2_sum = torch.sum(y ** 2, dim=-1, keepdim=False)
    ncc_score_den = torch.einsum('i,j->ij', x_2_sum, y_2_sum)
    ncc_score_den = torch.sqrt(ncc_score_den)
    ncc_score_den = torch.max(torch.tensor(1e-7, device=x.device), ncc_score_den)
    return torch.clamp(ncc_score_num / ncc_score_den, -1, 1)

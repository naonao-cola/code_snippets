'''
Copyright (C) 2023 TuringVision

Image filter function for pytorch.
'''
import numpy as np

__all__ = ['gaussian_blur', 'median_blur', 'peak_local_max']

def _gaussian_kernel(ksize=5, sig=1.):
    """\
    creates gaussian kernel with side length ksize and a sigma of sig
    """
    ax = np.arange(-ksize // 2 + 1., ksize // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
    kernel = kernel / np.sum(kernel)
    return kernel.astype(np.float32)


def gaussian_blur(img_batch, ksize=5, sig=1.):
    '''
    In:
        img_batch: (NxCxHxW)
    Out:
        blur_img (NxCxHxW)
    '''
    import torch
    with torch.no_grad():
        channel = img_batch.shape[1]
        ksize = ksize + 1 -  ksize % 2
        gkernel = torch.from_numpy(_gaussian_kernel(ksize=ksize, sig=sig))
        gkernel = gkernel[np.newaxis, np.newaxis]
        gkernel = gkernel.to(img_batch.device)
        if channel > 1:
            gkernel = torch.cat((gkernel,)*channel, dim=0)
        padding = ksize // 2
        blur_img = torch.conv2d(img_batch, gkernel, padding=padding, groups=channel)
        return blur_img


def median_blur(img_batch, ksize=5):
    '''
    In: NxCxHxW
    Out: NxCxHxW
    '''
    import torch.nn.functional as F
    ksize = ksize + 1 -  ksize % 2
    padding = ksize // 2
    x = F.pad(img_batch, (padding, )*4, mode='constant', value=0)
    x = x.unfold(2, ksize, 1).unfold(3, ksize, 1)
    x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
    return x


def _get_max_coord(t, mask, threshold=0.9, num_peaks=8):
    import torch
    n, h, w = t.shape
    t[mask == 0] = 0
    scores = t.reshape(n, -1)
    values_list, idxs_list = scores.topk(num_peaks, dim=1)
    values_list = values_list.cpu().numpy()
    idxs_list = idxs_list.cpu().numpy()

    zyx_out_list = [list() for _ in range(n)]
    for i in range(n):
        values = values_list[i]
        idxs = idxs_list[i]
        for v, idx in zip(values, idxs):
            if v > 0 and (v >= threshold or len(zyx_out_list[i]) < 2):
                y = idx // w
                x = idx % w
                zyx_out_list[i].append((y, x))
    return zyx_out_list


def peak_local_max(t, min_distance=(31, 31),
                   threshold_low=0.5,
                   threshold_high=0.9,
                   indices=True,
                   num_peaks=1000):
    ''' Find peaks in an image as boolean mask.
    In:
        t: (NxHxW) tensor of image batch
        indices : bool, optional
            If True, the output will be an array representing peak coordinates.
            If False, the output will be a boolean array shaped as
            `image.shape` with peaks present at True elements.
        threshold_low : float, optional
            Minimum intensity of average peak.
        threshold_high : float, optional
            Minimum intensity of good peaks.
        num_peaks : int, optional
            Maximum number of peaks. When the number of peaks exceeds `num_peaks`,
            return `num_peaks` peaks based on highest peak intensity.

    output : ndarray or ndarray of bools

            * If `indices = True`  : (row, column, ...) coordinates of peaks.
            * If `indices = False` : Boolean array shaped like `image`, with peaks
            represented by True values.
    '''
    import torch
    pool_kernel = (min_distance[0] - (1 - min_distance[0]%2),
                   min_distance[1] - (1 - min_distance[1]%2))
    padding = (pool_kernel[0]//2, pool_kernel[1]//2)
    # NxHxW -> Nx1xHxW
    t = t.unsqueeze(1)
    max_poll_t = torch.nn.MaxPool2d(pool_kernel, stride=1, padding=padding)(t)
    local_max_mask = (max_poll_t == t) * (max_poll_t >= threshold_low)
    # Nx1xHxW -> NxHxW
    local_max_mask = local_max_mask[:, 0]
    if indices is False:
        return local_max_mask

    # Nx1xHxW -> NxHxW
    t = t[:, 0]
    zyx_list = _get_max_coord(t, local_max_mask,
                              threshold_high,
                              num_peaks=num_peaks)
    return zyx_list

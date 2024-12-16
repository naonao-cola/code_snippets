'''
Copyright (C) 2019 ThunderSoft

Phase only defect detector
'''
def _rfft(self,
        signal_ndim: int,
        normalized: bool = False,
        onesided: bool = True
        ):
    import torch
    HAS_FFT_MODULE = (torch.__version__ >= "1.7.0")
        # old-day's torch.rfft
    if not HAS_FFT_MODULE:
        return torch.rfft(self, signal_ndim, normalized, onesided)
    import torch.fft

    if signal_ndim > 4:
        raise RuntimeError("signal_ndim is expected to be 1, 2, 3.")

    m = torch.fft.rfftn if onesided else torch.fft.fftn
    dim = [-3, -2, -1][3 - signal_ndim:]
    return torch.view_as_real(m(self, dim=dim, norm="ortho" if normalized else None))


def _irfft(self,
        signal_ndim: int,
        normalized: bool = False,
        onesided: bool = True,
        ):
    import torch
    HAS_FFT_MODULE = (torch.__version__ >= "1.7.0")
    # old-day's torch.irfft
    if not HAS_FFT_MODULE:
        return torch.irfft(self, signal_ndim, normalized, onesided)
    import torch.fft

    if signal_ndim > 4:
        raise RuntimeError("signal_ndim is expected to be 1, 2, 3.")
    if not torch.is_complex(self):
        self = torch.view_as_complex(self)

    m = torch.fft.irfftn if onesided else torch.fft.ifftn
    size = list(self.size())
    size[-1] = 2*size[-1] - 1
    out = m(self, s=size, norm="ortho" if normalized else None)
    return out.real if torch.is_complex(out) else out


def get_phase_only_img(image, blur_func):
    '''
    In:
        image: NxCxHxW torch.float32
    Out:
        phase_only_img: NxCxHxW torch.float32
    '''
    import torch
    dft = _rfft(image, 2)
    mag = dft ** 2
    # NxCxHxWx2 -> NxCxHxW
    mag = mag.sum(dim=-1) ** 0.5 + 1e-12
    # NxCxHxW -> NxCxHxWx1
    mag = mag.unsqueeze(-1)
    # NxCxHxWx1 -> NxHxWx1
    mag = mag.mean(dim=1)
    # NxHxWx1 -> Nx1xHxWx1
    mag = mag.unsqueeze(1)
    phase = dft / mag
    phase_only_img = _irfft(phase, 2)
    phase_only_img = blur_func(phase_only_img)
    phase_only_img = phase_only_img.abs()
    return phase_only_img


def get_tiled_phase_only_img(img_batch, tile_w, tile_h, blur_func, device='cpu'):
    '''Calculated by dividing the large image into four completely repeating
       small images from left-top, right-top, left-bottom, right-bottom
    In:
        img_batch: (batch_size, 1, height, width) np.uint8
            or list of (channel, height, width) np.uint8
        tile_w: int
        tile_h: int
        blur_func: callable
        device: 'cpu' or 'cuda'
    Out:
        amap_batch: (batch_size, height, width) torch.float32
    '''
    import torch
    with torch.no_grad():
        batch_size = len(img_batch)
        _, height, width = img_batch[0].shape

        tiled_img_batch = torch.zeros((batch_size, 4, tile_h, tile_w), dtype=torch.float32, device=device)
        amap_batch = torch.full((batch_size, height, width), 10000, dtype=torch.float32, device=device)
        crop_coord_list = [
            (0, 0, tile_w, tile_h),
            (width - tile_w, 0, width, tile_h),
            (0, height - tile_h, tile_w, height),
            (width - tile_w, height - tile_h, width, height)
        ]
        dst_paste_coord_list = [
            (0, 0, tile_w, tile_h),
            (width - tile_w, 0, width, tile_h),
            (0, height - tile_h, tile_w, height),
            (width - tile_w, height - tile_h, width, height)
        ]

        for i in range(batch_size):
            img_t = torch.from_numpy(img_batch[i][0])
            if device != 'cpu':
                img_t = img_t.cuda()
            for c in range(4):
                sx, sy, ex, ey = crop_coord_list[c]
                tiled_img_batch[i, c, :, :] = img_t[sy:ey, sx:ex]

        map_s = get_phase_only_img(tiled_img_batch, blur_func)
        for c in range(4):
            sx, sy, ex, ey = dst_paste_coord_list[c]
            amap_batch_c = map_s[:, c, 0:tile_h, 0:tile_w]
            amap_batch[:, sy:ey, sx:ex] = torch.min(amap_batch_c, amap_batch[:, sy:ey, sx:ex])
        return amap_batch


def get_tile_shape(img_gray):
    '''Calculate the top-right clipping area by the matching position of the
         bottom-left corner of the image.
    In: (H, W) np.array np.uint8
    Out: (H, W) int
    '''
    import cv2
    import numpy as np
    from scipy.signal import argrelextrema

    height, width = img_gray.shape
    method = cv2.TM_SQDIFF
    t_w = width//12
    t_h = height//12
    x_end = width//3
    y_shift = height*2//3
    bottom_left_area = img_gray[y_shift:, :x_end]
    top_right_area = img_gray[:t_h, -t_w:]
    res = cv2.matchTemplate(bottom_left_area, top_right_area, method)
    res_arg_min = res.argmin()
    match_y, match_x = res_arg_min//res.shape[1], res_arg_min%res.shape[1]

    # get the bottommost match
    v_best_res = res[:, match_x]
    local_min_idxs = argrelextrema(v_best_res, np.less, order=20)[0]
    vmin = v_best_res.min()
    vmax = v_best_res.max()
    vmin_trehsold = (vmin + (vmax - vmin)*0.2)
    for idx in reversed(local_min_idxs):
        if v_best_res[idx] <= vmin_trehsold:
            match_y = idx
            break

    # get the leftmost match
    h_best_res = res[match_y]
    local_min_idxs = argrelextrema(h_best_res, np.less, order=20)[0]
    hmin = h_best_res.min()
    hmax = h_best_res.max()
    hmin_trehsold = (hmin + (hmax - hmin)*0.2)
    for idx in local_min_idxs:
        if h_best_res[idx] <= hmin_trehsold:
            match_x = idx
            break
    return (match_y+y_shift, width-(match_x+t_w))

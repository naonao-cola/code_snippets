'''
Copyright (C) 2023 TuringVision

Phase only defect detector
'''
import cv2
import numpy as np
from functools import partial
from .basic_detector import BasicDefectDetector


__all__ = ['PhotDefectDetector', 'TiledPhotDefectDetector']

CLEAN_BORDER_MARGIN = 5

def _gaussian_kernel(ksize=5, sig=1.):
    """\
    creates gaussian kernel with side length ksize and a sigma of sig
    """
    ax = np.arange(-ksize // 2 + 1., ksize // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
    kernel = kernel / np.sum(kernel)
    return kernel.astype(np.float32)


def _gaussian_blur(img_batch, ksize=5, sig=1., use_gpu=True):
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
        if use_gpu:
            gkernel = gkernel.cuda()
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

def _border_clean(amap_batch):
    '''
    In:
        amap_batch: (NxHxW)
    Out:
        amap_batch (NxHxW)
    '''
    import torch
    with torch.no_grad():
        border = CLEAN_BORDER_MARGIN
        amap_batch[:, 0:border, :] = 0
        amap_batch[:, -border:, :] = 0
        amap_batch[:, :, 0:border] = 0
        amap_batch[:, :, -border:] = 0
        return amap_batch


class PhotDefectDetector(BasicDefectDetector):
    def __init__(self, border_coef=1.618, blur_ksize=7, blur_sig=2.0, use_gpu=True):
        '''
        border_coef: Image boundary weakening factor, 1.618 means significant weakening,
                     1.0 means no weakening
        blur_ksize: kernel size of gaussian blur
        blur_sig: sigma of gaussian blur
        use_gpu:
        '''
        BasicDefectDetector.__init__(self, use_gpu)
        self._border_coef = border_coef
        self._blur_func = partial(_gaussian_blur, ksize=blur_ksize,
                                  sig=blur_sig, use_gpu=use_gpu)


    def get_phase_only_img(self, img_batch, border_clean=True):
        '''
        In:
            img_batch: (batch_size, height, width) np.uint8
                or list of (height, width) np.uint8
            border_clean: (bool) Is need clean border?
        Out:
            po_img_batch: (batch_size, height, width) torch.float32
        '''
        import torch
        from .impl import phot_detector_impl
        for i in range(len(img_batch)):
            img_batch[i] = img_batch[i][np.newaxis]

        with torch.no_grad():
            batch_size = len(img_batch)
            channel, height, width = img_batch[0].shape
            device = 'cuda' if self._use_gpu else 'cpu'
            img_batch_t = torch.zeros(batch_size, channel, height, width,
                                      dtype=torch.float32, device=device)
            for i in range(batch_size):
                img_t = torch.from_numpy(img_batch[i])
                if self._use_gpu:
                    img_t = img_t.cuda()
                img_batch_t[i, :, :, :] = img_t
            amap_batch = phot_detector_impl.get_phase_only_img(img_batch_t, self._blur_func)[:, 0]
            if border_clean:
                amap_batch = self.amap_border_clean(amap_batch)
            return amap_batch

    def amap_border_clean(self, amap_batch):
        if self._border_coef > 1.0:
            border_ratio = 0.1
            bw = int(amap_batch.shape[-1] * border_ratio)
            bh = int(amap_batch.shape[-2] * border_ratio)
            amap_batch[:, :bh] /= np.sqrt(self._border_coef)
            amap_batch[:, :2*bh] /= np.sqrt(self._border_coef)
            amap_batch[:, -bh:] /= np.sqrt(self._border_coef)
            amap_batch[:, -2*bh:] /= np.sqrt(self._border_coef)
            amap_batch[:, :bw] /= np.sqrt(self._border_coef)
            amap_batch[:, :, :2*bw] /= np.sqrt(self._border_coef)
            amap_batch[:, :, -bw:] /= np.sqrt(self._border_coef)
            amap_batch[:, :, -2*bw:] /= np.sqrt(self._border_coef)
        amap_batch = _border_clean(amap_batch)
        return amap_batch

    def amap_normalize(self, amap_batch):
        import torch
        for i, amap in enumerate(amap_batch):
            max_value = max(float(amap.max()), 0.0001)
            amap = (amap * (1/max_value))
            amap = amap ** 2
            amap = amap * 255
            amap_batch[i] = torch.clamp_min(amap - 7, 0)

        amap_uint8 = amap_batch.type(torch.uint8)
        amap_uint8 = median_blur(amap_uint8.unsqueeze(1), ksize=5)[:, 0]
        amap_uint8 = amap_uint8.cpu().numpy()
        return amap_uint8

    def get_anomaly_map_from_gray(self, img_batch):
        '''
        In:
            img_batch: (batch_size, height, width) np.uint8
                or list of (height, width) np.uint8
        Out:
            amap: (batch_size, height, width) np.uint8
        '''
        amap_batch = self.get_phase_only_img(img_batch)
        amap_uint8 = self.amap_normalize(amap_batch)
        return amap_uint8


class TiledPhotDefectDetector(PhotDefectDetector):
    def __init__(self, **kwargs):
        '''
        border_coef: Image boundary weakening factor, 1.618 means significant weakening,
                     1.0 means no weakening
        blur_ksize: kernel size of gaussian blur
        blur_sig: sigma of gaussian blur
        use_gpu:
        '''
        PhotDefectDetector.__init__(self, **kwargs)

    def get_tile_shape(self, img_gray):
        '''Calculate the top-right clipping area by the matching position of the
             bottom-left corner of the image.
        In: (H, W) np.array np.uint8
        Out: (H, W) int
        '''
        from .impl import phot_detector_impl
        return phot_detector_impl.get_tile_shape(img_gray)

    def get_phase_only_img(self, img_batch, border_clean=True):
        '''Calculated by dividing the large image into four completely repeating
           small images from left-top, right-top, left-bottom, right-bottom
        In:
            img_batch: (batch_size, height, width) np.uint8
                or list of (height, width) np.uint8
            border_clean: (bool) Is need clean border?
        Out:
            amap_batch: (batch_size, height, width) torch.float32
        '''
        from .impl import phot_detector_impl
        for i in range(len(img_batch)):
            img_batch[i] = img_batch[i][np.newaxis]

        _, height, width = img_batch[0].shape
        img_shape = (height, width)
        tile_h, tile_w = self.get_tile_shape(img_batch[0][0])
        tile_shape = (tile_h, tile_w)

        device = 'cuda' if self._use_gpu else 'cpu'
        amap_batch = phot_detector_impl.get_tiled_phase_only_img(img_batch,
                                                                 tile_w, tile_h,
                                                                 self._blur_func,
                                                                 device)
        if border_clean:
            if self._border_coef:
                height, width = img_shape
                tile_h, tile_w = tile_shape
                crop_coord_list = [
                    (0, 0, tile_w, tile_h),
                    (width - tile_w, 0, width, tile_h),
                    (0, height - tile_h, tile_w, height),
                    (width - tile_w, height - tile_h, width, height)
                ]
                for c in range(4):
                    sx, sy, ex, ey = crop_coord_list[c]
                    if sy > 0:
                        amap_batch[:, :1*sy] /= np.sqrt(self._border_coef)
                        amap_batch[:, :2*sy] /= np.sqrt(self._border_coef)
                    if sx > 0:
                        amap_batch[:, :, :1*sx] /= np.sqrt(self._border_coef)
                        amap_batch[:, :, :2*sx] /= np.sqrt(self._border_coef)
                    if ey < height:
                        ey = ey - height
                        amap_batch[:, 1*ey:] /= np.sqrt(self._border_coef)
                        amap_batch[:, 2*ey:] /= np.sqrt(self._border_coef)
                    if ex < width:
                        ex = ex - width
                        amap_batch[:, :, 1*ex:] /= np.sqrt(self._border_coef)
                        amap_batch[:, :, 2*ex:] /= np.sqrt(self._border_coef)
            amap_batch = _border_clean(amap_batch)
        return amap_batch

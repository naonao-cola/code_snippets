'''
Copyright (C) 2023 TuringVision

Template matching defect detector
'''

import cv2
import numpy as np
from functools import partial
from .basic_detector import BasicDefectDetector


__all__ = ['MatchTemplateDetector', 'FastMatchTemplateDetector']


class MatchTemplateDetector(BasicDefectDetector):
    def __init__(self, pattern_shape=(64, 64),
                 stride=(32, 32),
                 border_coef=1.618,
                 threshold_low=0.90,
                 threshold_high=0.95,
                 sub_constant=6,
                 max_similar_cnt=4,
                 blur_ksize=7, blur_sig=2.0, device='cuda'):
        '''
        pattern_shape: (h, w)
        stride: (h, w)
        threshold_high : float, optional
            Minimum intensity of good similar patterns.
        threshold_low : float, optional
            Minimum intensity of average similar pattern.
        max_similar_cnt : int, optional
            Maximum number of similar patterns used to calculate diff.
        sub_constant: int
        border_coef: Image boundary weakening factor, 1.618 means significant weakening,
                     1.0 means no weakening
        blur_ksize: kernel size of gaussian blur
        blur_sig: sigma of gaussian blur
        '''
        from ..cv.filter import gaussian_blur
        self._device = device
        BasicDefectDetector.__init__(self, device != 'cpu')
        self._border_coef = border_coef
        self._pattern_shape = pattern_shape
        self._stride = stride
        self._sub_constant = sub_constant
        self._threshold_low = threshold_low
        self._threshold_high = threshold_high
        self._max_similar_cnt = max_similar_cnt
        self._blur_func = partial(gaussian_blur, ksize=blur_ksize, sig=blur_sig)

    def get_primary_anomaly_map(self, img_batch, border_clean=True, **kwargs):
        '''
        In:
            img_batch: (batch_size, height, width) np.uint8
                or list of (height, width) np.uint8
        Out:
            amap_batch: (batch_size, height, width) torch.float32
        '''
        import torch
        from .impl import match_template_detector_impl
        with torch.no_grad():
            amap_batch = list()
            for i in range(len(img_batch)):
                img_t = torch.from_numpy(img_batch[i])
                if self._device != 'cpu':
                    img_t = img_t.cuda(self._device)
                func = match_template_detector_impl.get_saliency_map
                amap = func(img_t, self._pattern_shape,
                            self._stride,
                            self._threshold_low,
                            self._threshold_high,
                            self._max_similar_cnt, **kwargs)
                amap_batch.append(amap)
            # NxHxW
            amap_batch = torch.stack(amap_batch)
            # NxHxW -> Nx1xHxW
            amap_batch = amap_batch.unsqueeze(1)
            self._blur_func(amap_batch)
            amap_batch = amap_batch[:, 0]
            if border_clean:
                amap_batch = self.amap_border_clean(amap_batch)
            return amap_batch

    def amap_border_clean(self, amap_batch):
        if self._border_coef > 1.0:
            bw, bh = self._pattern_shape
            amap_batch[:, :bh] /= np.sqrt(self._border_coef)
            amap_batch[:, :2*bh] /= np.sqrt(self._border_coef)
            amap_batch[:, -bh:] /= np.sqrt(self._border_coef)
            amap_batch[:, -2*bh:] /= np.sqrt(self._border_coef)
            amap_batch[:, :, :bw] /= np.sqrt(self._border_coef)
            amap_batch[:, :, :2*bw] /= np.sqrt(self._border_coef)
            amap_batch[:, :, -bw:] /= np.sqrt(self._border_coef)
            amap_batch[:, :, -2*bw:] /= np.sqrt(self._border_coef)
        return amap_batch

    def amap_normalize(self, amap_batch):
        import torch
        from ..cv.filter import median_blur
        amap_batch = median_blur(amap_batch.unsqueeze(1), ksize=5)[:, 0]
        amap_batch = torch.clamp_min(amap_batch - self._sub_constant, 0)
        for i, amap in enumerate(amap_batch):
            max_value = max(float(amap.max()), 0.0001)
            amap = (amap * (1/max_value))
            amap = amap ** 2
            amap = amap * 255
            amap_batch[i] = amap

        amap_uint8 = amap_batch.type(torch.uint8)
        amap_uint8 = amap_uint8.cpu().numpy()
        return amap_uint8

    def get_anomaly_map_from_gray(self, img_batch, **kwargs):
        '''
        In:
            img_batch: (batch_size, height, width) np.uint8
                or list of (height, width) np.uint8
        Out:
            amap: (batch_size, height, width) np.uint8
        '''
        amap_batch = self.get_primary_anomaly_map(img_batch, **kwargs)
        amap_uint8 = self.amap_normalize(amap_batch)
        return amap_uint8


class FastMatchTemplateDetector(BasicDefectDetector):
    def __init__(self,
                 pattern_shape_s=[(350, 8), (8, 350)],
                 stride_s=[(162, 8), (8, 162)],
                 device='cuda'):
        '''
        pattern_shape_s (list): [(h, w), ...]
        stride_s (list): [(h, w), ...]
        '''
        self._device = device
        BasicDefectDetector.__init__(self, device != 'cpu')
        self._pattern_shape_s = pattern_shape_s
        self._stride_s = stride_s

    def get_primary_anomaly_map(self, img_batch, border_clean=True, **kwargs):
        '''
        In:
            img_batch: (batch_size, height, width) np.uint8
                or list of (height, width) np.uint8
        Out:
            amap_batch: (batch_size, height, width) torch.float32
        '''
        import torch
        from .impl import match_template_detector_impl
        with torch.no_grad():
            amap_batch = list()
            for i in range(len(img_batch)):
                img_t = torch.from_numpy(img_batch[i])
                if self._device != 'cpu':
                    img_t = img_t.cuda(self._device)
                func = match_template_detector_impl.get_saliency_map_pro
                amap = func(img_t, self._pattern_shape_s, self._stride_s, **kwargs)
                amap_batch.append(amap)
            # NxHxW
            amap_batch = torch.stack(amap_batch)
            return amap_batch

    def amap_normalize(self, amap_batch):
        import torch
        amap_uint8 = amap_batch.type(torch.uint8)
        amap_uint8 = amap_uint8.cpu().numpy()
        return amap_uint8

    def get_anomaly_map_from_gray(self, img_batch, **kwargs):
        '''
        In:
            img_batch: (batch_size, height, width) np.uint8
                or list of (height, width) np.uint8
        Out:
            amap: (batch_size, height, width) np.uint8
        '''
        amap_batch = self.get_primary_anomaly_map(img_batch, **kwargs)
        amap_uint8 = self.amap_normalize(amap_batch)
        return amap_uint8

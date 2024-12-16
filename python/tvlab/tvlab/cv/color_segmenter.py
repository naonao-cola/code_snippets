"""
Copyright (C) 2023 TuringVision

a vision tool  for configuring and performing color segmenting.
"""
from typing import Union
import numpy as np
from .geometry import Region
import cv2
import timeit
from .color_checker import rgb2hsi


__all__ = ['ColorSegmenter']


class ColorSegmenter:
    """
    The Color Segmenter tool analyzes a color image in order to produce a grey
    scale image consisting of light pixels against a dark background, where the
    light pixels correspond to features from the color input that fell into one
    or more desirable color ranges. The grey scale image a Color Segmenter tool
    produces represents only those features of the color image you are
    interested in, and can be further analyzed with another vision tool, such as
    a Blob tool.
    """

    def __init__(self, color_space=None):
        """
        :param color_space(str): one of ('RGB', 'HSI', 'HSV'), default 'RGB'
        """
        self._set_color_space(color_space)
        self.color_table = {}

    def _set_color_space(self, color_space):
        if color_space is None:
            color_space = 'RGB'
        color_space = color_space.upper()
        assert color_space in ['RGB', 'HSI', 'HSV']
        self.color_space = color_space

    def add(self, img, roi: Region, cls_name: str):
        """
        add a conference color configure based on the pixels in the roi of img,
        the configure will store into a dictionary with cls_name as its key
        """
        self.color_table[cls_name] = self._get_runtime_color(img, roi)

    def get_color(self, cls_name):
        return self.color_table[cls_name]

    def set_color(self, cls_name, color):
        """
        :param:
            color: list of dictionary corresponding to three channels in RGB or
            HSI color space. the dictionary has three keys:
                thr: nominal value, ranges [0, 255]
                tol_low: low tolerance, thr + tol_low >= 0
                tol_high: high tolerance, thr + tol_high <= 255
                eg: [
                        {'thr': 165.696, 'tol_low': -21.8182, 'tol_high': 21.8182},
                        {'thr': 55.1333, 'tol_low': -12.1454, 'tol_high': 12.1454},
                        {'thr': 19.4613, 'tol_low': -7.76617, 'tol_high': 7.76617}
                    ]
        """
        self.color_table[cls_name] = color

    def export(self, yaml_path):
        with open(yaml_path, 'wt', encoding='utf-8') as fp:
            import yaml
            config = {'color_table': self.color_table,
                      'color_space': self.color_space}
            yaml.dump(config, fp)

    def load(self, yaml_dir):
        with open(yaml_dir, 'rt', encoding='utf-8') as fp:
            import yaml
            config = yaml.load(fp, Loader=yaml.UnsafeLoader)
            self.color_table = config['color_table']
            self.color_space = config['color_space']

    def _get_runtime_color(self, img, roi: Region):
        assert len(img.shape) == 3 and img.shape[-1] == 3

        mask = roi.to_mask(img.shape[0:2]).astype(np.uint8)
        pixels = img[np.where(mask)]

        nominal_list = pixels.mean(0).tolist()
        if self.color_space != 'RGB':
            pixels = rgb2hsi(pixels.reshape(1, -1, 3)).reshape(-1, 3)
            nominal_list = rgb2hsi(nominal_list)
        std_list = pixels.std(0).tolist()

        tolerance_list = [[max(-m, min(-5, -s)), min(255 - m, max(5, s))]
                          for m, s in zip(nominal_list, std_list)]

        runtime_color = [{'thr': nominal_list[i],
                          'tol_low': tolerance_list[i][0],
                          'tol_high': tolerance_list[i][1]} for i in range(3)]

        return runtime_color

    def segment(self, img, cls_names: Union[None, list] = None):
        '''
        do color segmenting
        :parameter img: a cv2 image in RGB color space
        :parameter cls_names: None or list of str, specifying reference colors
            eg1: None, use all reference colors
            eg2: ['cls_name1', 'cls_name2', ...]
        :return: a dictionary
            key: color name
            value: grey image corresponding to color reference, the pixel value
                   0 stand for background
        '''
        assert len(img.shape) == 3 and img.shape[-1] == 3
        if self.color_space != 'RGB':
            img = rgb2hsi(img)

        if cls_names is None:
            color_table = self.color_table
        else:
            err_names = [name for name in cls_names if name not in self.color_table]
            assert len(cls_names) != 0 and len(err_names) == 0
            color_table = dict([(name, self.color_table[name]) for name in cls_names])

        segment_dict = dict()
        for name, color in color_table.items():
            segment_dict[name] = self._segment(img, color).astype(np.uint8) * 255

        return segment_dict

    def _segment(self, img, color):
        res = np.zeros(img.shape[0:2], dtype=np.bool)
        for i, v in enumerate(color):
            thr_low = np.round(v['thr'] + v['tol_low'])
            thr_high = np.round(v['thr'] + v['tol_high'])
            tmp = (img[:, :, i] >= thr_low) & (img[:, :, i] <= thr_high)
            if i == 0:
                res = tmp
            else:
                res &= tmp

        return res

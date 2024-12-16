"""
Copyright (C) 2023 TuringVision

a vision tool  for configuring and performing color checking.
"""
from typing import Union
import numpy as np
from .geometry import Region
import cv2


__all__ = ['ColorChecker', 'rgb2hsi']


def rgb2hsi(rgb: Union[list, np.ndarray]):
    """
    a color space converter, convert from RGB to HSI (Hue, Saturation, Intensity)
    :param rgb:
        eg1: [52.28099173553719, 74.63636363636364, 57.09090909090909], one pixel of [r, g, b] form
        eg2: cv2 form images in RGB color space
    :return:
        eg1: [93.71984996378384, 14.411288396569374, 61.33608815426998] in [h, s, i] form,
        value in each channel ranges [0, 255]
        eg2: cv2 from images in HSI color space, dtype is float64
    """
    if isinstance(rgb, list):
        r, g, b = rgb
    else:
        r, g, b = cv2.split(rgb)

    x = (2 * r - g - b) / 4
    y = (g - b) * np.sqrt(3) / 4
    s = np.sqrt(np.square(x) + np.square(y)) * np.sqrt(2)
    i = (r + g + b) / 3

    h = np.arctan2(y, x) * 128 / np.pi

    if isinstance(rgb, list):
        if g < b:
            h += 256
        return [h, s, i]
    else:
        h[g < b] += 256
        return cv2.merge((h, s, i))


class ColorChecker:
    """
    The Color Checker tool compares a region of color in a run-time image against
    a table of reference colors, and generates a set of scores to indicate how
    closely the area of the run-time image matches each known color. The higher
    the comparison score, the greater the similarity.
    """

    def __init__(self, color_space=None, omega=None):
        """
        :param color_space(str): one of ('RGB', 'HSI', 'HSV'), default 'RGB'
        :param omega(tuple): color weights
            eg: (0.5, 1.0, 1.0), they can't all be 0's
        """
        self._set_color_space(color_space)
        self.set_omega(omega)
        self.color_table = {}

    def _set_color_space(self, color_space):
        if color_space is None:
            color_space = 'RGB'
        color_space = color_space.upper()
        assert color_space in ['RGB', 'HSI', 'HSV']
        self.color_space = color_space

    def add(self, img, roi: Region, cls_name: str):
        '''
        add a conference color configure based on the pixels in the roi of img,
        the configure will store into a dictionary with cls_name as its key
        '''
        self.color_table[cls_name] = self._get_runtime_color(img, roi)

    def get_color(self, cls_name):
        return self.color_table[cls_name]

    def set_color(self, cls_name, color):
        """
        :param:
            color: list, [r, g, b] for RGB color space, or [h, s, i] for HSI
                value ranges [0, 255]
                eg: [52.63, 74.78, 57.21]
        """
        self.color_table[cls_name] = color

    def get_omega(self):
        return self.omega

    def set_omega(self, omega):
        """
        :param:
            omega: list, weights for each channel in RGB or HSI, value ranges
            [0, 1], do not set 0 at the same time, default [1, 1, 1]
        """
        if omega is None:
            omega = [1, 1, 1]
        else:
            assert sum(omega) > 1e-10 and len(omega) == 3
        self.omega = omega

    def export(self, yaml_path):
        with open(yaml_path, 'wt', encoding='utf-8') as fp:
            import yaml
            config = {'color_table': self.color_table,
                      'omega': self.omega,
                      'color_space': self.color_space}
            yaml.dump(config, fp)

    def load(self, yaml_dir):
        with open(yaml_dir, 'rt', encoding='utf-8') as fp:
            import yaml
            config = yaml.load(fp, Loader=yaml.UnsafeLoader)
            self.color_table = config['color_table']
            self.omega = config['omega']
            self.color_space = config['color_space']

    def check(self, img, roi: Region, cls_names: Union[None, list] = None):
        '''
        do color checking in the specified roi of the image
        :parameter cls_names: None or list of str, specifying reference colors
            eg1: None, use all reference colors
            eg2: ['cls_name1', 'cls_name2', ...]
        :return:
            distance_list: list of color distance dictionay, each element
            contain reference color name and color distance, the list is sorted
            by color distance in descending order.
            the value of color distance, ranges [0, 1], the higher the closer.
                eg: [{'distance': 0.9717694064351142, 'name': 'Lemon'},
                     {'distance': 0.8684965741387338, 'name': 'Orange'}, ...]

            confidence: the confidence of the differentiation between the top 2.
        '''
        self.runtime_color = self._get_runtime_color(img, roi)
        distance_list = self._calc_distance_list(cls_names)
        confidence = self._calc_confidence()

        return distance_list, confidence

    def _get_runtime_color(self, img, roi: Region):
        assert len(img.shape) == 3 and img.shape[-1] == 3

        mask = roi.to_mask(img.shape[0:2]).astype(np.uint8)

        runtime_color = list(cv2.mean(img, mask)[:3])
        if self.color_space != 'RGB':
            runtime_color = rgb2hsi(runtime_color)

        return runtime_color

    def _calc_distance_list(self, cls_names):
        if cls_names is None:
            color_table = self.color_table
        else:
            err_names = [name for name in cls_names if name not in self.color_table]
            assert len(cls_names) != 0 and len(err_names) == 0
            color_table = dict([(name, self.color_table[name]) for name in cls_names])

        distance_list = []
        for name, reference_color in color_table.items():
            distance_list.append({'name': name, 'distance': self._calc_distance(reference_color)})
        distance_list = sorted(distance_list, key=lambda x: x['distance'], reverse=True)

        self.distance_list = distance_list

        return distance_list

    def _calc_confidence(self):
        dist_list = [x['distance'] for x in self.distance_list]
        confidence = 1
        if len(dist_list) > 1:
            confidence = (dist_list[0] - dist_list[1]) / (dist_list[0] + dist_list[1])
        self.confidence = confidence

        return confidence

    def _calc_distance(self, reference_color):
        ssd_list = [np.square(self.runtime_color[i] / 255.0 - reference_color[i] / 255.0) * self.omega[i] for i in range(3)]
        cd = 1 - np.sqrt(np.sum(ssd_list) / sum(self.omega))

        return cd

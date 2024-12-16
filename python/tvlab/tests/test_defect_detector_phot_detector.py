from tvlab import *
import unittest
import cv2
import numpy as np


class TestPhotDefectDetector(unittest.TestCase):
    def test_get_anomaly_map_from_gray(self):
        phot_detector = PhotDefectDetector()
        img = cv2.imread(osp.normpath('./data/010.jpg'))
        gray_img = rgb2gray_cpu(img)
        re = phot_detector.get_anomaly_map_from_gray([gray_img])
        self.assertEqual(re.shape, (1, 1024, 1281))

    def test_get_phase_only_img(self):
        phot_detector = PhotDefectDetector()
        img = cv2.imread(osp.normpath('./data/010.jpg'))
        gray_img = rgb2gray_cpu(img)
        re = phot_detector.get_phase_only_img([gray_img])
        self.assertEqual(str(re.shape), 'torch.Size([1, 1024, 1281])')

    def test_amap_normalize(self):
        phot_detector = PhotDefectDetector()
        img = cv2.imread(osp.normpath('./data/010.jpg'))
        gray_img = rgb2gray_cpu(img)
        re = phot_detector.get_phase_only_img([gray_img])
        re_a = phot_detector.amap_border_clean(re)
        re_n = phot_detector.amap_normalize(re_a)
        self.assertEqual(re_n.shape, (1, 1024, 1281))
        self.assertEqual(str(type(re_n)), "<class 'numpy.ndarray'>")


class TestTiledPhotDefectDetector(unittest.TestCase):
    def test_get_tile_shape(self):
        tile_phot = TiledPhotDefectDetector()
        img = cv2.imread(osp.normpath('./data/010.jpg'))
        img_gray = rgb2gray_cpu(img)
        re = tile_phot.get_tile_shape(img_gray)
        self.assertTrue(isinstance(re, tuple), True)

    def test_get_phase_only_img(self):
        tile_phot = TiledPhotDefectDetector()
        img = cv2.imread(osp.normpath('./data/010.jpg'))
        img_gray = rgb2gray_cpu(img)
        re_phase = tile_phot.get_phase_only_img([img_gray])
        flg = True
        if re_phase is None:
            flg = False
        self.assertEqual(str(re_phase.shape), 'torch.Size([1, 1024, 1280])')
        self.assertTrue(flg, True)


if __name__ == '__main__':
    unittest.main()

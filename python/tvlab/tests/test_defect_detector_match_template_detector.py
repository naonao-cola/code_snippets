from tvlab import *
import unittest
import cv2
import numpy as np


IN_SIZE = 384


class TestMatchTemplateDetector(unittest.TestCase):
    def test_get_anomaly_map_from_gray(self):
        mt_detector = MatchTemplateDetector(device='cpu')
        img = cv2.imread(osp.normpath('./data/010.jpg'))
        img = cv2.resize(img, (IN_SIZE, IN_SIZE))
        img_gray = rgb2gray_cpu(img)
        re = mt_detector.get_anomaly_map_from_gray([img_gray])
        self.assertEqual(re.shape, (1, IN_SIZE, IN_SIZE))


class TestFastMatchTemplateDetector(unittest.TestCase):
    def test_get_anomaly_map_from_gray(self):
        fast_mt_detector = FastMatchTemplateDetector(
            pattern_shape_s=[(IN_SIZE-32, 8), (8, IN_SIZE-32)],
            stride_s=[(32, 8), (8, 32)],
            device='cpu')
        img = cv2.imread(osp.normpath('./data/010.jpg'))
        img = cv2.resize(img, (IN_SIZE, IN_SIZE))
        img_gray = rgb2gray_cpu(img)
        re_fast = fast_mt_detector.get_anomaly_map_from_gray([img_gray])
        self.assertEqual(re_fast.shape, (1, IN_SIZE, IN_SIZE))


if __name__ == '__main__':
    unittest.main()

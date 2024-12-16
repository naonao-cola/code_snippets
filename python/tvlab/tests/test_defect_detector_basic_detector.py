from tvlab import *
import unittest
import cv2
import numpy as np


class TestBasicDefectDetector(unittest.TestCase):
    def test_rgb2gray_cpu(self):
        color_img = cv2.imread(osp.normpath('./data/010.jpg'))
        gray_cpu = rgb2gray_cpu(color_img)
        flg = False
        if len(color_img.shape) == 3 and len(gray_cpu.shape) == 2:
            flg = True
        self.assertEqual(flg, True)

    def test_rgb2gray_gpu(self):
        color_img = cv2.imread(osp.normpath('./data/010.jpg'))
        gray_gpu = rgb2gray_gpu(color_img)
        flg = False
        if len(color_img.shape) == 3 and len(gray_gpu.shape) == 2:
            flg = True
        self.assertEqual(flg, True)

    def test_get_bboxes_from_single_binary(self):
        bd = BasicDefectDetector
        img = cv2.imread(osp.normpath('./data/010.jpg'))
        img_gray = rgb2gray_cpu(img)
        re = bd.get_bboxes_from_single_binary(img_gray)
        flg = True
        if re is None:
            flg = False
        self.assertTrue(isinstance(re, list), True)
        self.assertTrue(flg, True)

    def test_get_bboxes_from_single_amap(self):
        bd = BasicDefectDetector
        img = cv2.imread(osp.normpath('./data/010.jpg'))
        img_gray = rgb2gray_cpu(img)
        re = bd.get_bboxes_from_single_amap(img_gray)
        flg = True
        if re is None:
            flg = False
        self.assertTrue(isinstance(re, list), True)
        self.assertTrue(flg, True)


if __name__ == '__main__':
    unittest.main()

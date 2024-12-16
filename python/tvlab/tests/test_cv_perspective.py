from tvlab import *
import unittest
import cv2
import timeit
import torch
import numpy as np


class TestPerspective(unittest.TestCase):
    def test_init(self):
        pe = Perspective(((54, 3), (248, 78), (5, 140), (208, 200)), (200, 200))
        re_m = pe.m
        re_wh = pe.dst_wh
        re_invert_m = pe.invert_m
        self.assertEqual(round(re_m[0][0], 2), 0.78)
        self.assertEqual(re_wh, (200, 200))
        self.assertEqual(round(re_invert_m[0][0], 2), 1.13)

    def test_tfm_xy(self):
        pe = Perspective(((54, 3), (248, 78), (5, 140), (208, 200)), (200, 200))
        re1 = pe.tfm_xy((54, 3), True)
        re2 = pe.tfm_xy((54, 3), False)
        re1_t = 110.6
        re2_t = 0.0
        self.assertEqual(round(re1[0], 1), re1_t)
        self.assertEqual(round(re2[0], 1), re2_t)

    def test_tfm_pts(self):
        pe = Perspective(((54, 3), (248, 78), (5, 140), (208, 200)), (200, 200))
        pts = [(54, 3), (248, 78), (5, 140), (208, 200)]
        re1 = pe.tfm_pts(pts, True)
        re2 = pe.tfm_pts(pts, False)
        re1_t = [[110, 27], [273, 139], [25, 100], [215, 202]]
        re2_t = [[0, 0], [200, 0], [0, 200], [200, 199]]
        self.assertEqual(re1, re1_t)
        self.assertEqual(re2, re2_t)

    def test_tfm_bbox_label(self):
        pe = Perspective(((54, 3), (248, 78), (5, 140), (208, 200)), (200, 200))
        bbox_label = {'labels': ['A'], 'bboxes': [[54, 12, 200, 170]]}
        bb = BBoxLabel(bbox_label)
        re = pe.tfm_bbox_label(bb)
        self.assertEqual(re['labels'], ['A'])
        self.assertEqual(len(re['bboxes']), 1)

    def test_tfm_polygon_label(self):
        pe = Perspective(((54, 3), (248, 78), (5, 140), (208, 200)), (200, 200))
        polygon_label = {'labels': ['A'], 'polygons': [[54, 3, 248, 78, 5, 140, 208, 200]]}
        polygonlable = PolygonLabel(polygon_label)
        re = pe.tfm_polygon_label(polygonlable)
        self.assertEqual(re['labels'], ['A'])
        self.assertEqual(re['polygons'], [[0, 0, 200, 0, 0, 200, 200, 199]])

    def test_tfm_img(self):
        pe = Perspective(((54, 3), (248, 78), (5, 140), (208, 200)), (200, 200))
        img = cv2.imread(osp.normpath('./data/006.jpg'))
        re = pe.tfm_img(img)
        self.assertEqual(re.shape, (200, 200, 3))


if __name__ == '__main__':
    unittest.main()

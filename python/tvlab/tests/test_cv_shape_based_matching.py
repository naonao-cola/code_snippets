from tvlab import *
import unittest
import cv2
import timeit
import os
import os.path as osp
import numpy as np


class TestShapeBasedMatching(unittest.TestCase):
    def test_init(self):
        sbm_first = ShapeBasedMatching(features_num=256, T=(2, 3), weak_threshold=31.0, strong_threshold=61.0,
                                       gaussion_kenel=8)
        self.assertEqual(sbm_first.T, (2, 3))
        self.assertEqual(sbm_first.max_align, 12)
        sbm_second = ShapeBasedMatching(features_num=256, T=(1, 1), weak_threshold=31.0, strong_threshold=61.0,
                                        gaussion_kenel=8)
        self.assertEqual(sbm_second.T, (1, 1))
        self.assertEqual(sbm_second.max_align, 8)

    def test_add(self):
        sbm = ShapeBasedMatching(64, T=(4,))
        path = osp.normpath('./data/template/')
        train_img = cv2.imread(osp.normpath('./data/007.jpg'))
        train_img = cv2.resize(train_img, (int(train_img.shape[0] / 4), int(train_img.shape[1] / 4)))
        c_id = 'camera'
        sbm.add(train_img, angle_range=(0, 360, 20.0), scale_range=(0.97, 1.01, 0.01), class_id='camera')
        sbm.save(path)
        flg = False
        for filename in os.listdir(path):
            if filename[:-5] == c_id:
                flg = True
                break
        self.assertTrue(flg, True)

    def test_show(self):
        sbm = ShapeBasedMatching(64, T=(4,))
        train_img = cv2.imread(osp.normpath('./data/007.jpg'))
        sbm.load(osp.normpath('./data/template/'), 'camera')
        to_show = sbm.show(train_img, class_id='camera')
        re = to_show.shape
        self.assertEqual(re, (2144, 2792, 3))

    def test_find(self):
        sbm = ShapeBasedMatching(64, T=(4,))
        sbm.load(osp.normpath('./data/template/'), 'camera')
        test_img = cv2.imread(osp.normpath('./data/009.jpg'))
        test_img = cv2.resize(test_img, (int(test_img.shape[0] / 4), int(test_img.shape[1] / 4)))
        matches = sbm.find(test_img, class_ids='camera', topk=1, subpixel=True, debug=True)
        re = round(matches['camera'][0][0])
        flg = False
        if re > 100 and re < 600:
            flg = True
        self.assertTrue(flg, True)

    def test_draw_match_rect(self):
        sbm = ShapeBasedMatching(64, T=(4,))
        test_img = cv2.imread(osp.normpath('./data/009.jpg'))
        test_img = cv2.resize(test_img, (int(test_img.shape[0] / 4), int(test_img.shape[1] / 4)))
        matches = [(192.22337341308594, 325.9695739746094, 272.2419738769531, 617.5488891601562, 0.0853489339351654,
                    1.0008896589279175, 100.0)]
        re = sbm.draw_match_rect(test_img, matches)
        self.assertEqual(re.shape, (648, 486, 3))


if __name__ == '__main__':
    suite = unittest.TestSuite()

    tests = [
        TestShapeBasedMatching("test_init"),
        TestShapeBasedMatching("test_add"),
        TestShapeBasedMatching("test_show"),
        TestShapeBasedMatching("test_find"),
        TestShapeBasedMatching("test_draw_match_rect")
    ]
    suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

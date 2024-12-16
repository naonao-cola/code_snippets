import numpy as np
from tvlab import Line
import unittest
import cv2
from tvlab.cv import CaliperTool
import timeit
import os.path as osp


class TestCaliperTool(unittest.TestCase):
    RUNNING_TIME_BUFFER = 5

    def test_find_point(self):
        # fp = CaliperTool(debug=False, threshold=30)
        img = cv2.imread(osp.normpath('./data/004.jpg'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        start = timeit.default_timer()
        re = CaliperTool.find_point(img, (106, 108, 100, 5, 90.0),
                                    filter_size=2,
                                    threshold=60,
                                    polarity=CaliperTool.LIGHTTODARK,
                                    debug=False,
                                    sub_pix=True
                                    )
        end = timeit.default_timer()
        time = end - start
        flg = False
        if time < 0.1 * self.RUNNING_TIME_BUFFER:
            flg = True
        self.assertEqual(re, (106.5, 103.0))
        self.assertTrue(flg, True)

    def test_find_circle(self):
        img = cv2.imread(osp.normpath('./data/000.jpg'))
        start = timeit.default_timer()
        re = CaliperTool.find_circle(img, (250, 250, 250, 0, 360),
                                     num_caliper=101,
                                     side_x_length=101,
                                     side_y_length=11,
                                     filter_num_point=1,
                                     filter_size=2,
                                     threshold=61,
                                     polarity=CaliperTool.LIGHTTODARK,
                                     direction=1,
                                     debug=False,
                                     sub_pix=False,
                                     return_ext_points=True)
        end = timeit.default_timer()
        time = end - start
        flg = False
        re2 = tuple(np.around(re[0], 1))
        if time < 0.1 * self.RUNNING_TIME_BUFFER:
            flg = True
        self.assertEqual(re2, (249.6, 249.5, 250.1))
        self.assertTrue(flg, True)

    def test_find_line(self):
        img = cv2.imread(osp.normpath('./data/000.jpg'))
        start = timeit.default_timer()
        line = Line((171, 101), (321, 101))
        re = CaliperTool.find_line(img, line,
                                   side_x_length=50,
                                   num_caliper=15,
                                   polarity=CaliperTool.LIGHTTODARK,
                                   direction=0,
                                   sub_pix=False,
                                   debug=False,
                                   return_ext_points=True)
        end = timeit.default_timer()
        time = end - start
        flg = False
        if time < 0.1 * self.RUNNING_TIME_BUFFER:
            flg = True
        trues = ([0, 93.0], [500, 93.0])
        preds = re[0].to_xy()
        self.assertAlmostEqual(preds[0][0], trues[0][0], delta=1e-4)
        self.assertAlmostEqual(preds[0][1], trues[0][1], delta=1e-4)
        self.assertAlmostEqual(preds[1][0], trues[1][0], delta=1e-4)
        self.assertAlmostEqual(preds[1][1], trues[1][1], delta=1e-4)
        self.assertTrue(flg, True)


if __name__ == '__main__':
    unittest.main()

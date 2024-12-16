from tvlab import Xld
import unittest
import cv2
import timeit
import os
import os.path as osp


class TestXld(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestXld, self).__init__(*args, **kwargs)
        self.gray_img = cv2.imread(osp.normpath('./data/cv/xld_test.png'), 0)

    def test_init(self):
        '''
            for the xld_test.png
            halcon ： the number of contours is 10
            number of points : contour1:96   contour2:101   contour3:160  contour4:193  contour5:121
                               contour6:262  contour7:154  contour8:146  contour9:22  contour10:31
            run time about 'edges_sub_pix' is  0.0080604
            ours code run time is 0.004126026993617415
        '''
        sigma = 1.0
        th_l = 10.0
        th_h = 35.0
        print("gray_img shape:", self.gray_img.shape)
        start = timeit.default_timer()
        xld_res = Xld.from_img(self.gray_img, sigma, th_l, th_h)
        end = timeit.default_timer()
        time = end - start
        self.assertTrue(time < 0.10, True)
        self.assertAlmostEqual(len(xld_res[0]), 96, delta=30)
        self.assertAlmostEqual(len(xld_res[1]), 101, delta=30)
        self.assertAlmostEqual(len(xld_res[2]), 160, delta=30)
        self.assertAlmostEqual(len(xld_res[3]), 193, delta=30)
        self.assertAlmostEqual(len(xld_res[4]), 121, delta=30)
        self.assertTrue(len(xld_res), 10)

        xld_res2 = Xld.from_img(self.gray_img, sigma, th_l, th_h)
        xld_res3 = Xld.from_img(self.gray_img, sigma, th_l, th_h)
        xld_res2.extend(xld_res3)
        self.assertTrue(len(xld_res2) == 20, True)

    def test_filter(self):
        xld_res = Xld.from_img(self.gray_img)
        start = timeit.default_timer()
        filter_result = xld_res.filter(lambda item: len(item) > 40)
        end = timeit.default_timer()
        time = end - start
        self.assertTrue(time < 0.10, True)
        self.assertTrue(len(filter_result) == 8, True)

    def test_filter_by_open(self):
        start = timeit.default_timer()
        xld_res = Xld.from_img(self.gray_img)
        filter_result = xld_res.filter_by_open()
        end = timeit.default_timer()
        time = end - start
        self.assertTrue(time < 0.10, True)
        self.assertTrue(len(filter_result) == 8, True)

    def test_filter_by_closed(self):
        xld_res = Xld.from_img(self.gray_img)
        start = timeit.default_timer()
        filter_result = xld_res.filter_by_closed()
        end = timeit.default_timer()
        time = end - start
        self.assertTrue(time < 0.10, True)
        self.assertTrue(len(filter_result) == 2, True)

    def test_smooth(self):
        xld_res = Xld.from_img(self.gray_img)
        smooth_result = xld_res.smooth()
        self.assertTrue(len(smooth_result) == 10, True)
        self.assertTrue(len(xld_res[0]) == len(smooth_result[0]), True)
        self.assertTrue(len(xld_res[9]) == len(smooth_result[9]), True)

    def test_distance(self):
        xld_res = Xld.from_img(self.gray_img)
        smooth_res = xld_res.smooth()
        calc_distance_res = xld_res.distance(smooth_res)
        self.assertTrue(len(calc_distance_res) == 10, True)
        self.assertIsInstance(calc_distance_res, list, True)


if __name__ == '__main__':
    suite = unittest.TestSuite()

    tests = [
        TestXld("test_init"),
        TestXld("test_filter"),
        TestXld('test_filter_by_closed'),
        TestXld('test_filter_by_open'),
        TestXld("test_smooth"),
        TestXld("test_distance"),

    ]
    suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

from tvlab import ColorSegmenter, Region, rgb2hsi
import unittest
import cv2
import timeit
import numpy as np
import os


class TestColorSegmenter(unittest.TestCase):
    root_dir = os.path.normpath('./data/cv/')
    img_dir = os.path.join(root_dir, 'blister.png')
    yaml_dir = os.path.join(root_dir, 'cs.yaml')
    img_ref_dir = os.path.join(root_dir, 'blister.bmp')

    def test_init(self):
        img = cv2.imread(self.img_dir, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        cs = ColorSegmenter(color_space='RGB')
        self.assertEqual(cs.color_space, 'RGB')

    def test_add(self):
        img = cv2.imread(self.img_dir, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        cs = ColorSegmenter(color_space='hsi')

        roi = [150, 200, 1 - 1, 10 - 1]
        roi_rect = [roi[0],
                    roi[1],
                    roi[0] + roi[2],
                    roi[1] + roi[3]]
        bbox = Region.from_bbox(roi_rect)
        color_true = [40.29059658678971, 91.40752157235202, 117.43333333333332]
        cs.add(img, roi=bbox, cls_name='Lemon')
        color_cal = [c['thr'] for c in cs.color_table['Lemon']]
        self.assertAlmostEqual(color_cal, color_true, delta=1e-3)

        cs = ColorSegmenter(color_space='RGB')
        roi = [150, 200, 1 - 1, 10 - 1]
        roi_rect = [roi[0],
                    roi[1],
                    roi[0] + roi[2],
                    roi[1] + roi[3]]
        bbox = Region.from_bbox(roi_rect)
        color_true = [164.8, 156.1, 31.4]
        cs.add(img, roi=bbox, cls_name='Lemon')
        color_cal = [c['thr'] for c in cs.color_table['Lemon']]
        self.assertAlmostEqual(color_cal, color_true, delta=1e-3)

        cs.export(self.yaml_dir)
        flg = os.path.exists(self.yaml_dir)
        self.assertTrue(flg, True)

        cs.color_table = {}
        cs.load(self.yaml_dir)
        color_cal = [c['thr'] for c in cs.color_table['Lemon']]
        self.assertAlmostEqual(color_cal, color_true, delta=1e-3)

    def test_set_color(self):
        img = cv2.imread(self.img_dir, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        cs = ColorSegmenter(color_space='RGB')
        color_true = [
            {'thr': 165.696, 'tol_low': -21.8182, 'tol_high': 21.8182},
            {'thr': 55.1333, 'tol_low': -12.1454, 'tol_high': 12.1454},
            {'thr': 19.4613, 'tol_low': -7.76617, 'tol_high': 7.76617}
        ]
        cs.set_color('TopLeft_Rgn', color_true)
        self.assertEqual(cs.color_table['TopLeft_Rgn'], color_true)

    def test_match_rgb(self):
        img = cv2.imread(self.img_dir, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_segment_true = cv2.imread(self.img_ref_dir, cv2.IMREAD_GRAYSCALE)

        # R/G/B
        reference_color = dict()
        reference_color['TopLeft_Rgn'] = [
            {'thr': 165.696, 'tol_low': -21.8182, 'tol_high': 21.8182},
            {'thr': 55.1333, 'tol_low': -12.1454, 'tol_high': 12.1454},
            {'thr': 19.4613, 'tol_low': -7.76617, 'tol_high': 7.76617}
        ]
        reference_color['TopRight_Rgn'] = [
            {'thr': 113.128, 'tol_low': -18.7367, 'tol_high': 18.7367},
            {'thr': 34.375, 'tol_low': -7.38531, 'tol_high': 7.38531},
            {'thr': 10.6278, 'tol_low': -4.0433, 'tol_high': 4.0433}
        ]
        reference_color['LowerLeft_Rgn'] = [
            {'thr': 153.135, 'tol_low': -20.2838, 'tol_high': 20.2838},
            {'thr': 53.1875, 'tol_low': -12.821, 'tol_high': 12.821},
            {'thr': 19.4803, 'tol_low': -8.55653, 'tol_high': 8.55653}
        ]
        reference_color['LowerRight_Rgn'] = [
            {'thr': 145.291, 'tol_low': -28.1713, 'tol_high': 28.1713},
            {'thr': 47.8421, 'tol_low': -12.7685, 'tol_high': 12.7685},
            {'thr': 16.4241, 'tol_low': -7.71199, 'tol_high': 7.71199}
        ]
        reference_color['Center_Rgn'] = [
            {'thr': 219.845, 'tol_low': -22.8118, 'tol_high': 22.8118},
            {'thr': 73.7173, 'tol_low': -12.0544, 'tol_high': 12.0544},
            {'thr': 25.9732, 'tol_low': -6.26498, 'tol_high': 6.26498}
        ]

        cs = ColorSegmenter(color_space='RGB')
        for k, v in reference_color.items():
            cs.set_color(k, v)

        start = timeit.default_timer()

        img_segment_dict = cs.segment(img)
        img_segment = np.zeros(img.shape[0:2], dtype=np.uint8)
        for v in img_segment_dict.values():
            img_segment |= v

        end = timeit.default_timer()

        time = end - start
        flg = False
        if time < 0.02:
            flg = True

        self.assertAlmostEqual(np.mean(img_segment),
                               np.mean(img_segment_true), delta=1e-3)

        self.assertTrue(flg, True)


if __name__ == '__main__':
    suite = unittest.TestSuite()

    tests = [
        TestColorSegmenter("test_init"),
        TestColorSegmenter("test_add"),
        TestColorSegmenter("test_set_color"),
        TestColorSegmenter("test_match_rgb"),
    ]
    suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

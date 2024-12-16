from tvlab import ColorChecker, Region, rgb2hsi
import unittest
import cv2
import timeit
import os.path as osp
import os


class TestColorChecker(unittest.TestCase):
    root_dir = osp.normpath('./data/cv/')
    img_dir = os.path.join(root_dir, 'JellOBoxes.png')
    yaml_dir = os.path.join(root_dir, 'cc.yaml')

    def test_init(self):
        img = cv2.imread(self.img_dir, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        cc = ColorChecker(color_space='RGB')
        self.assertEqual(cc.omega, [1, 1, 1])

    def test_add(self):
        img = cv2.imread(self.img_dir, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        cc = ColorChecker(color_space='hsi')
        bbox = Region.from_bbox((10, 10, 20, 20))
        color_true = [93.71984996378384, 14.411288396569374, 61.33608815426998]
        cc.add(img, roi=bbox, cls_name='Lemon')
        self.assertAlmostEqual(cc.color_table['Lemon'], color_true, delta=1e-3)

        cc = ColorChecker(color_space='RGB')
        bbox = Region.from_bbox((10, 10, 20, 20))
        color_true = [52.28099173553719, 74.63636363636364, 57.09090909090909]
        cc.add(img, roi=bbox, cls_name='Lemon')
        self.assertAlmostEqual(cc.color_table['Lemon'], color_true, delta=1e-3)

        cc.export(self.yaml_dir)
        flg = os.path.exists(self.yaml_dir)
        self.assertTrue(flg, True)

        cc.color_table = {}
        cc.load(self.yaml_dir)
        self.assertAlmostEqual(cc.color_table['Lemon'], color_true, delta=1e-3)

    def test_set_color(self):
        img = cv2.imread(self.img_dir, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        cc = ColorChecker(color_space='RGB')
        color_true = [52.63, 74.78, 57.21]
        cc.set_color('Lemon', color_true)
        self.assertEqual(cc.color_table['Lemon'], color_true)

    def test_set_omega(self):
        img = cv2.imread(self.img_dir, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        cc = ColorChecker(color_space='RGB')
        omega_true = [0.5, 1, 1]
        cc.set_omega(omega_true)
        self.assertEqual(cc.omega, omega_true)

    def test_match_rgb(self):
        img = cv2.imread(self.img_dir, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        reference_color = dict()
        reference_color['Lemon'] = [134.704, 95.0957, 19.6649]
        reference_color['Orange'] = [119.808, 51.4508, 15.2382]
        reference_color['Lime'] = [54.0335, 66.1891, 23.8969]
        reference_color['Grape'] = [35.103, 16.8681, 16.0118]
        reference_color['Black Cherry'] = [47.1794, 16.9336, 10.7865]

        # test 1 smaller region
        # roi = [200, 350, 2 - 1, 2 - 1]  # [left, top, width, height]
        # roi_rect = [roi[0],
        #             roi[1],
        #             roi[0] + roi[2],
        #             roi[1] + roi[3]]
        # omega = [1, 1, 1]
        # dist_true_list = [0.981, 0.913, 0.82, 0.731, 0.752]
        # confidence_true = 0.0362

        # test 2 larger region
        # roi = [200, 350, 50 - 1, 50 - 1]
        # roi_rect = [roi[0],
        #             roi[1],
        #             roi[0] + roi[2],
        #             roi[1] + roi[3]]
        # omega = [1, 1, 1]
        # dist_true_list = [0.982, 0.889, 0.812, 0.715, 0.734]
        # confidence_true = 0.0499

        # test 3 float parameter
        roi = [247.96, 332.15 - 1, 37.4995 - 1, 31.6778 - 1]
        roi_rect = [roi[0],
                    roi[1],
                    roi[0] + roi[2],
                    roi[1] + roi[3]]
        omega = [0.5, 0.8, 0.1]
        dist_true_list = [0.972, 0.869, 0.814, 0.687, 0.705]
        confidence_true = 0.0556

        cc = ColorChecker(color_space='RGB')
        cc.set_omega(omega)
        cc.set_color('Lemon', reference_color['Lemon'])
        cc.set_color('Orange', reference_color['Orange'])
        cc.set_color('Lime', reference_color['Lime'])
        cc.set_color('Grape', reference_color['Grape'])
        cc.set_color('Black Cherry', reference_color['Black Cherry'])

        start = timeit.default_timer()

        bbox = Region.from_bbox(tuple(roi_rect))

        distance_list, confidence = cc.check(img, roi=bbox)
        dist_list = [x['distance'] for x in distance_list]

        end = timeit.default_timer()

        time = end - start
        flg = False
        if time < 0.003:
            flg = True

        for dist, dist_true in zip(dist_list, sorted(dist_true_list, reverse=True)):
            self.assertAlmostEqual(dist, dist_true, delta=1e-3)
        self.assertAlmostEqual(confidence, confidence_true, delta=1e-3)

        self.assertTrue(flg, True)

    def test_match_hsi(self):
        img = cv2.imread(self.img_dir, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        reference_color = dict()
        reference_color['Lemon'] = [134.704, 95.0957, 19.6649]
        reference_color['Orange'] = [119.808, 51.4508, 15.2382]
        reference_color['Lime'] = [54.0335, 66.1891, 23.8969]
        reference_color['Grape'] = [35.103, 16.8681, 16.0118]
        reference_color['Black Cherry'] = [47.1794, 16.9336, 10.7865]

        # test 1 smaller region
        # roi = [200, 350, 2 - 1, 2 - 1]
        # roi_rect = [roi[0],
        #             roi[1],
        #             roi[0] + roi[2],
        #             roi[1] + roi[3]]
        # omega = [1, 1, 1]
        # dist_true_list = [0.989, 0.951, 0.866, 0.811, 0.834]
        # confidence_true = 0.0195

        # test 2 larger region
        roi = [200, 350, 100 - 1, 80 - 1]
        roi_rect = [roi[0],
                    roi[1],
                    roi[0] + roi[2],
                    roi[1] + roi[3]]
        omega = [1, 1, 1]
        dist_true_list = [0.981, 0.937, 0.869, 0.807, 0.828]
        confidence_true = 0.0233

        cc = ColorChecker(color_space='HSI')
        cc.set_omega(omega)
        cc.set_color('Lemon', rgb2hsi(reference_color['Lemon']))
        cc.set_color('Orange', rgb2hsi(reference_color['Orange']))
        cc.set_color('Lime', rgb2hsi(reference_color['Lime']))
        cc.set_color('Grape', rgb2hsi(reference_color['Grape']))
        cc.set_color('Black Cherry', rgb2hsi(reference_color['Black Cherry']))

        start = timeit.default_timer()

        bbox = Region.from_bbox(tuple(roi_rect))

        distance_list, confidence = cc.check(img, roi=bbox)
        dist_list = [x['distance'] for x in distance_list]

        end = timeit.default_timer()

        time = end - start
        flg = False
        if time < 0.003:
            flg = True

        for dist, dist_true in zip(dist_list, sorted(dist_true_list, reverse=True)):
            self.assertAlmostEqual(dist, dist_true, delta=2e-3)
        self.assertAlmostEqual(confidence, confidence_true, delta=1e-3)

        self.assertTrue(flg, True)


if __name__ == '__main__':
    suite = unittest.TestSuite()

    tests = [
        TestColorChecker("test_init"),
        TestColorChecker("test_add"),
        TestColorChecker("test_set_color"),
        TestColorChecker("test_set_omega"),
        TestColorChecker("test_match_rgb"),
        TestColorChecker("test_match_hsi")
    ]
    suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

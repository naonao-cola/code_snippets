from tvlab import *
import unittest
import cv2
import numpy as np
import os


class TestMahalanobisDetector(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestMahalanobisDetector, self).__init__(*args, **kwargs)
        self.model_path = osp.normpath('./models/defect_detection/mahan_detector_model.pth')
        os.makedirs(osp.dirname(self.model_path), exist_ok=True)

    def test_init(self):
        mahan_detector = MahalanobisDetector()
        self.assertEqual(mahan_detector.device, 'cuda')
        self.assertEqual(mahan_detector._border_coef, 1.618)

    def test_train(self):
        mahan_detector = MahalanobisDetector()
        ibll = ImageLabelList.from_folder(osp.normpath('./data/defect_data/defect_image/'))
        print("mahan model train...")
        vp_err = None
        vp_err = mahan_detector.train(ibll, [], bs=8, workers=8, border=0,
                                      basemodel='densenet161', fv_rf=16,
                                      percent_cb=None, debug=True)
        print("mahan model save...")
        mahan_detector.save(self.model_path)
        c_id = osp.basename(self.model_path)[:-4]
        path = osp.dirname(self.model_path)
        flg_save = False
        for filename in os.listdir(path):
            if filename[:-4] == c_id:
                flg_save = True
        flg = True
        if vp_err is None:
            flg = False
        self.assertTrue(flg, True)
        self.assertTrue(flg_save, True)

    def test_get_anomaly_map_from_rgb(self):
        mahan_detector = MahalanobisDetector()
        mahan_detector.load(self.model_path)
        img = cv2.imread(osp.normpath('./data/defect_data/test/010.jpg'))
        amap = mahan_detector.get_anomaly_map_from_rgb([img])
        self.assertEqual(str(type(amap)), "<class 'numpy.ndarray'>")
        self.assertEqual(amap.shape, (1, 64, 80))

    def test_get_primary_anomaly_map(self):
        mahan_detector = MahalanobisDetector()
        mahan_detector.load(self.model_path)
        img = cv2.imread(osp.normpath('./data/defect_data/test/010.jpg'))
        amap = mahan_detector.get_primary_anomaly_map([img])
        self.assertEqual(str(type(amap)), "<class 'torch.Tensor'>")

    def test_amap_border_clean(self):
        mahan_detector = MahalanobisDetector()
        mahan_detector.load(self.model_path)
        img = cv2.imread(osp.normpath('./data/defect_data/test/010.jpg'))
        amap = mahan_detector.get_primary_anomaly_map([img])
        amap = mahan_detector.amap_border_clean(amap)
        self.assertEqual(str(type(amap)), "<class 'torch.Tensor'>")

    def test_amap_normalize(self):
        mahan_detector = MahalanobisDetector()
        mahan_detector.load(self.model_path)
        img = cv2.imread(osp.normpath('./data/defect_data/test/010.jpg'))
        amap = mahan_detector.get_primary_anomaly_map([img])
        amap = mahan_detector.amap_border_clean(amap)
        amap = mahan_detector.amap_normalize(amap)
        self.assertEqual(str(type(amap)), "<class 'numpy.ndarray'>")


if __name__ == '__main__':
    suite = unittest.TestSuite()

    tests = [
        TestMahalanobisDetector("test_init"),
        TestMahalanobisDetector("test_train"),
        TestMahalanobisDetector("test_get_anomaly_map_from_rgb"),
        TestMahalanobisDetector("test_get_primary_anomaly_map"),
        TestMahalanobisDetector("test_amap_border_clean"),
        TestMahalanobisDetector("test_amap_normalize"),
    ]
    suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

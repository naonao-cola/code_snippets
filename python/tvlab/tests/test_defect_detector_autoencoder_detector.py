from tvlab import *
import unittest
import cv2
import numpy as np


class TestMsAeDefectDetector(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestMsAeDefectDetector, self).__init__(*args, **kwargs)
        self.model_path = osp.normpath('./models/defect_detection/msae_test_model.pth')
        os.makedirs(osp.dirname(self.model_path), exist_ok=True)

    def test_init(self):
        msae = MsAeDefectDetector(border_coef=1.2, device='cuda')
        self.assertEqual(msae.device, 'cuda')
        self.assertEqual(msae._border_coef, 1.2)

    def test_train(self):
        ill = ImageLabelList.from_folder(osp.normpath('./data/defect_data/test/'))
        msae = MsAeDefectDetector(device='cpu')
        vp_err = msae.train(ill, [], bs=2, workers=2, ae_bs=8192, ae_epochs=10, vp_bs=8192, vp_epochs=10, debug=False)
        msae.save(self.model_path)
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
        msae = MsAeDefectDetector(device='cpu')
        msae.load(self.model_path)
        img = cv2.imread(osp.normpath('./data/defect_data/test/010.jpg'))
        amap = msae.get_anomaly_map_from_rgb([img])
        self.assertEqual(str(type(amap)), "<class 'numpy.ndarray'>")

    def test_get_primary_anomaly_map(self):
        msae = MsAeDefectDetector(device='cpu')
        msae.load(self.model_path)
        img = cv2.imread(osp.normpath('./data/defect_data/test/010.jpg'))
        amap = msae.get_primary_anomaly_map([img])
        self.assertEqual(str(type(amap)), "<class 'torch.Tensor'>")

    def test_amap_border_clean(self):
        msae = MsAeDefectDetector(device='cpu')
        msae.load(self.model_path)
        img = cv2.imread(osp.normpath('./data/defect_data/test/010.jpg'))
        amap = msae.get_primary_anomaly_map([img])
        amap = msae.amap_border_clean(amap)
        self.assertEqual(str(type(amap)), "<class 'torch.Tensor'>")

    def test_amap_normalize(self):
        msae = MsAeDefectDetector(device='cpu')
        msae.load(self.model_path)
        img = cv2.imread(osp.normpath('./data/defect_data/test/010.jpg'))
        amap = msae.get_primary_anomaly_map([img])
        amap = msae.amap_border_clean(amap)
        amap = msae.amap_normalize(amap)
        self.assertEqual(str(type(amap)), "<class 'numpy.ndarray'>")


if __name__ == '__main__':
    suite = unittest.TestSuite()

    tests = [
        TestMsAeDefectDetector("test_init"),
        TestMsAeDefectDetector("test_train"),
        TestMsAeDefectDetector("test_get_anomaly_map_from_rgb"),
        TestMsAeDefectDetector("test_get_primary_anomaly_map"),
        TestMsAeDefectDetector("test_amap_border_clean"),
        TestMsAeDefectDetector("test_amap_normalize")
    ]
    suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

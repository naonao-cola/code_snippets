import os
import unittest

from tvlab import *


class TestTvdlAnomalyDetectionTrainFPS(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestTvdlAnomalyDetectionTrainFPS,
              self).__init__(*args, **kwargs)

        self.work_dir = osp.normpath('models/tvdl_anomaly_detection_MSAE')
        os.makedirs(self.work_dir, exist_ok=True)

        self.dataset_dir = osp.normpath('data/defect_data/defect_image')
        self.dataset_json_dir = self.dataset_dir

        import imgaug.augmenters as iaa

        iaa_resize = iaa.Sequential([
            iaa.Resize({"height": 128, "width": 64,
                        "keep-aspect-ratio": False})
        ])
        self.train_schedule = {
            'lr': 1.0,
            'bs': 2,
            'num_workers': 2,
            'epochs': 2,
            'gpus': [0],
            'check_per_epoch': 1,
            'monitor': 'loss'
        }

        self.train_tfms = [iaa_resize]
        self.valid_tfms = [iaa_resize]

        self.ill = ImageLabelList.from_folder(self.dataset_dir).label_from_folder()
        self.ill = ImagePolygonLabelList.merge(self.ill, self.ill)
        self.ill.split(valid_pct=0.2)

    def test_train(self):
        tvdlAnomalyDetectionTrain = TvdlAnomalyDetectionTrain(self.work_dir)
        tvdlAnomalyDetectionTrain.train(self.ill, self.train_schedule, self.train_tfms, self.valid_tfms)
        tvdlAnomalyDetectionTrain.package_model(osp.join(self.work_dir, 'model.capp'),
                                                import_cmd='from tvlab import TestTvdlAnomalyDetectionInference',
                                                model_fmt='all')
        flg_capp, flg_onnx, flg_pth = False, False, False
        for root, dirs, files in os.walk(self.work_dir):
            for file in files:
                model_fmt = os.path.splitext(file)[-1]
                if model_fmt == '.capp':
                    flg_capp = True
                elif model_fmt == '.onnx':
                    flg_onnx = True
                elif model_fmt in ['.pth', '.ckpt']:
                    flg_pth = True
        self.assertTrue(flg_capp, True)
        self.assertTrue(flg_onnx, True)
        self.assertTrue(flg_pth, True)

    def test_predict(self):
        tvdlSegmentationInference = TvdlAnomalyDetectionInference(
            osp.join(self.work_dir, 'model.capp'), use_onnx=False, devices=['cuda'])
        tvdlSegmentationInference.load_model()
        y_pred = tvdlSegmentationInference.predict(self.ill, self.valid_tfms)

        self.assertTrue(isinstance(y_pred, list))


if __name__ == '__main__':
    suite = unittest.TestSuite()

    tests = [
        TestTvdlAnomalyDetectionTrainFPS("test_train"),
        TestTvdlAnomalyDetectionTrainFPS("test_predict"),
    ]

    suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

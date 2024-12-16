from tvlab import *
import unittest
import os
import shutil


class TestTvdlDetectionTrain(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestTvdlDetectionTrain, self).__init__(*args, **kwargs)

        self.work_dir = osp.normpath('models/tvdl_detection_yolo')
        shutil.rmtree(self.work_dir, ignore_errors=True)
        os.makedirs(self.work_dir, exist_ok=True)

        self.dataset_dir = osp.normpath('data/coco_instance/')
        self.dataset_xml_dir = os.path.join(self.dataset_dir, "detection_xml")

        import imgaug.augmenters as iaa
        iaa_aug_seg = iaa.Sequential([
            iaa.GaussianBlur((0, 1.0)),
            iaa.AdditiveGaussianNoise(loc=0, scale=(
                0.0, 0.02 * 255), per_channel=False),
            iaa.Multiply((0.8, 1.25), per_channel=False),
            iaa.Add((0, 30), per_channel=False),
            iaa.LinearContrast((0.8, 1.25), per_channel=False),
            iaa.Affine(
                scale={'x': (0.99, 1.01), 'y': (0.99, 1.01)},
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}),
            iaa.PerspectiveTransform(scale=(0, 0.02)),
        ], random_order=False)

        self.train_schedule = {
            'backbone': 'resnet18',
            'lr': 0.001,
            'bs': 4,
            'num_workers': 2,
            'epochs': 4,
            'gpus': [0],
            'check_per_epoch': 2
        }

        iaa_resize = iaa.Sequential([
            iaa.Resize({"height": 512, "width": 512,
                        "keep-aspect-ratio": False})
        ])
        self.train_tfms = [iaa_aug_seg, iaa_resize]
        self.valid_tfms = [iaa_resize]

        self.ibll = ImageBBoxLabelList.from_pascal_voc(
            self.dataset_dir, self.dataset_xml_dir)
        self.ibll.split(valid_pct=0.2)

    def test_train(self):
        trainer = TvdlDetectionTrain(self.work_dir)
        trainer.train(
            self.ibll, self.train_schedule, self.train_tfms, self.valid_tfms)
        trainer.package_model(osp.join(self.work_dir, 'model.capp'),
                              import_cmd='from tvlab import TvdlDetectionInference',
                              classes=self.ibll.labelset(),
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
        inf = TvdlDetectionInference(
            osp.join(self.work_dir, 'model.capp'), use_onnx=False, devices=['cuda'])
        inf.load_model()
        y_pred = inf.predict(
            self.ibll, self.valid_tfms, 6, 2)
        self.assertTrue(isinstance(y_pred, list))
        if len(y_pred) != 0:
            self.assertEqual(list(y_pred[0].keys()), ['labels', 'bboxes'])


if __name__ == '__main__':
    suite = unittest.TestSuite()

    tests = [
        TestTvdlDetectionTrain("test_train"),
        TestTvdlDetectionTrain("test_predict"),
    ]

    suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

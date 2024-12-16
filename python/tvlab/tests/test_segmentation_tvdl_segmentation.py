from tvlab import *
import unittest
import os
import numpy as np
from imgaug import augmenters as iaa


class TestTvdlSegmentationTrain(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestTvdlSegmentationTrain, self).__init__(*args, **kwargs)

        self.work_dir = osp.normpath('models/tvdl_segmentation_unet')
        import shutil
        shutil.rmtree(self.work_dir, ignore_errors=True)
        os.makedirs(self.work_dir, exist_ok=True)

        self.dataset_dir = osp.normpath('data/segmentation_data/')
        self.dataset_json_dir = self.dataset_dir

        ipll = ImagePolygonLabelList.from_labelme(
            self.dataset_dir, self.dataset_json_dir)
        self.ipll = ImagePolygonLabelList.merge(ipll, ipll)
        self.ipll.split(valid_pct=0.2)

        y0, y1 = 1428, 1776
        x0, x1 = 1000, 2100
        img_w, img_h = get_image_res(ipll.x[0])

        def polygon_tfm(p, img_size):
            p = np.array(p).reshape(-1, 2)
            # recover scale
            p[:, 0].dot((x1 - x0) / 256)
            p[:, 1].dot((y1 - y0) / 256)
            # recover bias
            p[:, 0] += x0
            p[:, 1] += y0
            return p.flatten().tolist()

        from imgaug import augmenters as iaa
        aug_seq = iaa.Sequential([
            iaa.GaussianBlur((0, 1.0)),
            iaa.AdditiveGaussianNoise(loc=0, scale=(
                0.0, 0.02 * 255), per_channel=False),
            iaa.Multiply((0.8, 1.25), per_channel=False),
            iaa.Add((0, 30), per_channel=False),
            iaa.LinearContrast((0.8, 1.25), per_channel=False),
            iaa.Affine(
                scale={'x': (0.99, 1.01), 'y': (0.7, 1.01)},
                translate_percent={"x": (-0.01, 0.01), "y": (-0.01, 0.01)}),
            iaa.PerspectiveTransform(scale=(0, 0.02)),
        ], random_order=False)

        fixed_crop = iaa.CropToFixedSize(
            x1 - x0, y1 - y0, position=((img_w - x1) / x0, (img_h - y1) / y0))
        resize = iaa.Resize({"height": 256, "width": 256})

        self.train_tfms = [fixed_crop, resize, aug_seq]
        self.valid_tfms = [fixed_crop, resize]
        self.polygon_tfm = polygon_tfm

        self.train_schedule = {
            'lr': 0.001,
            'bs': 1,
            'num_workers': 1,
            'epochs': 4,
            'gpus': [0],
            'check_per_epoch': 2
        }

    def test_train(self):
        trainer = TvdlSegmentationTrain(self.work_dir)
        trainer.train(
            self.ipll, self.train_schedule, self.train_tfms, self.valid_tfms)
        trainer.package_model(osp.join(self.work_dir, 'model.capp'),
                              import_cmd='from tvlab import TvdlSegmentationInference',
                              classes=self.ipll.labelset(),
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
        inf = TvdlSegmentationInference(
            osp.join(self.work_dir, 'model.capp'),
            use_onnx=False, devices=['cuda'])
        inf.load_model()
        y_pred = inf.predict(self.ipll, self.valid_tfms,
                             polygon_tfm=self.polygon_tfm)

        self.assertTrue(isinstance(y_pred, list))


if __name__ == '__main__':
    suite = unittest.TestSuite()

    tests = [
        TestTvdlSegmentationTrain("test_train"),
        TestTvdlSegmentationTrain("test_predict"),
    ]

    suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

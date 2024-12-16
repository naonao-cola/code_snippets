'''
Copyright (C) 2023 TuringVision

Test category image data process class.
'''
import unittest
from tvlab import *
from imgaug import augmenters as iaa
import os
import numpy as np
import shutil


class TestFastSegmentationTrain(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestFastSegmentationTrain, self).__init__(*args, **kwargs)

        self.TASK_PATH = osp.normpath('models/segmentation/mask_rcnn_x50')
        shutil.rmtree(self.TASK_PATH, ignore_errors=True)
        os.makedirs(self.TASK_PATH, exist_ok=True)
        self.model_path = os.path.join(self.TASK_PATH, 'model.capp')
        self.import_cmd = 'from tvlab import FastSegmentationInference'
        self.cfg_file = osp.normpath('config/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml')

        self.DATA_PATH = osp.normpath('data/segmentation_data')
        self.DATA_PATH_JSON = self.DATA_PATH
        ipll = ImagePolygonLabelList.from_labelme(
            self.DATA_PATH, json_dir=self.DATA_PATH_JSON)
        self.train_ipll = ImagePolygonLabelList.merge(ipll, ipll)
        self.train_schedule = {'bs': 2, 'num_workers': 2}

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

    def test_train_model(self):
        print("=" * 60)
        trainer = FastSegmentationTrain(self.TASK_PATH, self.cfg_file)
        cfg = trainer.cfg
        cfg.SOLVER.BASE_LR = 0.001
        cfg.SOLVER.STEPS = (1500,)
        cfg.SOLVER.WARMUP_ITERS = 10
        cfg.SOLVER.MAX_ITER = 20
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
        cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False

        trainer.train(self.train_ipll, self.train_schedule,
                      train_tfms=self.train_tfms, valid_tfms=self.valid_tfms)
        trainer.package_model(self.model_path, self.import_cmd)
        eva_result = trainer.evaluate(
            self.TASK_PATH, polygon_tfm=self.polygon_tfm)
        self.assertEqual(len(eva_result), 3)

    def test_evaluate(self):
        print("=" * 60)
        html_path = os.path.join(self.TASK_PATH, 'index.html')
        self.assertTrue(os.path.exists(html_path))
        evad = EvalSegmentation.from_pkl(
            os.path.join(self.TASK_PATH, 'evaluate.pkl'))
        result = evad.get_result_list()
        self.assertTrue("negative_area" in result[0])

    def test_inference(self):
        print("=" * 60)
        inf = FastSegmentationInference(self.model_path)
        ipll = ImagePolygonLabelList.from_folder(self.DATA_PATH)
        ipll.split(1.0)
        result = inf.predict(ipll, self.valid_tfms,
                             polygon_tfm=self.polygon_tfm)
        self.assertTrue(isinstance(result, list))


if __name__ == '__main__':
    suite = unittest.TestSuite()

    tests = [
        TestFastSegmentationTrain("test_train_model"),
        TestFastSegmentationTrain("test_evaluate"),
        TestFastSegmentationTrain("test_inference"),
    ]
    suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

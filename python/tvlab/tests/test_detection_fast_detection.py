'''
Copyright (C) 2023 TuringVision

Test category image data process class.
'''
import unittest
from tvlab import *
from imgaug import augmenters as iaa
import os
import shutil


class TestFastDetectionTrain(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestFastDetectionTrain, self).__init__(*args, **kwargs)

        self.TASK_PATH = osp.normpath('models/detection/faster_rcnn_x50')
        shutil.rmtree(self.TASK_PATH, ignore_errors=True)
        os.makedirs(self.TASK_PATH, exist_ok=True)
        self.model_path = os.path.join(self.TASK_PATH, 'model.capp')
        self.import_cmd = 'from tvlab import FastDetectionInference'
        self.cfg_file = osp.normpath('config/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml')

        self.DATA_PATH = osp.normpath("data/coco_instance/")
        self.DATA_PATH_XML = os.path.join(self.DATA_PATH, "detection_xml")
        ibll = ImageBBoxLabelList.from_pascal_voc(
            self.DATA_PATH, xml_dir=self.DATA_PATH_XML)
        self.train_ibll = ImageBBoxLabelList.merge(ibll, ibll)

        self.train_schedule = {'bs': 2, 'num_workers': 2}

        aug_seq = iaa.Sequential([
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

        resize = iaa.Resize(256)

        self.train_tfms = [resize, aug_seq]
        self.valid_tfms = [resize]

    def test_train_model(self):
        print("=" * 60)
        trainer = FastDetectionTrain(self.TASK_PATH, self.cfg_file)
        cfg = trainer.cfg
        cfg.SOLVER.BASE_LR = 0.001
        cfg.SOLVER.STEPS = (1500,)
        cfg.SOLVER.WARMUP_ITERS = 10
        cfg.SOLVER.MAX_ITER = 20
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
        cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False

        trainer.train(self.train_ibll, self.train_schedule, self.train_tfms, self.valid_tfms)

        trainer.package_model(self.model_path, self.import_cmd)

        eva_result = trainer.evaluate(self.TASK_PATH)
        self.assertEqual(len(eva_result), 3)

    def test_evaluate(self):
        print("=" * 60)
        html_path = os.path.join(self.TASK_PATH, 'index.html')
        self.assertTrue(os.path.exists(html_path))
        evad = EvalDetection.from_pkl(
            os.path.join(self.TASK_PATH, 'evaluate.pkl'))
        result = evad.get_result_list()
        self.assertTrue("dog" in result[0])

    def test_inference(self):
        print("=" * 60)
        inf = FastDetectionInference(self.model_path)
        ibll = ImageBBoxLabelList.from_folder(self.DATA_PATH)
        ibll.split(1.0)
        y_pred = inf.predict(ibll, self.valid_tfms, 2, 2)
        self.assertTrue(isinstance(y_pred, list))
        self.assertEqual(len(y_pred), len(ibll.x))


if __name__ == '__main__':
    suite = unittest.TestSuite()

    tests = [
        TestFastDetectionTrain("test_train_model"),
        TestFastDetectionTrain("test_evaluate"),
        TestFastDetectionTrain("test_inference"),
    ]
    suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

'''
Copyright (C) 2023 TuringVision

Test category image data process class.
'''
import os
import unittest
from tvlab import *
from imgaug import augmenters as iaa


class TestFastOCRTrain(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestFastOCRTrain, self).__init__(*args, **kwargs)
        import paddleocr
        base_dir_name = os.path.dirname(paddleocr.__file__)

        self.TASK_PATH = 'models/ocr/end2end/'
        os.makedirs(self.TASK_PATH, exist_ok=True)
        self.model_path = os.path.join(self.TASK_PATH, 'end2end_model.capp')
        self.import_cmd = 'from tvlab import FastOcrEnd2EndInference'
        self.cfg_file = os.path.join(base_dir_name, "configs/det/det_mv3_db_v1.1.yml")
        self.DATA_PATH = "data/MTWI/image_train"
        self.DATA_PATH_TXT = "data/MTWI/txt_train"
        ioll = ImageOCRPolygonLabelList.from_mtwi(self.DATA_PATH, self.DATA_PATH_TXT)
        self.train_ioll = ImageOCRPolygonLabelList.merge(ioll, ioll)

        self.train_schedule = {'bs': 2, 'num_workers': 2, 'epochs': 50}

        aug_seq = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Affine(rotate=(-10, 10)),
            iaa.Resize((0.5, 3))])

        def aug_img(img, gt):
            return imgaug_img_ocr_polygon_tfm(img, gt, aug_seq)

        self.train_tfms = [aug_img]

    def test_train_model(self):
        print("=" * 60)
        trainer = FastOcrEnd2EndTrain(self.TASK_PATH, self.cfg_file)
        trainer.train(self.train_ioll, self.train_schedule, self.train_tfms)
        trainer.package_model(self.model_path, self.import_cmd)
        eva_result = trainer.evaluate()
        self.assertIsInstance(eva_result, EvalSegmentation)

    def test_inference(self):
        print("=" * 60)
        inf = FastOcrEnd2EndInference(self.model_path)
        ioll = ImageOCRPolygonLabelList.from_folder(self.DATA_PATH)
        ioll.split(1.0)
        result = inf.predict(ioll)


if __name__ == '__main__':
    suite = unittest.TestSuite()

    tests = [
        TestFastOCRTrain("test_train_model"),
        TestFastOCRTrain("test_inference"),
    ]
    suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

'''
Copyright (C) 2023 TuringVision
'''
import unittest
from tvlab import *
from imgaug import augmenters as iaa
import os
import os.path as osp
import shutil


class TestFastCategoryTrain(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestFastCategoryTrain, self).__init__(*args, **kwargs)
        self.TASK_PATH = osp.normpath('models/category/ants_bees_resnet18')
        shutil.rmtree(self.TASK_PATH, ignore_errors=True)
        os.makedirs(self.TASK_PATH, exist_ok=True)

        self.model_path = os.path.join(self.TASK_PATH, 'model.capp')
        self.import_cmd = 'from tvlab import FastCategoryInference'

        self.DATA_PATH = osp.normpath('data/hymenoptera_data')
        ill = ImageLabelList.from_folder(self.DATA_PATH).label_from_folder()
        train_ill = ImageLabelList.merge(ill, ill)
        _, _ = train_ill.split()
        self.train_ill = train_ill

        self.train_schedule = {
            'basemodel': 'resnet18',
            'monitor': 'accuracy',
            'bs': 4,
            'num_workers': 4,
            'steps': [
                {'epochs': 2,
                 'lr': 0.001,
                 'freeze_layer': -1  # 参与训练的层，一般分3层
                 }
            ],
        }

        from imgaug import augmenters as iaa
        aug_seq = iaa.Sequential([
            iaa.Fliplr(p=0.5),
            iaa.Flipud(p=0.5),
            iaa.Crop(percent=(0, 0.1), keep_size=True),
            iaa.AdditiveGaussianNoise(
                loc=0, scale=(0.0, 0.1 * 255), per_channel=0.5),
            iaa.Multiply((0.8, 1.2), per_channel=0.2),
            iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-25, 25),
                shear=(-8, 8))
        ], random_order=True)

        resize = iaa.Resize(128)
        center_crop = iaa.CropToFixedSize(112, 112, position='center')
        random_crop = iaa.CropToFixedSize(112, 112)

        self.train_tfms = [resize, random_crop, aug_seq]
        self.valid_tfms = [resize, center_crop]

    def test_train_model(self):
        print("=" * 60)
        trainer = FastCategoryTrain(self.TASK_PATH)
        trainer.train(self.train_ill, self.train_schedule,
                      self.train_tfms, self.valid_tfms)
        trainer.package_model(self.model_path, self.import_cmd)
        eva_result = trainer.evaluate(self.TASK_PATH)
        self.assertEqual(len(eva_result), 3)

    def test_evaluate(self):
        print("=" * 60)
        html_path = os.path.join(self.TASK_PATH, 'index.html')
        self.assertTrue(os.path.exists(html_path))
        evac = EvalCategory.from_pkl(
            os.path.join(self.TASK_PATH, 'evaluate.pkl'))
        result = evac.get_result()
        self.assertCountEqual(result['classes'], ['ants', 'bees', 'Other'])

    def test_inference(self):
        print("=" * 60)
        inf = FastCategoryInference(self.model_path)
        ill = ImageLabelList.from_folder(self.DATA_PATH)
        ill.split(1.0)
        result = inf.predict(ill, self.valid_tfms)
        self.assertEqual(len(result), len(ill))


if __name__ == '__main__':
    suite = unittest.TestSuite()

    tests = [
        TestFastCategoryTrain("test_train_model"),
        TestFastCategoryTrain("test_evaluate"),
        TestFastCategoryTrain("test_inference"),
    ]
    suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

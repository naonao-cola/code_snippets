'''
Copyright (C) 2023 TuringVision

Test category image data process class.
'''
import unittest
from tvlab import *
from imgaug import augmenters as iaa
import os
import shutil


class TestCategoryExperiment(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestCategoryExperiment, self).__init__(*args, **kwargs)
        self.TASK_PATH_REG = osp.normpath('models/category/ants_bees_')
        self.CATEGORY_DIR = osp.normpath('models/category')
        shutil.rmtree(self.CATEGORY_DIR, ignore_errors=True)
        os.makedirs(self.CATEGORY_DIR, exist_ok=True)

        self.DATA_PATH = osp.normpath('data/hymenoptera_data')
        ill = ImageLabelList.from_folder(self.DATA_PATH).label_from_folder()
        train_ill = ImageLabelList.merge(ill, ill)
        _, _ = train_ill.split()
        self.train_ill = train_ill

        self.train_schedule = {
            'basemodel': '',
            'monitor': 'accuracy',
            'bs': 4,
            'num_workers': 0,
            'steps': [
                {'epochs': 2,
                 'lr': 0.001,
                 'freeze_layer': -1
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

    def test_train_experiment(self):
        print("=" * 60)
        for arch in ['resnet18', 'resnet50']:  # , 'resnet101', 'densenet161']:
            print('training:', arch)
            task_path = self.TASK_PATH_REG + arch
            os.makedirs(task_path, exist_ok=True)
            trainer = FastCategoryTrain(task_path)
            self.train_schedule['basemodel'] = arch
            trainer.train(self.train_ill, self.train_schedule,
                          self.train_tfms, self.valid_tfms)
            model_path = os.path.join(task_path, 'model.capp')
            import_cmd = 'from tvlab import FastCategoryInference'
            trainer.package_model(model_path, import_cmd)
            trainer.evaluate(task_path)

    def test_experiment(self):
        print("=" * 60)
        exp = CategoryExperiment(self.CATEGORY_DIR)
        self.assertEqual(len(exp), 2)
        self.assertIsInstance(exp[0], CategoryModelInfo)


if __name__ == '__main__':
    suite = unittest.TestSuite()

    tests = [
        TestCategoryExperiment("test_train_experiment"),
        TestCategoryExperiment("test_experiment"),
    ]
    suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

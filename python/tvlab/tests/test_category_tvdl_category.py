from tvlab import *
import unittest
import os
from imgaug import augmenters as iaa
import shutil


class TestTvdlCategoryTrain(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestTvdlCategoryTrain, self).__init__(*args, **kwargs)
        self.work_dir = osp.normpath('models/tvdl_category')
        shutil.rmtree(self.work_dir, ignore_errors=True)
        os.makedirs(self.work_dir, exist_ok=True)
        self.dataset_dir = osp.normpath('data/hymenoptera_data/')

        import albumentations as A
        aug_seq = A.Compose([
            A.Flip(),
            A.RandomContrast(),
            A.ColorJitter(),
            A.GaussNoise(),
            A.RandomBrightness(),
            A.RandomGamma(),
            A.ShiftScaleRotate(),
        ])

        resize_to = A.Resize(256, 256)
        rand_crop_to = A.RandomCrop(224, 224)
        center_crop_to = A.CenterCrop(224, 224)

        self.train_tfms = [aug_seq, resize_to, rand_crop_to]
        self.valid_tfms = [resize_to, center_crop_to]
        self.train_schedule = {
            'backbone': 'resnet18',
            'lr': 0.01,
            'bs': 2,
            'num_workers': 2,
            'epochs': 4,
            'gpus': [0],
            'check_per_epoch': 2
        }
        self.ill = ImageLabelList.from_folder(
            self.dataset_dir).label_from_folder()
        self.ill.split(valid_pct=0.2, seed=1234)

    def test_train(self):
        tvdlCategoryTrain = TvdlCategoryTrain(self.work_dir)
        tvdlCategoryTrain.train(
            self.ill, self.train_schedule, self.train_tfms, self.valid_tfms)
        tvdlCategoryTrain.package_model(osp.join(self.work_dir, 'model.capp'),
                                        import_cmd='from tvlab import TvdlCategoryInference',
                                        classes=self.ill.labelset(),
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
        tvdlCategoryInference = TvdlCategoryInference(
            osp.join(self.work_dir, 'model.capp'), use_onnx=False, devices=['cuda'])
        tvdlCategoryInference.load_model()
        y_pred = tvdlCategoryInference.predict(self.ill, self.valid_tfms, 2, 2)


if __name__ == '__main__':
    suite = unittest.TestSuite()

    tests = [
        TestTvdlCategoryTrain("test_train"),
        TestTvdlCategoryTrain("test_predict"),
    ]

    suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

'''
Copyright (C) 2023 TuringVision
'''
import unittest
from imgaug import augmenters as iaa
import os
import shutil
from tvlab import *



class TestFastSimilarCnn(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestFastSimilarCnn, self).__init__(*args, **kwargs)
        self.TASK_PATH = osp.normpath('models/category/fast_similar/cnn')
        shutil.rmtree(self.TASK_PATH, ignore_errors=True)
        os.makedirs(self.TASK_PATH, exist_ok=True)

        self.model_path = os.path.join(self.TASK_PATH, 'model.capp')
        self.import_cmd = 'from tvlab import FastSimilarCnnInference'

        self.DATA_PATH = osp.normpath('data/defect_data/defect_image')
        ill = ImageLabelList.from_folder(self.DATA_PATH)
        labels = ["OK"] * 5 + ["NG"] * 5
        ill = ImageMultiLabelList(ill.x, labels)
        train_ill = ImageLabelList.merge(ill, ill)
        _, _ = train_ill.split()
        self.train_ill = train_ill

        self.train_schedule = {
            'basemodel': 'resnet18',  # 基础模型
            'monitor': 'accuracy',  # 最佳模型的监控指标
            'bs': 2,  # batch_size
            'num_workers': 0, # set to 0. some env, workers > 0 will stuck
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
            iaa.AdditiveGaussianNoise(
                loc=0, scale=(0.0, 0.1 * 255), per_channel=0.5),
            iaa.Multiply((0.8, 1.2), per_channel=0.2),
        ], random_order=True)

        resize = iaa.Resize(64+16)
        center_crop = iaa.CropToFixedSize(64, 64, position='center')
        random_crop = iaa.CropToFixedSize(64, 64)

        self.train_tfms = [resize, random_crop, aug_seq]
        self.valid_tfms = [resize, center_crop]

    def test_train_model(self):
        print("=" * 60)
        trainer = FastSimilarCnnTrain(self.TASK_PATH)
        trainer.train(self.train_ill, self.train_schedule,
                      self.train_tfms, self.valid_tfms)
        print("package_model ...")
        trainer.package_model(self.model_path, self.import_cmd)
        print("evaluate ...")
        eva_result = trainer.evaluate(self.TASK_PATH,
                                      preds=[{'OK': 0.6, 'NG': 0.1} for i in range(10)],
                                      target=[0, 1, 1, 1, 0, 0, 1, 0, 1, 0],
                                      class_list=['OK', 'NG'],
                                      conf=0.5)
        self.assertEqual(len(eva_result), 3)

    def test_evaluate(self):
        print("=" * 60)
        html_path = os.path.join(self.TASK_PATH, 'index.html')
        self.assertTrue(os.path.exists(html_path))
        evac = EvalCategory.from_pkl(
            os.path.join(self.TASK_PATH, 'evaluate.pkl'))
        result = evac.get_result()
        self.assertCountEqual(result['classes'], ['OK', 'NG', 'Other'])

    def test_inference(self):
        print("=" * 60)
        inf = FastSimilarCnnInference(self.model_path)
        ill = ImageLabelList.from_folder(self.DATA_PATH)
        ill.split(1.0)
        result = inf.predict(ill, self.valid_tfms)
        self.assertEqual(len(result), len(ill))


if __name__ == '__main__':
    suite = unittest.TestSuite()
    tests = [
        TestFastSimilarCnn("test_train_model"),
        TestFastSimilarCnn("test_evaluate"),
        TestFastSimilarCnn("test_inference"),
    ]
    suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

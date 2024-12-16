import unittest
from tvlab import *
from imgaug import augmenters as iaa
import os
import shutil


class TestMultiTaskCategory(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestMultiTaskCategory, self).__init__(*args, **kwargs)
        self.TASK_PATH = osp.normpath('models/category/multi_resnet18')

        shutil.rmtree(self.TASK_PATH, ignore_errors=True)
        os.makedirs(self.TASK_PATH, exist_ok=True)

        self.model_path = os.path.join(self.TASK_PATH, 'model.capp')
        self.import_cmd = 'from tvlab import MultiTaskCategoryTrain'

        self.DATA_PATH = osp.normpath('data/defect_data/defect_image')
        imll = ImageMultiLabelList.from_folder(self.DATA_PATH)
        labels = []
        for path in imll.x:
            main_code = path.split('/')[-1].split('_')[0]
            sub_code = path.split('/')[-1].split('_')[-5]
            labels.append((main_code, sub_code))
        ill = ImageMultiLabelList(imll.x, labels)
        train_ill = ImageMultiLabelList.merge(ill, ill)
        _, _ = train_ill.split()
        self.train_ill = train_ill

        self.train_schedule = {
            'basemodel': 'resnet18',  # 基础模型
            'monitor': 'accuracy',  # 最佳模型的监控指标
            'bs': 4,  # batch_size
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
        trainer = MultiTaskCategoryTrain(self.TASK_PATH)
        trainer.train(self.train_ill, self.train_schedule,
                      self.train_tfms, self.valid_tfms)
        trainer.package_model(self.model_path, self.import_cmd)
        eva_result = trainer.evaluate(self.TASK_PATH)
        print(len(eva_result))
        self.assertEqual(len(eva_result), 3)

    def test_evaluate(self):
        print("=" * 60)
        html_path = os.path.join(self.TASK_PATH, 'index.html')
        self.assertTrue(os.path.exists(html_path))
        evac0 = EvalCategory.from_pkl(os.path.join(
            self.TASK_PATH, 'evaluate_task_0.pkl'))
        result0 = evac0.get_result()
        evac1 = EvalCategory.from_pkl(os.path.join(
            self.TASK_PATH, 'evaluate_task_1.pkl'))
        result = evac1.get_result()
        self.assertEqual(len(result0['classes']), len(
            ['ACT', 'GA1', 'GA2', 'SDT', 'Other']))
        self.assertEqual(len(result['classes']), len(
            ['D8D', 'H2U', 'L2T', 'P6U', 'R1D', 'R2D', 'R3T', 'Other']))

    def test_inference(self):
        print("=" * 60)
        inf = MultiTaskCategoryInference(self.model_path)
        ill = ImageLabelList.from_folder(self.DATA_PATH)
        ill.split(1.0)
        result = inf.predict(ill, self.valid_tfms)
        self.assertEqual(len(result[0]), len(ill))
        self.assertEqual(len(result[1]), len(ill))


if __name__ == '__main__':
    suite = unittest.TestSuite()
    tests = [
        TestMultiTaskCategory("test_train_model"),
        TestMultiTaskCategory("test_evaluate"),
        TestMultiTaskCategory("test_inference"),
    ]
    suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

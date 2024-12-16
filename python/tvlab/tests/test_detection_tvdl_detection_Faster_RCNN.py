from tvlab import *
import unittest
import os
from imgaug import augmenters as iaa
from tvdl.common import DetectionModelBase


class TvdlDetectionTrainFRCNN(TvdlDetectionTrain):
    ''' Tvdl detection model training based Faster RCNN
    '''
    SUPPORT_SCHEDULE_KEYS = ['backbone', 'lr', 'bs', 'num_workers', 'monitor',
                             'epochs', 'gpus', 'check_per_epoch', 'img_c',
                             'backbone_out_indices', 'anchor_sizes', 'aspect_ratios']

    def build_model(self):
        from tvdl.detection import FasterRCNN

        backbone = self._get_param('backbone', 'resnet18')
        lr = self._get_param('lr', 0.001)
        img_c = self._get_param('img_c', 3)
        backbone_out_indices = self._get_param('backbone_out_indices', (3, 4))
        anchor_sizes = self._get_param('anchor_sizes', ((64,), (128,), (256,)))
        aspect_ratios = self._get_param('aspect_ratios', (0.25, 0.5, 1.0))

        model = FasterRCNN(classes=self.classes,
                           backbone=backbone,
                           lr=lr,
                           img_c=img_c,
                           backbone_out_indices=backbone_out_indices,
                           anchor_sizes=anchor_sizes,
                           aspect_ratios=aspect_ratios)
        return model


class TvdlDetectionInferenceFRCNN(TvdlDetectionInference):
    ''' Tvdl detection model inference based Faster RCNN
    '''

    def predict(self, ill, tfms=None, bs=1, num_workers=1, box_tfm=None):
        '''
        ill (ImageBoxLabelList)
        tfms (list) tfm list
        bs (int) batch size
        num_works (int) works's num
        box_tfm (Callable) box transform function
        output:
            [{'labels': ['a', 'b', ...], 'bboxes': [[l,t,r,b,conf], ...]}, ...]
        '''
        from tvdl.detection import FasterRCNN
        import torch

        _, valid_dl = ill.dataloader(
            tfms, tfms, bs=bs, num_workers=num_workers)
        y_pred = []
        for bx, bx_info in tqdm(valid_dl):
            if not self.use_onnx:
                bx = bx.to(self.model.device)
            outputs = self.model.forward(bx)
            keys = ['boxes', 'labels', 'scores']
            outputs = [[op[k].cpu().numpy() for k in keys] for op in outputs]
            if self.use_onnx:
                outputs = torch.from_numpy(outputs[0])

            yp = FasterRCNN.post_process(outputs, bx_info, self.get_class_list(), box_tfm)
            y_pred.extend(yp)
        return y_pred


class TestTvdlDetectionTrainFRCNN(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestTvdlDetectionTrainFRCNN, self).__init__(*args, **kwargs)

        self.work_dir = osp.normpath('models/tvdl_detection_Faster_RCNN')
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
        self.train_tfms = [iaa_resize, iaa_aug_seg]
        self.valid_tfms = [iaa_resize]

        self.ibll = ImageBBoxLabelList.from_pascal_voc(
            self.dataset_dir, self.dataset_xml_dir)
        self.ibll.split(valid_pct=0.2)

    def test_train(self):
        tvdlDetectionTrain = TvdlDetectionTrainFRCNN(self.work_dir)
        tvdlDetectionTrain.train(
            self.ibll, self.train_schedule, self.train_tfms, self.valid_tfms)
        tvdlDetectionTrain.package_model(osp.join(self.work_dir, 'model.capp'),
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
        tvdlDetectionInference = TvdlDetectionInferenceFRCNN(
            osp.join(self.work_dir, 'model.capp'), use_onnx=False, devices=['cuda'])
        tvdlDetectionInference.load_model()
        y_pred = tvdlDetectionInference.predict(
            self.ibll, self.valid_tfms, 6, 2)

        self.assertTrue(isinstance(y_pred, list))
        if len(y_pred) != 0:
            self.assertEqual(list(y_pred[0].keys()), ['labels', 'bboxes'])


if __name__ == '__main__':
    suite = unittest.TestSuite()

    tests = [
        TestTvdlDetectionTrainFRCNN("test_train"),
        TestTvdlDetectionTrainFRCNN("test_predict"),
    ]

    suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

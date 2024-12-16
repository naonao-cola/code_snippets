from tvlab import *
import unittest
import os
from imgaug import augmenters as iaa
from tvdl.common import DetectionModelBase
from test_detection_tvdl_detection import TestTvdlDetectionTrain


class TvdlDetectionTrainCenterNet(TvdlDetectionTrain):
    ''' Tvdl detection model training based centerNet
    '''

    def build_model(self):
        from tvdl.detection import CenterNet

        backbone = self._get_param('backbone', 'resnet18')
        lr = self._get_param('lr', 0.001)
        img_c = self._get_param('img_c', 3)
        backbone_out_indices = self._get_param('backbone_out_indices', (3, 4))

        model = CenterNet(classes=self.classes,
                          backbone=backbone,
                          lr=lr,
                          img_c=img_c,
                          backbone_out_indices=backbone_out_indices)
        return model


class TvdlDetectionInferenceCenterNet(TvdlDetectionInference):
    ''' Tvdl detection model inference based centerNet
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
        from tvdl.detection import CenterNet
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

            DetectionModelBase.post_process(
                outputs, bx_info, self.get_class_list(), box_tfm)
        return y_pred


class TestTvdlDetectionTrainCenterNet(TestTvdlDetectionTrain):
    def __init__(self, *args, **kwargs):
        super(TestTvdlDetectionTrainCenterNet, self).__init__(*args, **kwargs)

        self.work_dir = osp.normpath('models/tvdl_detection_CenterNet')
        import shutil
        shutil.rmtree(self.work_dir, ignore_errors=True)
        os.makedirs(self.work_dir, exist_ok=True)

    def test_train(self):
        trainer = TvdlDetectionTrainCenterNet(self.work_dir)
        trainer.train(
            self.ibll, self.train_schedule, self.train_tfms, self.valid_tfms)
        trainer.package_model(osp.join(self.work_dir, 'model.capp'),
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
        inf = TvdlDetectionInferenceCenterNet(
            osp.join(self.work_dir, 'model.capp'), use_onnx=False, devices=['cuda'])
        inf.load_model()
        y_pred = inf.predict(
            self.ibll, self.valid_tfms, 4, 2)

        self.assertTrue(isinstance(y_pred, list))
        if len(y_pred) != 0:
            self.assertEqual(list(y_pred[0].keys()), ['labels', 'bboxes'])


if __name__ == '__main__':
    suite = unittest.TestSuite()

    tests = [
        TestTvdlDetectionTrainCenterNet("test_train"),
        TestTvdlDetectionTrainCenterNet("test_predict"),
    ]

    suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

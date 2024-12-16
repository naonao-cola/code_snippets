from tvlab import *
import unittest
import os
from tests.test_segmentation_tvdl_segmentation import TestTvdlSegmentationTrain


class TvdlSegmentationTrainMaskRCNN(TvdlSegmentationTrain):
    ''' Tvdl Mask RCNN model training
    '''
    SUPPORT_SCHEDULE_KEYS = ['backbone', 'lr', 'bs', 'num_workers', 'monitor',
                             'epochs', 'gpus', 'check_per_epoch', 'img_c',
                             'backbone_out_indices', 'anchor_sizes', 'aspect_ratios']

    def build_model(self):
        from tvdl.segmentation import MaskRCNN

        backbone = self._get_param('backbone', 'resnet18')
        img_c = self._get_param('img_c', 3)
        lr = self._get_param('lr', 0.001)
        backbone_out_indices = self._get_param('backbone_out_indices', (3, 4))
        anchor_sizes = self._get_param('anchor_sizes', ((64,), (128,), (256,)))
        aspect_ratios = self._get_param('aspect_ratios', (0.25, 0.5, 1.0))

        model = MaskRCNN(self.classes,
                         backbone=backbone,
                         img_c=img_c,
                         lr=lr,
                         backbone_out_indices=backbone_out_indices,
                         anchor_sizes=anchor_sizes,
                         aspect_ratios=aspect_ratios)
        return model

    def train(self, ill, train_schedule, train_tfms=[], valid_tfms=[], cbs=[]):
        if 'monitor' not in train_schedule.keys():
            train_schedule["monitor"] = "loss_objectness"
        super(TvdlSegmentationTrainMaskRCNN, self).train(
            ill, train_schedule, train_tfms, valid_tfms, cbs)


class TvdlSegmentationInferenceMaskRCNN(TvdlSegmentationInference):
    ''' Tvdl MaskRCNN model inference
    '''

    def predict(self, ill, tfms=None, bs=1, num_workers=1,
                mask_threshold=0.5, polygon_tfm=None):
        '''
        ill (ImagePolygonLabelList)
        tfms (list) tfm list
        bs (int) batch size
        num_works (int) works's num
        polygon_tfm (callable)
        mask_threshold (float) mask score threshold
        output:
            [{'labels': ['a', 'b', ...], 'polygons': [[x1,y1,x2,y2,x3,y3,...,conf], ...]}, ...]
        '''
        from tvdl.segmentation import MaskRCNN

        _, valid_dl = ill.dataloader(
            tfms, tfms, bs=bs, num_workers=num_workers)
        y_pred = []
        for bx, bx_info in tqdm(valid_dl):
            if not self.use_onnx:
                bx = bx.to(self.model.device)
            outputs = self.model.forward(bx)
            keys = ['boxes', 'labels', 'scores', 'masks']
            outputs = [[op[k].cpu().numpy() for k in keys] for op in outputs]
            pp = MaskRCNN.post_process(outputs, bx_info, self.get_class_list(),
                                       polygon_tfm, mask_threshold)
            y_pred.extend(pp)
        return y_pred


class TestTvdlSegmentationTrainMaskRCNN(TestTvdlSegmentationTrain):
    def __init__(self, *args, **kwargs):
        super(TestTvdlSegmentationTrainMaskRCNN,
              self).__init__(*args, **kwargs)

        self.work_dir = osp.normpath('models/tvdl_segmentation_MaskRCNN')
        import shutil
        shutil.rmtree(self.work_dir, ignore_errors=True)
        os.makedirs(self.work_dir, exist_ok=True)

    def test_train(self):
        trainer = TvdlSegmentationTrainMaskRCNN(self.work_dir)
        trainer.train(
            self.ipll, self.train_schedule, self.train_tfms, self.valid_tfms)
        trainer.package_model(osp.join(self.work_dir, 'model.capp'),
                              import_cmd='from tvlab import TvdlSegmentationInference',
                              classes=self.ipll.labelset(),
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
        inf = TvdlSegmentationInferenceMaskRCNN(
            osp.join(self.work_dir, 'model.capp'),
            use_onnx=False, devices=['cuda'])
        inf.load_model()
        y_pred = inf.predict(self.ipll, self.valid_tfms,
                             polygon_tfm=self.polygon_tfm)

        self.assertTrue(isinstance(y_pred, list))


if __name__ == '__main__':
    suite = unittest.TestSuite()

    tests = [
        TestTvdlSegmentationTrainMaskRCNN("test_train"),
        TestTvdlSegmentationTrainMaskRCNN("test_predict"),
    ]

    suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

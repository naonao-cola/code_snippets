from tvlab import *
import unittest
import os
from tests.test_segmentation_tvdl_segmentation import TestTvdlSegmentationTrain


class TvdlSegmentationTrainU2Net(TvdlSegmentationTrain):
    ''' Tvdl U2Net model training
    '''
    SUPPORT_SCHEDULE_KEYS = ['lr', 'bs', 'num_workers', 'monitor',
                             'epochs', 'gpus', 'check_per_epoch', 'img_c',
                             'model_cfg']

    def build_model(self):
        from tvdl.segmentation import U2Net

        img_c = self._get_param('img_c', 3)
        lr = self._get_param('lr', 0.001)
        model_cfg = self._get_param('model_cfg', 'lite')

        model = U2Net(self.classes,
                      img_c=img_c,
                      lr=lr,
                      model_cfg=model_cfg)
        return model


class TvdlSegmentationInferenceU2Net(TvdlSegmentationInference):
    ''' Tvdl U2Net model inference
    '''

    def predict(self, ill, tfms=None, bs=1, num_workers=1,
                mask_threshold=0.2, area_threshold=25,
                blob_filter_func=None, polygon_tfm=None):
        '''
        ill (ImagePolygonLabelList)
        tfms (list) tfm list
        bs (int) batch size
        num_works (int) works's num
        polygon_tfm (callable)
        mask_threshold (float) mask score threshold
        area_threshold (int) blob area threshold
        blob_filter_func (callable): return False for unwanted blob
            eg:
                def filter_blob(blob):
                    if blob.roundness < 0.5:
                        return False
        output:
            [{'labels': ['a', 'b', ...], 'polygons': [[x1,y1,x2,y2,x3,y3,...,conf], ...]}, ...]
        '''
        from tvdl.segmentation import U2Net

        _, valid_dl = ill.dataloader(
            tfms, tfms, bs=bs, num_workers=num_workers)
        y_pred = []
        for bx, bx_info in tqdm(valid_dl):
            if not self.use_onnx:
                bx = bx.to(self.model.device)
            outputs = self.model.forward(bx)
            outputs = outputs[0] if self.use_onnx else outputs.cpu().numpy()
            pp = U2Net.post_process(outputs, bx_info, self.get_class_list(),
                                    mask_threshold, area_threshold,
                                    blob_filter_func, polygon_tfm)
            y_pred.extend(pp)
        return y_pred


class TestTvdlSegmentationTrainU2Net(TestTvdlSegmentationTrain):
    def __init__(self, *args, **kwargs):
        super(TestTvdlSegmentationTrainU2Net, self).__init__(*args, **kwargs)

        self.work_dir = osp.normpath('models/tvdl_segmentation_U2Net')
        import shutil
        shutil.rmtree(self.work_dir, ignore_errors=True)
        os.makedirs(self.work_dir, exist_ok=True)

    def test_train(self):
        trainer = TvdlSegmentationTrainU2Net(self.work_dir)
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
        inf = TvdlSegmentationInferenceU2Net(
            osp.join(self.work_dir, 'model.capp'), use_onnx=False, devices=['cuda'])
        inf.load_model()
        y_pred = inf.predict(self.ipll, self.valid_tfms,
                             polygon_tfm=self.polygon_tfm)
        self.assertTrue(isinstance(y_pred, list))


if __name__ == '__main__':
    suite = unittest.TestSuite()

    tests = [
        TestTvdlSegmentationTrainU2Net("test_train"),
        TestTvdlSegmentationTrainU2Net("test_predict"),
    ]

    suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

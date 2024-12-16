from tvlab import *
import unittest
import cv2
import timeit
import os
import numpy as np
from imgaug import augmenters as iaa


class TvdlAnomalyDetectionTrainMSAE(TvdlTrain):
    ''' Tvdl MSAE model training
    '''
    SUPPORT_SCHEDULE_KEYS = ['lr', 'bs', 'num_workers', 'monitor',
                             'epochs', 'gpus', 'check_per_epoch', 'img_c',
                             'noise_std', 'backbone_out_indices', 'replay_buffer_size',
                             'ae_epochs', 'ae_steps', 'vp_steps']

    def build_model(self):
        from tvdl.anomaly_detection import MSAE

        img_c = self._get_param('img_c', 3)
        lr = self._get_param('lr', 0.001)
        backbone_out_indices = self._get_param('backbone_out_indices', (3,))
        batch_size = self._get_param('batch_size', 8192)
        replay_buffer_size = self._get_param('replay_buffer_size', 8192 * 16)
        model = MSAE(img_c=img_c,
                     lr=lr,
                     backbone_out_indices=backbone_out_indices,
                     batch_size=batch_size,
                     replay_buffer_size=replay_buffer_size)
        return model

    def train(self, ill, train_schedule, train_tfms=[], valid_tfms=[], cbs=[]):
        if 'monitor' not in train_schedule.keys():
            train_schedule["monitor"] = "ae_loss"
        super(TvdlAnomalyDetectionTrainMSAE, self).train(ill, train_schedule, train_tfms, valid_tfms, cbs)


class TvdlAnomalyDetectionInferenceMSAE(TvdlInference):
    ''' Tvdl MSAE model inference
    '''

    def predict(self, ill, tfms=None, bs=1, num_workers=1,
                vp_disable=False, v_min=0, v_max=10.0, border_coef=1.618, border_ratio=0.05):
        '''
        ill (ImagePolygonLabelList)
        tfms (list) tfm list
        bs (int) batch size
        num_works (int) works's num
        output: amap (numpy.ndarry NxHxW): 0 ~ 1.0
        '''
        from tvdl.anomaly_detection import MSAE

        _, valid_dl = ill.dataloader(
            tfms, tfms, bs=bs, num_workers=num_workers)
        y_pred = []
        for bx, bx_info in tqdm(valid_dl):
            if not self.use_onnx:
                bx = bx.to(self.model.device)
            amap = self.model.forward(bx, vp_disable=vp_disable).cpu().numpy()

            amap = MSAE.post_process(amap, v_min=v_min, v_max=v_max,
                                     border_coef=border_coef, border_ratio=border_ratio)
            y_pred.extend(amap)
        return y_pred


class TestTvdlAnomalyDetectionTrainMSAE(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestTvdlAnomalyDetectionTrainMSAE,
              self).__init__(*args, **kwargs)

        self.work_dir = osp.normpath('models/tvdl_anomaly_detection_MSAE')
        os.makedirs(self.work_dir, exist_ok=True)

        self.dataset_dir = osp.normpath('data/defect_data/defect_image')
        self.dataset_json_dir = self.dataset_dir

        import imgaug.augmenters as iaa
        iaa_aug = iaa.Sequential([
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

        iaa_resize = iaa.Sequential([
            iaa.Resize({"height": 64, "width": 64,
                        "keep-aspect-ratio": False})
        ])
        self.train_schedule = {
            'lr': 0.001,
            'bs': 2,
            'num_workers': 2,
            'epochs': 2,
            'gpus': [0],
            'check_per_epoch': 2
        }

        self.train_tfms = [iaa_aug, iaa_resize]
        self.valid_tfms = [iaa_resize]

        self.ipll = ImagePolygonLabelList.from_labelme(
            self.dataset_dir, self.dataset_json_dir)
        self.ipll = ImagePolygonLabelList.merge(self.ipll, self.ipll)
        self.ipll.split(valid_pct=0.2)

    def test_train(self):
        tvdlAnomalyDetectionTrain = TvdlAnomalyDetectionTrainMSAE(
            self.work_dir)
        tvdlAnomalyDetectionTrain.train(
            self.ipll, self.train_schedule, self.train_tfms, self.valid_tfms)
        tvdlAnomalyDetectionTrain.package_model(osp.join(self.work_dir, 'model.capp'),
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
        tvdlSegmentationInference = TvdlAnomalyDetectionInferenceMSAE(
            osp.join(self.work_dir, 'model.capp'), use_onnx=False, devices=['cuda'])
        tvdlSegmentationInference.load_model()
        y_pred = tvdlSegmentationInference.predict(self.ipll, self.valid_tfms)

        self.assertTrue(isinstance(y_pred, list))


if __name__ == '__main__':
    suite = unittest.TestSuite()

    tests = [
        TestTvdlAnomalyDetectionTrainMSAE("test_train"),
        TestTvdlAnomalyDetectionTrainMSAE("test_predict"),
    ]

    suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

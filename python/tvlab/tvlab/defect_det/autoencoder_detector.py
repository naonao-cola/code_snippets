'''
Copyright (C) 2023 TuringVision

Pre-trained deep feature based autoencoder defect detect model.
'''
import os
import os.path as osp
import cv2
import numpy as np
from .cnn_feature_based_detector import CnnFeatureBasedDefectDetector

__all__ = ['MsAeDefectDetector']



class MsAeDefectDetector(CnnFeatureBasedDefectDetector):
    def __init__(self, tile_shape=None, overlap=0.5, border_coef=1.618, device='cuda'):
        '''
        pre-trained deep feature based autoencoder defect detect model.

        Usage:
        1. train
        det = MsAeDefectDetector()
        det.train(ibll, tfms, ...)
        det.save('model.pth')

        2. inference
        det = MsAeDefectDetector()
        det.load('model.pth')

        amap_batch = det.get_primary_anomaly_map(img_batch)
        or:
        bboxes_batch = det.get_bboxes_from_rgb(img_batch)


        border_coef: Image boundary weakening factor, 1.618 means significant weakening,
        1.0 means no weakening
        device:
        tile_shape (tuple): (h, w)
        overlap (float or int or (h, w)):
        '''
        from .impl.autoencoder_detector_impl import MsAeDefectDetectorImpl
        MsAeDefectDetector.build_ae_model = MsAeDefectDetectorImpl.build_ae_model
        MsAeDefectDetector.build_vp_model = MsAeDefectDetectorImpl.build_vp_model
        MsAeDefectDetector.train_autoencoder = MsAeDefectDetectorImpl.train_autoencoder
        MsAeDefectDetector.train_valuepredictor = MsAeDefectDetectorImpl.train_valuepredictor

        CnnFeatureBasedDefectDetector.__init__(self, tile_shape=tile_shape,
            overlap=overlap,
            border_coef=border_coef,
            device=device)
        self.ae_model = None
        self.vp_model = None


    def train(self, ibll, tfms, bs=8, workers=8,
              ae_bs=16656, ae_epochs=1000,
              vp_bs=16656, vp_epochs=3000,
              noise_std=0.2, vp_train_ok_pct=99.9,
              vp_disable=False,
              basemodel='densenet161', fv_rf=16,
              border=0,
              ae_early_stop_patience=50, ae_early_stop_min_delta=0.001,
              vp_early_stop_patience=100, vp_early_stop_min_delta=0.001,
              percent_cb=None, debug=True):
        '''
        In:
            ibll (ImageBBoxLabelList): dataset
            tfms (list): list of transform function
            bs (int): batch_size of image loader
            workers (int): num of workers for image loader
            ae_bs (int): batch_size of AutoEncoder feature loader
            ae_epochs (int): epochs of AutoEncoder training
            vp_bs (int): batch_size of ValuePredictor feature loader
            vp_epochs (int): epochs of ValuePredictor training
            noise_std (float): noise for feature traning
            basemodel (str): base model name
            fv_rf (int): receptive filed of feature
            border (int or tuple (h, w)): remove size of feature border
            percent_cb (functional): callable for current status
            debug (bool): debug enable
        Out:
            vp_err (torch.Tensor): vp_err of dataset (useful for threshold calculations)
        '''
        # 1. get normalized fv
        normalized_fv = self.get_normalized_fv(ibll, tfms, bs=bs, workers=workers,
                                               basemodel=basemodel, fv_rf=fv_rf,
                                               border=border,
                                               percent_cb=percent_cb, debug=debug)
        # 2. train autoencoder
        ae_err = self.train_autoencoder(normalized_fv, bs=ae_bs,
                                        epochs=ae_epochs, noise_std=noise_std,
                                        fv_rf=fv_rf, percent_cb=percent_cb,
                                        debug=debug, debug_ok_pct=vp_train_ok_pct,
                                        early_stop_patience=ae_early_stop_patience,
                                        early_stop_min_delta=ae_early_stop_min_delta)

        if vp_disable:
            return ae_err

        # 3. train value predicter
        vp_err = self.train_valuepredictor(normalized_fv, ae_err, bs=vp_bs,
                                           epochs=vp_epochs, noise_std=noise_std,
                                           train_ok_pct=vp_train_ok_pct,
                                           fv_rf=fv_rf, percent_cb=percent_cb,
                                           debug=debug,
                                           early_stop_patience=vp_early_stop_patience,
                                           early_stop_min_delta=vp_early_stop_min_delta)
        return vp_err

    def forward_with_flatten_ori_fv(self, ori_fv):
        '''
        In:
            fv (torch.Tensor): N, C
        out:
            err (torch.Tensor): N
        '''
        normalized_fv = self._fv_normalize(ori_fv, self.fv_mean, self.fv_std)
        ae_err = self.ae_model.get_err(normalized_fv)
        vp_err = ae_err
        if self.vp_model:
            vp_err = self.vp_model.get_err(normalized_fv, ae_err)
        return vp_err.abs()

    def load_from_model_info(self, model_info):
        v = model_info['ae_model_s'][self.fv_rf]
        ae_model = self.build_ae_model(self.fv_mean.shape[0])
        ae_model.load_state_dict(v)
        self.ae_model = ae_model.to(self.device).eval()
        if 'vp_model_s' in model_info:
            vp_model = self.build_vp_model(self.fv_mean.shape[0])
            v = model_info['vp_model_s'][self.fv_rf]
            vp_model.load_state_dict(v)
            self.vp_model = vp_model.to(self.device).eval()

    def save_to_model_info(self, model_info):
        model_info['ae_model_s'] = {self.fv_rf: self.ae_model.state_dict()}
        if self.vp_model:
            model_info['vp_model_s'] = {self.fv_rf: self.vp_model.state_dict()}

    def export_onnx(self, onnx_path, input_shape):
        '''
        onnx_path (str): onnx model save path
        input_shape (tuple): (H, W, C)
        '''
        import torch
        from torch import nn
        class MsAeInfModule(nn.Module):
            def __init__(self, msae):
                super().__init__()
                self.msae = msae
                self.fe_model = msae.fe_model
                self.ae_model = msae.ae_model
                self.vp_model = msae.vp_model

            def forward(self, x):
                return self.msae.forward(x)

        h, w, c = input_shape
        inf_model = MsAeInfModule(self)
        input_sample = torch.randn((1, h, w, c), device=self.device)
        torch.onnx.export(inf_model, input_sample, onnx_path,
                          export_params=True,
                          do_constant_folding=True,
                          opset_version=11,
                          input_names = ['input'],
                          output_names = ['output'],
                          dynamic_axes={'input' : {0 : 'batch_size'},
                                        'output' : {0 : 'batch_size'}})

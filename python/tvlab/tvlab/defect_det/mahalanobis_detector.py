'''
Copyright (C) 2023 TuringVision

Mahalanobis distance defect detector
'''
import numpy as np
import matplotlib.pyplot as plt
from .basic_detector import BasicDefectDetector
from .cnn_feature_based_detector import CnnFeatureBasedDefectDetector


__all__ = ['MahalanobisDetector']


def tensor_mahalanobis(u, v, VI):
    import torch
    delta = u - v
    m = (delta @ VI) * delta
    m = m.sum(dim=-1)
    return torch.sqrt(m)


class MahalanobisDetector(CnnFeatureBasedDefectDetector):
    def __init__(self, tile_shape=None, overlap=0.5, border_coef=1.618, device='cuda'):
        '''
        Using Mahalanobis distance of pre-trained deep feature as anomaly score.

        Usage:
            1. train
            det = MahalanobisDetector()
            det.train(ibll, tfms, ...)
            det.save('model.pth')

            2. inference
            det = MahalanobisDetector()
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
        CnnFeatureBasedDefectDetector.__init__(self, tile_shape=tile_shape,
                                               overlap=overlap,
                                               border_coef=border_coef,
                                               device=device)
        self.train_mean = None
        self.train_cov = None

    def train(self, ibll, tfms, bs=8, workers=8, border=0,
              basemodel='densenet161', fv_rf=16,
              percent_cb=None, debug=True):
        '''
        In:
            ibll (ImageBBoxLabelList): dataset
            tfms (list): list of transform function
            bs (int): batch_size of image loader
            workers (int): num of workers for image loader
            basemodel (str): base model name
            fv_rf (int): receptive filed of feature
            border (int or tuple (h, w)): remove size of feature border
            percent_cb (functional): callable for current status
            debug (bool): debug enable
        Out:
            vp_err (torch.Tensor): vp_err of dataset (useful for threshold calculations)
        '''
        from fastprogress.fastprogress import progress_bar
        import torch
        from sklearn.covariance import LedoitWolf
        from scipy.spatial.distance import mahalanobis
        all_fv_dict = self.get_ori_fv(ibll, tfms, bs=bs, workers=workers,
                                      basemodel=basemodel, fv_rf=fv_rf,
                                      border=border,
                                      percent_cb=percent_cb, debug=debug)
        ok_fv = all_fv_dict['ok']
        mean = torch.mean(ok_fv, dim=0).cpu().numpy()
        cov = LedoitWolf().fit(ok_fv.cpu().numpy()).covariance_
        cov_inv = np.linalg.inv(cov)
        ok_vp_err = np.array([mahalanobis(fv, mean, cov_inv) for fv in progress_bar(ok_fv)])
        ok_vp_err = torch.from_numpy(ok_vp_err)
        ng_fv = all_fv_dict['ng']
        ng_vp_err = None
        if ng_fv is not None:
            ng_vp_err = np.array([mahalanobis(fv, mean, cov_inv) for fv in progress_bar(ng_fv)])
            ng_vp_err = torch.from_numpy(ng_vp_err)

        self.train_mean = torch.from_numpy(mean).to(self.device)
        cov_inv = cov_inv.astype(np.float32)
        self.train_cov = torch.from_numpy(cov_inv).to(self.device)
        vp_err = {'ok': ok_vp_err, 'ng': ng_vp_err}
        if debug:
            self.show_err_dist(ok_vp_err, ok_fv, ng_vp_err, ng_fv,
                               fv_rf=fv_rf, prefix='vp_error')
        return vp_err

    def forward_with_flatten_ori_fv(self, ori_fv):
        '''
        In:
            fv (torch.Tensor): N, C
        out:
            err (torch.Tensor): N
        '''
        vp_err = tensor_mahalanobis(ori_fv, self.train_mean, self.train_cov)
        return vp_err

    def load_from_model_info(self, model_info):
        self.train_mean = model_info['train_mean'].to(self.device)
        self.train_cov = model_info['train_cov'].to(self.device)

    def save_to_model_info(self, model_info):
        model_info['train_mean'] = self.train_mean.cpu()
        model_info['train_cov'] = self.train_cov.cpu()

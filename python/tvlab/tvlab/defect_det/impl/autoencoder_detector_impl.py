import os
import os.path as osp
import cv2
import numpy as np
from tqdm.auto import trange, tqdm
from tvlab.defect_det.cnn_feature_based_detector import CnnFeatureBasedDefectDetector
from tvlab.utils.basic import dump_cuda_mem


def get_ae(c):
    from torch import nn
    class AutoEncoder(nn.Module):
        def __init__(self, in_c):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(in_c, in_c//2),
                                     nn.BatchNorm1d(in_c//2),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(in_c//2, in_c//4),
                                     nn.BatchNorm1d(in_c//4),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(in_c//4, in_c//8),
                                     nn.BatchNorm1d(in_c//8),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(in_c//8, in_c//4),
                                     nn.BatchNorm1d(in_c//4),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(in_c//4, in_c//2),
                                     nn.BatchNorm1d(in_c//2),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(in_c//2, in_c),
                                     nn.Tanh())
            self.loss_function = nn.MSELoss()

        def forward(self, x):
            '''
            In:
                x (torch.Tensor): (N,C)

            Out:
                x (torch.Tensor): (N,C)
            '''
            x = self.net(x)
            return x

        def get_loss(self, x):
            '''
            In:
                x (torch.Tensor): (N,C)
                y (torch.Tensor): (N,C)

            Out:
                loss (torch.Tensor): (1,)
            '''
            rec_x = self.forward(x)
            return self.loss_function(x, rec_x)

        def get_err(self, x):
            '''
            In:
                x (torch.Tensor): (N,C)
            Out:
                err (torch.Tensor): (N,)
            '''
            rec_x = self.forward(x)
            err = (rec_x - x).abs()
            err = err.norm(dim=1)
            return err

    return AutoEncoder(c)


def get_vp(c):
    from torch import nn
    class ValuePredictor(nn.Module):
        def __init__(self, in_c):
            super().__init__()
            mf_num = max(16, in_c // 16)
            self.net = nn.Sequential(nn.Linear(in_c, mf_num),
                                     nn.BatchNorm1d(mf_num),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(mf_num, mf_num//2),
                                     nn.BatchNorm1d(mf_num//2),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(mf_num//2, 1),
                                     nn.ReLU())
            self.loss_function = nn.L1Loss()

        def forward(self, x):
            '''
            In:
                x (torch.Tensor): (N,C)
            Out:
                y (torch.Tensor): (N,)
            '''
            return self.net(x)

        def get_loss(self, x, y):
            '''
            In:
                x (torch.Tensor): (N,C)
            Out:
                y (torch.Tensor): (N,)
            '''
            x = self.forward(x)

            x = x.view(-1, 1)
            y = y.view(-1, 1)
            return self.loss_function(x, y)

        def get_err(self, x, y):
            '''
            In:
                x (torch.Tensor): (N,C)
            Out:
                y (torch.Tensor): (N,)
            '''
            x = self.forward(x)

            x = x.flatten()
            y = y.flatten()
            err = x - y
            return err

    return ValuePredictor(c)


class MsAeDefectDetectorImpl(CnnFeatureBasedDefectDetector):
    def __init__(self, tile_shape=None, overlap=0.5, border_coef=1.618, device='cuda'):
        '''
        Using Mahalanobis distance of pre-trained deep feature as anomaly score.

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
        CnnFeatureBasedDefectDetector.__init__(self, tile_shape=tile_shape,
                                               overlap=overlap,
                                               border_coef=border_coef,
                                               device=device)
        self.ae_model = None
        self.vp_model = None

    def build_ae_model(self, in_c):
        return get_ae(in_c)

    def build_vp_model(self, in_c):
        return get_vp(in_c)

    def train_autoencoder(self, normalized_fv, bs=4096, epochs=20, noise_std=0.2,
                         fv_rf=16, percent_cb=None, debug=True, debug_ok_pct=99.9,
                         early_stop_patience=-1, early_stop_min_delta=0.001):
        '''
        In:
            normalized_fv (torch.Tensor): (N, C)

        Out:
            ae_err (torch.Tensor): (N,)
        '''
        import torch
        from torch import optim
        torch.cuda.empty_cache()
        dump_cuda_mem()

        norm_ok_fv = normalized_fv['ok']
        if self.ae_model is not None:
            ae_model = self.ae_model
        else:
            ae_model = self.build_ae_model(norm_ok_fv.shape[1])

        ae_model = ae_model.to(self.device).train()

        optimizer = optim.Adam(ae_model.parameters(), lr=1e-3)

        total = norm_ok_fv.shape[0]
        t_bar = trange(epochs, desc='train ae')
        loss_history = []
        for epoch in t_bar:
            train_loss = []
            shuffle_idxs = torch.randperm(total)
            for i in range(0, total, bs):
                idxs = shuffle_idxs[i:i+bs]
                fv_b = norm_ok_fv[idxs]
                fv_b = fv_b.to(self.device)
                fv_b += torch.randn_like(fv_b) * noise_std
                fv_b.clamp_(-1.0, 1.0)
                optimizer.zero_grad()
                loss = ae_model.get_loss(fv_b)
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())
            mean_loss = sum(train_loss)/len(train_loss)
            loss_history.append(mean_loss)
            t_bar.set_postfix_str(f'loss: {mean_loss:.6f}')
            self._status_cb('training_ae', int(100*epoch/epochs), percent_cb)
            if early_stop_patience > 0 and epoch > early_stop_patience:
                delta = loss_history[-early_stop_patience-1] - min(loss_history[-early_stop_patience:])
                if delta < early_stop_min_delta:
                    print('train ae early stop!')
                    break

        ae_model = ae_model.eval()
        self.ae_model = ae_model

        with torch.no_grad():
            ok_ae_err = [ae_model.get_err(norm_ok_fv[i:i+bs].to(self.device)).cpu()
                      for i in trange(0, total, bs, desc='get_ok_ae_err')]
        # N,
        ok_ae_err = torch.cat(ok_ae_err)

        norm_ng_fv = normalized_fv['ng']
        ng_ae_err = None
        if norm_ng_fv is not None:
            total = norm_ng_fv.shape[0]
            with torch.no_grad():
                ng_ae_err = [ae_model.get_err(norm_ng_fv[i:i+bs].to(self.device)).cpu()
                             for i in trange(0, total, bs, desc='get_ng_ae_err')]

            ng_ae_err = torch.cat(ng_ae_err)

        if debug:
            self.show_err_dist(ok_ae_err, norm_ok_fv, ng_ae_err, norm_ng_fv,
                               fv_rf=fv_rf, prefix='ae_error', debug_ok_pct=debug_ok_pct)

        return {'ok': ok_ae_err, 'ng': ng_ae_err}

    def train_valuepredictor(self, normalized_fv, values, bs=4096, epochs=20,
                             noise_std=0.2, train_ok_pct=99.9, fv_rf=16,
                             percent_cb=None, debug=True,
                             early_stop_patience=-1, early_stop_min_delta=0.01):
        '''
        In:
            normalized_fv (torch.Tensor): (N, C)
            ae_err (torch.Tensor): (N,)

        Out:
            vp_err (torch.Tensor): (N,)
        '''
        import torch
        from torch import optim
        torch.cuda.empty_cache()
        dump_cuda_mem()

        norm_ok_fv = normalized_fv['ok']

        if self.vp_model is not None:
            vp_model = self.vp_model
        else:
            vp_model = self.build_vp_model(norm_ok_fv.shape[1])
        vp_model = vp_model.to(self.device).train()

        optimizer = optim.Adam(vp_model.parameters(), lr=1e-3)

        total = norm_ok_fv.shape[0]
        ok_values = values['ok']
        ok_threshold = np.percentile(ok_values.cpu().numpy(), train_ok_pct)
        pick_ok_idxs = torch.where(ok_values < ok_threshold)[0]
        train_ok_total = len(pick_ok_idxs)

        t_bar = trange(epochs, desc='train vp')
        loss_history = []
        for epoch in t_bar:
            train_loss = []
            shuffle_idxs = torch.randperm(train_ok_total)
            for i in range(0, train_ok_total, bs):
                idxs = shuffle_idxs[i:i+bs]
                idxs = pick_ok_idxs[idxs]
                fv_b = norm_ok_fv[idxs]
                ae_err = ok_values[idxs]
                fv_b = fv_b.to(self.device)
                ae_err = ae_err.to(self.device)
                fv_b += torch.randn_like(fv_b) * noise_std
                fv_b.clamp_(-1.0, 1.0)

                optimizer.zero_grad()
                loss = vp_model.get_loss(fv_b, ae_err)
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())
            mean_loss = sum(train_loss)/len(train_loss)
            loss_history.append(mean_loss)
            t_bar.set_postfix_str(f'loss: {mean_loss:.6f}')
            self._status_cb('training_vp', int(100*epoch/epochs), percent_cb)
            if early_stop_patience > 0 and epoch > early_stop_patience:
                delta = loss_history[-early_stop_patience-1] - min(loss_history[-early_stop_patience:])
                if delta < early_stop_min_delta:
                    print('train vp early stop!')
                    break

        vp_model = vp_model.eval()
        self.vp_model = vp_model

        with torch.no_grad():
            ok_vp_err = [vp_model.get_err(norm_ok_fv[i:i+bs].to(self.device),
                                          ok_values[i:i+bs].to(self.device)).cpu()
                      for i in trange(0, total, bs, desc='get_ok_vp_err')]

        ok_vp_err = torch.cat(ok_vp_err)

        norm_ng_fv = normalized_fv['ng']
        ng_vp_err = None
        if norm_ng_fv is not None:
            total = norm_ng_fv.shape[0]
            ng_values = values['ng']
            with torch.no_grad():
                ng_vp_err = [vp_model.get_err(norm_ng_fv[i:i+bs].to(self.device),
                                              ng_values[i:i+bs].to(self.device)).cpu()
                             for i in trange(0, total, bs, desc='get_ng_vp_err')]

            ng_vp_err = torch.cat(ng_vp_err)

        if debug:
            self.show_err_dist(ok_vp_err, norm_ok_fv, ng_vp_err, norm_ng_fv,
                               fv_rf=fv_rf, prefix='vp_error')

        return {'ok': ok_vp_err, 'ng': ng_vp_err}

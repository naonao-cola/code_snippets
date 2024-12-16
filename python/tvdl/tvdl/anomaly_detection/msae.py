'''
Copyright (C) 2023 TuringVision

'''
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from ..common import AnomalyDetectionModelBase

__all__ = ['MSAE']


class MSAE(AnomalyDetectionModelBase):
    ''' MSAE (MultiScaleAutoEncoder)

    Usage:
        train_dl, valid_dl = ill.dataloader(...)
        model = MSAE()
        trainer.fit(model, train_dl)

    Model input:
        img (torch.float32): (n,c,h,w)

    Model output (torch.float32):
        - anomaly_map ((FloatTensor[N, H, W]): the predicted anomaly map (0 ~ 1.0)
    '''
    def __init__(self,
                 backbone: str="densenet161",
                 img_c: int=3,
                 lr: float=0.001,
                 noise_std: float=0.2,
                 backbone_out_indices=(3,),
                 batch_size: int=8192,
                 replay_buffer_size: int=8192*16,
                 ae_epochs: int=None,
                 ae_steps: int=8,
                 vp_steps: int=16,
                 **kwargs) -> None:
        '''
        backbone (str): timm backbone
            use this code to show all supported backbones:
            import timm
            timm.list_models(pretrained=True)
        img_c (int): img channel 1 or 3, or other positive integer
            Number of input (color) channels.
            eg: 1-channel (gray image); 25-channel image (maybe satellite image)
        lr (float): init learning rate
        noise_std (float): feature noise std
        backbone_out_indices (Tuple[int, ...]): timm backbone feature output indices
        batch_size (int): real batch size for ae/vp training
        replay_buffer_size (int): cpu memory cache size (number of feature)
        ae_epochs (int): number of epochs for ae training (default is max_epochs / 4)
                |--- ae training ---|------ vp training ----|
                |---> ae_epochs <---|
                |------------> max_epochs <-----------------|
        ae_steps (int): ae update step for single training step
        vp_steps (int): vp update step for single training step
        '''
        super().__init__(**kwargs)
        self.hparams.update({'backbone': backbone,
                             'img_c': img_c,
                             'lr': lr,
                             'noise_std': noise_std,
                             'backbone_out_indices': backbone_out_indices,
                             'batch_size': batch_size,
                             'replay_buffer_size': replay_buffer_size,
                             'ae_epochs': ae_epochs,
                             'ae_steps': ae_steps,
                             'vp_steps': vp_steps,
                            })
        self.build_model()
        self.automatic_optimization = False

    def build_model(self):
        import timm
        p = self.hparams
        self.backbone = timm.create_model(p.backbone,
                                          features_only=True,
                                          in_chans=p.img_c,
                                          pretrained=p.pretrained,
                                          out_indices=p.backbone_out_indices)
        self.feature_ch = sum(self.backbone.feature_info.channels())
        self.ae = AutoEncoder(self.feature_ch)
        self.vp = ValuePredictor(self.feature_ch)
        self.register_buffer('fv_mean', torch.zeros(self.feature_ch))
        self.register_buffer('fv_std', torch.zeros(self.feature_ch))

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()

    def on_train_start(self):
        from fastprogress.fastprogress import progress_bar as pbar
        super().on_train_start()

        if self.hparams.ae_epochs is None:
            self.hparams.ae_epochs = max(1, self.trainer.max_epochs // 4)

        if not hasattr(self, 'fv_sum'):
            self.fv_sum = torch.zeros(self.feature_ch, device=self.device)
            self.fv2_sum = torch.zeros(self.feature_ch, device=self.device)
            self.fv_size_n = 0
        self.replay_buffer = ReplayBuffer(self.hparams.batch_size,
                                          self.hparams.replay_buffer_size,
                                          self.device)

        loader = self.train_dataloader()
        with torch.no_grad():
            for x, _ in pbar(loader, leave=False):
                fv, _ = self.get_ori_fv(x.to(self.device))
                self.increment_mean_and_std(fv)
                fv = self.norm_fv(fv)
                self.replay_buffer.push(fv)

    def on_train_end(self):
        super().on_train_end()
        self.replay_buffer.clear()

    def increment_mean_and_std(self, f):
        n, c = f.shape
        self.fv_sum += f.sum(dim = 0)
        self.fv2_sum += (f ** 2).sum(dim = 0)
        self.fv_size_n += n
        self.fv_mean = self.fv_sum / self.fv_size_n
        self.fv_std = torch.sqrt(self.fv2_sum / self.fv_size_n -
                                2 * self.fv_sum * self.fv_mean / self.fv_size_n +
                                self.fv_mean ** 2)

    def get_ori_fv(self, x):
        with torch.no_grad():
            fv_list = self.backbone(x)
            fv = fv_list[0]

            if len(fv_list) > 1:
                n, c, h, w = fv.shape
                scale_fv_list = [fv]
                for fv_i in fv_list[1:]:
                    fv_i = F.interpolate(fv_i, size=(h, w), mode='bilinear', align_corners=False)
                    scale_fv_list.append(fv_i)
                fv = torch.cat(scale_fv_list, dim=1)

            n, c, h, w = fv.shape
            fv = fv.permute((0,2,3,1)).reshape(-1, c)
        return fv, (n, c, h, w)

    def norm_fv(self, fv):
        return torch.tanh((fv - self.fv_mean) / self.fv_std)

    def forward(self, x, vp_disable=False):
        fv, (n, c, h, w) = self.get_ori_fv(x)
        fv = self.norm_fv(fv)
        ae_err = self.ae.get_err(fv)
        if vp_disable:
            ae_err = ae_err.reshape(n, h, w)
            return ae_err

        vp_err = self.vp.get_err(fv, ae_err)
        vp_err = vp_err.reshape(n, h, w)
        return vp_err.abs()

    def on_epoch_start(self):
        if self.current_epoch == self.hparams.ae_epochs:
            self.ae.eval()

    def training_step(self, batch, batch_nb, optimizer_idx):
        x, _ = batch
        opt_ae, opt_vp = self.optimizers()

        with torch.no_grad():
            x, _ = batch
            fv, _ = self.get_ori_fv(x)
            fv = self.norm_fv(fv)
            self.replay_buffer.push(fv)

        if self.current_epoch < self.hparams.ae_epochs:
            # ae
            for _ in range(self.hparams.ae_steps):
                with torch.no_grad():
                    fv = self.replay_buffer.sample()

                noise_f = fv + torch.randn_like(fv) * self.hparams.noise_std
                noise_f.clamp_(-1.0, 1.0)
                opt_ae.zero_grad()
                ae_loss = self.ae.get_loss(noise_f)
                self.manual_backward(ae_loss)
                opt_ae.step()
                logs = {'ae_loss': ae_loss}
        else:
            # vp
            for _ in range(self.hparams.vp_steps):
                with torch.no_grad():
                    fv = self.replay_buffer.sample()
                    ae_err = self.ae.get_err(fv)

                noise_f = fv + torch.randn_like(fv) * self.hparams.noise_std
                noise_f.clamp_(-1.0, 1.0)
                opt_vp.zero_grad()
                vp_loss = self.vp.get_loss(noise_f, ae_err)
                self.manual_backward(vp_loss)
                opt_vp.step()
                logs = {'vp_loss': vp_loss}
        self.log_dict(logs, prog_bar=True)

    def predict(self, x, vp_disable=False, v_min=0, v_max=10.0, border_coef=1.618, border_ratio=0.05):
        '''
        x (torch.tensor (n, c, h, w)): input image

        v_min (float): clip v_min
        v_max (float: clip v_max
        border_coef (float): Image boundary weakening factor, 1.618 means significant weakening,
                                 1.0 means no weakening
        border_ratio (0.05): border_size / image_size

        out: amap (numpy.ndarry NxHxW): 0 ~ 1.0
        '''
        x = x.to(self.device)
        amap = self.forward(x, vp_disable=vp_disable).cpu().numpy()
        amap = MSAE.post_process(amap, v_min=v_min, v_max=v_max,
                border_coef=border_coef, border_ratio=border_ratio)
        return amap

    def configure_optimizers(self):
        from torch import optim
        opt_ae = optim.Adam(self.ae.parameters(), lr=self.hparams.lr)
        opt_vp = optim.Adam(self.vp.parameters(), lr=self.hparams.lr)
        return opt_ae, opt_vp


class ReplayBuffer():
    def __init__(self, bs, max_size, device):
        self.bs = bs
        self.max_size = max_size
        self.fv_buffer = []
        self.device = device

    def clear(self):
        self.fv_buffer.clear()

    def push(self, fv):
        fv = fv.detach().cpu()
        for fvi in fv:
            if len(self.fv_buffer) > self.max_size:
                i = random.randint(0, len(self.fv_buffer) - 1)
                self.fv_buffer[i] = fvi
            else:
                self.fv_buffer.append(fvi)

    def sample(self):
        select_size = min(self.bs, len(self.fv_buffer))
        select_idxs = random.sample(range(len(self.fv_buffer)), select_size)
        fv = torch.stack([self.fv_buffer[i] for i in select_idxs], dim=0)
        return fv.to(self.device)


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
        x = self.net(x)
        return x

    def get_loss(self, x):
        rec_x = self.forward(x)
        return self.loss_function(x, rec_x)

    def get_err(self, x):
        rec_x = self.forward(x)
        err = (rec_x - x).abs()
        err = err.norm(dim=1)
        return err


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
        return self.net(x)

    def get_loss(self, x, y):
        x = self.forward(x)

        x = x.view(-1, 1)
        y = y.view(-1, 1)
        return self.loss_function(x, y)

    def get_err(self, x, y):
        x = self.forward(x)

        x = x.flatten()
        y = y.flatten()
        err = x - y
        return err

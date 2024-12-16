'''
Copyright (C) 2023 TuringVision

Implementation of: Student-Teacher Feature Pyramid Matching for
Unsupervised Anomaly Detection implementation
'''
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from ..common import AnomalyDetectionModelBase

__all__ = ['FPS']


class FPS(AnomalyDetectionModelBase):
    ''' FPS (Feature Pyramid School)

    Usage:
        train_dl, valid_dl = ill.dataloader(...)
        model = FPS()
        trainer.fit(model, train_dl)

    Model input:
        img (torch.float32): (n,c,h,w)

    Model output (torch.float32):
        - anomaly_map ((FloatTensor[N, H, W]): the predicted anomaly map (0 ~ 1.0)
    '''
    def __init__(self,
                 backbone: str="resnet18",
                 img_c: int=3,
                 lr: float=1.0,
                 backbone_out_indices=(1,2,3),
                 copy_bn: bool=False,
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
        backbone_out_indices (Tuple[int, ...]): timm backbone feature output indices
        copy_bn (bool): student bn parameters copy from teacher
        '''
        super().__init__()
        self.hparams.update({'backbone': backbone,
                             'img_c': img_c,
                             'lr': lr,
                             'backbone_out_indices': backbone_out_indices,
                             'copy_bn': copy_bn,
                            })
        self.build_model()

    def build_model(self):
        import timm
        p = self.hparams
        self.teacher = timm.create_model(p.backbone,
                                         features_only=True,
                                         in_chans=p.img_c,
                                         pretrained=True,
                                         out_indices=p.backbone_out_indices)
        self.student = timm.create_model(p.backbone,
                                         features_only=True,
                                         in_chans=p.img_c,
                                         pretrained=False,
                                         out_indices=p.backbone_out_indices)
        for param in self.teacher.parameters():
            param.requires_grad = False
        if p.copy_bn:
            import copy
            state_dict = copy.deepcopy(self.teacher.state_dict())
            for key in list(state_dict.keys()):
                if 'bn' not in key:
                    state_dict.pop(key)
            self.student.load_state_dict(state_dict, strict=False)

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()

    def forward(self, x, return_amap=True):
        tx = self.teacher(x)
        sx = self.student(x)
        tx_norm = [f / torch.norm(f, p=2, dim=1, keepdim=True) for f in tx]
        sx_norm = [f / torch.norm(f, p=2, dim=1, keepdim=True) for f in sx]
        if not return_amap:
            return tx_norm, sx_norm

        n, _, h, w = x.shape
        out_size = (h, w)
        anomaly_map = None
        for txi, sxi in zip(tx_norm, sx_norm):
            #pairwise_distance not supported in onnx
            #scores = F.pairwise_distance(txi, sxi, p=2, keepdim=True) ** 2
            scores = ((txi - sxi).abs().norm(dim=1, p=2, keepdim=True)) ** 2
            a_map = F.interpolate(scores, size=out_size, mode='bilinear', align_corners=False)
            a_map = a_map[:,0,:,:]
            if anomaly_map is None:
                anomaly_map = a_map
            else:
                anomaly_map *= a_map
        return anomaly_map

    def training_step(self, batch, batch_nb):
        x, _ = batch
        tx, sx = self.forward(x, False)

        loss = 0
        for txi, sxi in zip(tx, sx):
            h, w = txi.shape[-2:]
            loss_w = 1.0 / (w * h)
            loss += loss_w * F.mse_loss(txi, sxi, reduction='sum')
        self.log_dict({'loss': loss})
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        tx, sx = self.forward(x, False)

        loss = 0
        for txi, sxi in zip(tx, sx):
            h, w = txi.shape[-2:]
            loss_w = 1.0 / (w * h)
            loss += loss_w * F.mse_loss(txi, sxi, reduction='sum')
        self.log_dict({'val_loss': loss})
        return {'val_loss': loss}

    def predict(self, x, v_min=0, v_max=1.0, border_coef=1.618, border_ratio=0.05):
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
        amap = self.forward(x).cpu().numpy()
        amap = FPS.post_process(amap, v_min=v_min, v_max=v_max,
                border_coef=border_coef, border_ratio=border_ratio)
        return amap

    def configure_optimizers(self):
        from torch import optim
        optimizer = optim.Adam(self.student.parameters(), lr=self.hparams.lr, betas=(0.5, 0.99))
        return optimizer

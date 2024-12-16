'''
Copyright (C) 2023 TuringVision

Implementation of U2Net
'''
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.ops import box_iou
from ..common import SegmentationModelBase

__all__ = ['U2Net']


class U2Net(SegmentationModelBase):
    '''
        Basic model for semantic segmentation with U2Net architecture.
        see: https://github.com/xuebinqin/U-2-Net

    Usage:
        train_dl, valid_dl = ibll.dataloader(...)
        model = U2Net(ibll.labelset())
        trainer.fit(model, train_dl, valid_dl)

    Model input:
        img (torch.float32): (n,c,h,w)
        target (dict):
            - masks (UInt8Tensor[N, H, W]): The segmentation masks for each one of the objects
            - boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with values of x
            between 0 and W and values of y between 0 and H
            - labels (Int64Tensor[N]): the class label for each ground-truth box

    Model output (torch.float32):
        - masks ((FloatTensor[N, C, H, W]): the predicted masks (0 ~ 1.0) for each one of category
    '''
    def __init__(self, classes,
                 img_c: int=3,
                 lr: float=1e-3,
                 model_cfg: str='lite',
                 ):
        '''
        classes (Tuple[str, ...]):
        img_c (int): img channel 1 or 3, or other positive integer
            Number of input (color) channels.
            eg: 1-channel (gray image); 25-channel image (maybe satellite image)
        lr (float): init learning rate
        model_cfg (str): one of ('full', 'lite', 'tiny')
        '''
        super().__init__()
        # self.save_hyperparameters() not work after cython compile

        self.hparams.update({'classes': classes,
                             'img_c': img_c,
                             'lr': lr,
                             'model_cfg': model_cfg,
                            })
        self.num_classes = len(classes)
        self.build_model()

    def get_model_cfg(self):
        in_c = self.hparams.img_c
        full_cfg = {
            # cfgs for building RSUs and sides
            # {stage : [name, (height(L), in_ch, mid_ch, out_ch, dilated), side]}
            'stage1': ['En_1', (7, in_c, 32, 64), -1],
            'stage2': ['En_2', (6, 64, 32, 128), -1],
            'stage3': ['En_3', (5, 128, 64, 256), -1],
            'stage4': ['En_4', (4, 256, 128, 512), -1],
            'stage5': ['En_5', (4, 512, 256, 512, True), -1],
            'stage6': ['En_6', (4, 512, 256, 512, True), 512],
            'stage5d': ['De_5', (4, 1024, 256, 512, True), 512],
            'stage4d': ['De_4', (4, 1024, 128, 256), 256],
            'stage3d': ['De_3', (5, 512, 64, 128), 128],
            'stage2d': ['De_2', (6, 256, 32, 64), 64],
            'stage1d': ['De_1', (7, 128, 16, 64), 64],
        }

        lite_cfg = {
            # cfgs for building RSUs and sides
            # {stage : [name, (height(L), in_ch, mid_ch, out_ch, dilated), side]}
            'stage1': ['En_1', (7, in_c, 16, 64), -1],
            'stage2': ['En_2', (6, 64, 16, 64), -1],
            'stage3': ['En_3', (5, 64, 16, 64), -1],
            'stage4': ['En_4', (4, 64, 16, 64), -1],
            'stage5': ['En_5', (4, 64, 16, 64, True), -1],
            'stage6': ['En_6', (4, 64, 16, 64, True), 64],
            'stage5d': ['De_5', (4, 128, 16, 64, True), 64],
            'stage4d': ['De_4', (4, 128, 16, 64), 64],
            'stage3d': ['De_3', (5, 128, 16, 64), 64],
            'stage2d': ['De_2', (6, 128, 16, 64), 64],
            'stage1d': ['De_1', (7, 128, 16, 64), 64],
        }

        tiny_cfg = {
            # cfgs for building RSUs and sides
            # {stage : [name, (height(L), in_ch, mid_ch, out_ch, dilated), side]}
            'stage1': ['En_1', (6, in_c, 16, 32), -1],
            'stage2': ['En_2', (5, 32, 16, 32), -1],
            'stage3': ['En_3', (4, 32, 16, 32, True), 32],
            'stage4': ['En_4', (4, 32, 16, 32, True), 32],
            'stage3d': ['De_3', (4, 64, 16, 32, True), 32],
            'stage2d': ['De_2', (5, 64, 16, 32), 32],
            'stage1d': ['De_1', (6, 64, 16, 32), 32],
        }

        cfg_map = {'full': full_cfg, 'lite': lite_cfg, 'tiny': tiny_cfg}
        return cfg_map[self.hparams.model_cfg]

    def build_model(self):
        cfg = self.get_model_cfg()
        self.u2net = U2NETModel(cfgs=cfg, out_ch=self.num_classes)

    def target_convert(self, images, targets):
        '''
        convert instance segmentation masks to semantic segmentation masks
        '''
        device = images.device
        bs, _, img_h, img_w = images.shape

        masks = torch.zeros((bs, self.num_classes, img_h, img_w),
                            dtype=torch.float32, device=device)
        for i, target in enumerate(targets):
            for l, mask in zip(target['labels'], target['masks']):
                h, w = mask.shape
                masks[i, l-1, :h, :w] = torch.max(masks[i, l-1, :h, :w], mask.type(torch.float32))

        return masks

    def loss(self, outs, targets):
        bce_loss = nn.BCELoss(reduction='mean')
        loss = 0.0
        for d in outs:
            loss += bce_loss(d, targets)
        return loss

    def forward(self, images, targets=None):
        outs = self.u2net(images)
        if targets:
            targets = self.target_convert(images, targets)
            loss = self.loss(outs, targets)
            return outs[0], loss
        return outs[0]

    def predict(self, x, x_info,
                mask_threshold=0.2,
                area_threshold=25,
                blob_filter_func=None,
                polygon_tfm=None):
        '''
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
        x = x.to(self.device)
        outputs = self.forward(x).cpu().numpy()
        return U2Net.post_process(outputs, x_info, self.hparams.classes,
                                  mask_threshold, area_threshold,
                                  blob_filter_func, polygon_tfm)

    def training_step(self, batch, batch_nb):
        if isinstance(self.train_dataloader(), tuple):
            images, targets = [], []
            for batch_i in batch:
                bx, by = batch_i
                images.append(bx)
                targets += by
            images = torch.cat(images, dim=0)
        else:
            images, targets = batch

        targets = [{k: v for k, v in t.items()} for t in targets]

        out, loss = self.forward(images, targets)
        self.log_dict({"loss": loss})
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        out, loss = self.forward(images, targets)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        loss_val = torch.stack([x['val_loss'] for x in outputs]).mean()
        log_dict = {'val_loss': loss_val}
        self.log_dict(log_dict, prog_bar=True)

    def configure_optimizers(self):
        from torch import optim
        optimizer = optim.SGD(self.parameters(), lr=self.hparams.lr,
                              momentum=0.9, weight_decay=0.005)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        return [optimizer], [scheduler]


def _upsample_like(x, size):
    return nn.Upsample(size=size, mode='bilinear', align_corners=False)(x)


def _size_map(x, height):
    import math
    # {height: size} for Upsample
    size = list(x.shape[-2:])
    sizes = {}
    for h in range(1, height):
        sizes[h] = size
        size = [math.ceil(w / 2.0) for w in size]
    return sizes


class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dilate=1):
        super(REBNCONV, self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1 * dilate, dilation=1 * dilate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu_s1(self.bn_s1(self.conv_s1(x)))


class RSU(nn.Module):
    def __init__(self, name, height, in_ch, mid_ch, out_ch, dilated=False):
        super(RSU, self).__init__()
        self.name = name
        self.height = height
        self.dilated = dilated
        self._make_layers(height, in_ch, mid_ch, out_ch, dilated)

    def forward(self, x):
        sizes = _size_map(x, self.height)
        x = self.rebnconvin(x)

        # U-Net like symmetric encoder-decoder structure
        def unet(x, height=1):
            if height < self.height:
                x1 = getattr(self, f'rebnconv{height}')(x)
                if not self.dilated and height < self.height - 1:
                    x2 = unet(getattr(self, 'downsample')(x1), height + 1)
                else:
                    x2 = unet(x1, height + 1)

                x = getattr(self, f'rebnconv{height}d')(torch.cat((x2, x1), 1))
                return _upsample_like(x, sizes[height - 1]) if not self.dilated and height > 1 else x
            else:
                return getattr(self, f'rebnconv{height}')(x)

        return x + unet(x)

    def _make_layers(self, height, in_ch, mid_ch, out_ch, dilated=False):
        self.add_module('rebnconvin', REBNCONV(in_ch, out_ch))
        self.add_module('downsample', nn.MaxPool2d(2, stride=2, ceil_mode=True))

        self.add_module(f'rebnconv1', REBNCONV(out_ch, mid_ch))
        self.add_module(f'rebnconv1d', REBNCONV(mid_ch * 2, out_ch))

        for i in range(2, height):
            dilate = 1 if not dilated else 2 ** (i - 1)
            self.add_module(f'rebnconv{i}', REBNCONV(mid_ch, mid_ch, dilate=dilate))
            self.add_module(f'rebnconv{i}d', REBNCONV(mid_ch * 2, mid_ch, dilate=dilate))

        dilate = 2 if not dilated else 2 ** (height - 1)
        self.add_module(f'rebnconv{height}', REBNCONV(mid_ch, mid_ch, dilate=dilate))


class U2NETModel(nn.Module):
    def __init__(self, cfgs, out_ch):
        super(U2NETModel, self).__init__()
        self.out_ch = out_ch
        self._make_layers(cfgs)

    def forward(self, x):
        sizes = _size_map(x, self.height)
        maps = []  # storage for maps

        # side saliency map
        def unet(x, height=1):
            if height < self.height:
                x1 = getattr(self, f'stage{height}')(x)
                x2 = unet(getattr(self, 'downsample')(x1), height + 1)
                x = getattr(self, f'stage{height}d')(torch.cat((x2, x1), 1))
                side(x, height)
                return _upsample_like(x, sizes[height - 1]) if height > 1 else x
            else:
                x = getattr(self, f'stage{height}')(x)
                side(x, height)
                return _upsample_like(x, sizes[height - 1])

        def side(x, h):
            # side output saliency map (before sigmoid)
            x = getattr(self, f'side{h}')(x)
            x = _upsample_like(x, sizes[1])
            maps.append(x)

        def fuse():
            # fuse saliency probability maps
            maps.reverse()
            x = torch.cat(maps, 1)
            x = getattr(self, 'outconv')(x)
            maps.insert(0, x)
            return [torch.sigmoid(x) for x in maps]

        unet(x)
        maps = fuse()
        return maps

    def _make_layers(self, cfgs):
        self.height = int((len(cfgs) + 1) / 2)
        self.add_module('downsample', nn.MaxPool2d(2, stride=2, ceil_mode=True))
        for k, v in cfgs.items():
            # build rsu block
            self.add_module(k, RSU(v[0], *v[1]))
            if v[2] > 0:
                # build side layer
                self.add_module(f'side{v[0][-1]}', nn.Conv2d(v[2], self.out_ch, 3, padding=1))
        # build fuse layer
        self.add_module('outconv', nn.Conv2d(int(self.height * self.out_ch), self.out_ch, 1))

'''
Copyright (C) 2023 TuringVision

Implementation of UNet
'''
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.ops import box_iou
from ..common import SegmentationModelBase

__all__ = ['UNet']


class UNet(SegmentationModelBase):
    '''
        Basic model for semantic segmentation with UNet architecture.

    Usage:
        train_dl, valid_dl = ibll.dataloader(...)
        model = UNet(ibll.labelset())
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
                 num_layers: int=5,
                 features_start: int=64,
                 bilinear: bool=True):
        '''
        classes (Tuple[str, ...]):
        img_c (int): img channel 1 or 3, or other positive integer
            Number of input (color) channels.
            eg: 1-channel (gray image); 25-channel image (maybe satellite image)
        lr (float): init learning rate
        num_layers: Number of layers in each side of U-net (default 5)
        features_start: Number of features in first layer (default 64)
        bilinear: Whether to use bilinear interpolation or transposed convolutions for upsampling.
        '''
        super().__init__()
        # self.save_hyperparameters() not work after cython compile

        self.hparams.update({'classes': classes,
                             'img_c': img_c,
                             'lr': lr,
                             'num_layers': num_layers,
                             'features_start': features_start,
                             'bilinear': bilinear
                            })
        self.num_classes = len(classes)
        self.build_model()

    def build_model(self):
        p = self.hparams

        layers = [DoubleConv(p.img_c, p.features_start)]

        feats = p.features_start
        for _ in range(p.num_layers - 1):
            layers.append(Down(feats, feats * 2))
            feats *= 2

        for _ in range(p.num_layers - 1):
            layers.append(Up(feats, feats // 2, p.bilinear))
            feats //= 2

        layers.append(nn.Conv2d(feats, self.num_classes, kernel_size=1))

        self.layers = nn.ModuleList(layers)

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

    def unet_forward(self, x):
        xi = [self.layers[0](x)]
        # Down path
        for layer in self.layers[1:self.hparams.num_layers]:
            xi.append(layer(xi[-1]))
        # Up path
        for i, layer in enumerate(self.layers[self.hparams.num_layers:-1]):
            xi[-1] = layer(xi[-1], xi[-2 - i])
        return self.layers[-1](xi[-1])

    def forward(self, images, targets=None):
        x = self.unet_forward(images).sigmoid()
        if targets:
            targets = self.target_convert(images, targets)
            loss = F.binary_cross_entropy(x, targets)
            return x, loss
        return x

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
        return UNet.post_process(outputs, x_info, self.hparams.classes,
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


class DoubleConv(nn.Module):
    """
    [ Conv2d => BatchNorm (optional) => ReLU ] x 2
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """
    Downscale with MaxPool => DoubleConvolution block
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """
    Upsampling (by either bilinear interpolation or transpose convolutions)
    followed by concatenation of feature map from contracting path, followed by DoubleConv.
    """

    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = False):
        super().__init__()
        self.upsample = None
        if bilinear:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(in_ch, in_ch // 2, kernel_size=1),
            )
        else:
            self.upsample = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)

        # Pad x1 to the size of x2
        diff_h = x2.shape[2] - x1.shape[2]
        diff_w = x2.shape[3] - x1.shape[3]

        x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])

        # Concatenate along the channels axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

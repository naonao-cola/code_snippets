'''
Copyright (C) 2023 TuringVision

Implementation of msrf
'''
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import box_iou
from ..common import SegmentationModelBase

__all__ = ['MSRFNet']


class MSRFNet(SegmentationModelBase):
    '''
        Basic model for semantic segmentation with MSRF_NET architecture.
        see: https://arxiv.org/abs/2105.07451

    Usage:
        train_dl, valid_dl = ibll.dataloader(...)
        model = MSRFNet(ibll.labelset())
        trainer.fit(model, train_dl, valid_dl)

    Model input:
        img (torch.float32): (n,c,h,w)
        target (dict):
            - masks (UInt8Tensor[N, H, W]): The segmentation masks for each one of the objects

    Model output (torch.float32):
        - masks ((FloatTensor[N, C, H, W]): the predicted masks (0 ~ 1.0) for each one of category
    '''

    def __init__(self, classes,
                 img_c: int = 3,
                 lr: float = 1e-3,
                 model_cfg: str = 'lite',
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

    def build_model(self):
        model_cfg = self.hparams.model_cfg
        if model_cfg == 'tiny':
            filters = [4, 8, 16, 32]
        elif model_cfg == 'lite':
            filters = [6, 12, 24, 48]
        else:
            filters = [16, 32, 64, 128]
        self.model = MSRF(c_in=self.hparams.img_c,
                          c_out=self.num_classes, c1=filters[0], c2=filters[1], c3=filters[2], c4=filters[3])

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
                masks[i, l - 1, :h,
                      :w] = torch.max(masks[i, l - 1, :h, :w], mask.type(torch.float32))

        return masks

    def gradient_1order(self, x, h_x=None, w_x=None):
        if h_x is None and w_x is None:
            h_x = x.size()[2]
            w_x = x.size()[3]
        r = F.pad(x, (0, 1, 0, 0))[:, :, :, 1:]
        l = F.pad(x, (1, 0, 0, 0))[:, :, :, :w_x]
        t = F.pad(x, (0, 0, 1, 0))[:, :, :h_x, :]
        b = F.pad(x, (0, 0, 0, 1))[:, :, 1:, :]
        xgrad = torch.pow(torch.pow((r - l) * 0.5, 2) +
                          torch.pow((t - b) * 0.5, 2), 0.5)
        return xgrad

    def loss(self, outs, targets):
        loss = 0.0
        bce_loss = nn.BCELoss(reduction='mean')
        for p in range(len(outs)):
            if p == len(outs) - 1:
                targets_gard = self.gradient_1order(targets)
                targets_gard = targets_gard.sum(axis=1).unsqueeze(1)
                targets_gard = (targets_gard != 0).int().float()
                lossp = bce_loss(outs[p], targets_gard)
            else:
                lossp = bce_loss(outs[p], targets)
            loss += lossp
        return loss

    def forward(self, images, targets=None):
        outs = self.model(images)
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
        x: (np.ndarray) inputs
        x_info: (dict) inputs' infomation
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
        return MSRFNet.post_process(outputs, x_info, self.hparams.classes,
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
        targets = self.target_convert(images, targets)
        inter = (2 * out * targets).sum() + 1
        union = (out + targets).sum() + 1
        iou = inter / union
        return {"val_loss": loss, "val_iou": iou}

    def validation_epoch_end(self, outputs):
        loss_val = torch.stack([x['val_loss'] for x in outputs]).mean()
        iou_val = torch.stack([x['val_iou'] for x in outputs]).mean()
        log_dict = {'val_loss': loss_val, "val_iou": iou_val}
        self.log_dict(log_dict, prog_bar=True)

    def configure_optimizers(self):
        from torch import optim
        # optimizer = optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=0.9, weight_decay=0.005)
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, betas=(
            0.9, 0.999), weight_decay=0.001)
        # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=5)
        return [optimizer], [scheduler]


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=2):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, (1, 1), bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, (1, 1), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(
            2, 1, (kernel_size, kernel_size), padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class SiLU(nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))


class Activate(nn.Module):
    def __init__(self, act='relu'):
        super(Activate, self).__init__()
        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        if act == 'silu':
            self.act = SiLU()
        if act == 'lrelu':
            self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.act(x)
        return x


class UpSample(nn.Module):
    def __init__(self, scale=2, mode='bilinear'):
        super(UpSample, self).__init__()
        self.scale = scale
        self.mode = mode

    def forward(self, x):
        size = x.shape[-2:]
        new_size = tuple([size[0] * self.scale, size[1] * self.scale])
        x = F.interpolate(x, size=new_size, mode=self.mode,
                          align_corners=False)  # 上采样
        return x


class CBA(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, down=2, up=1, bias=False, act='silu'):
        super(CBA, self).__init__()
        self.pad = kernel // 2
        self.conv = nn.Conv2d(in_channel, out_channel,
                              (kernel, kernel), padding=self.pad, bias=bias)
        self.BatchNorm = nn.BatchNorm2d(out_channel)
        self.act = Activate(act=act)
        self.resample = None
        if up == 1 and down != 1:
            self.resample = nn.MaxPool2d(kernel_size=down)
        elif up != 1:
            self.resample = UpSample(scale=up)
        if self.resample is not None:
            self.r = True
        else:
            self.r = False

    def forward(self, x):
        if self.r:
            x = self.resample(x)
        x = self.conv(x)
        x = self.BatchNorm(x)
        x = self.act(x)
        return x


class DSDFBlock(nn.Module):
    def __init__(self, in_channel_high, in_channel_low):
        super(DSDFBlock, self).__init__()
        self.low_input = in_channel_low
        self.high_input = in_channel_high
        self.convh1 = CBA(in_channel_high * 1, in_channel_high,
                          3, down=1, up=1, act='silu')
        self.convh2 = CBA(in_channel_high * 2, in_channel_high,
                          3, down=1, up=1, act='silu')
        self.convh3 = CBA(in_channel_high * 3, in_channel_high,
                          3, down=1, up=1, act='silu')
        self.convh4 = CBA(in_channel_high * 4, in_channel_high,
                          3, down=1, up=1, act='silu')
        self.convh5 = CBA(in_channel_high * 5, in_channel_high,
                          3, down=1, up=1, act='silu')
        self.convl1 = CBA(in_channel_low * 1, in_channel_low,
                          3, down=1, up=1, act='silu')
        self.convl2 = CBA(in_channel_low * 2, in_channel_low,
                          3, down=1, up=1, act='silu')
        self.convl3 = CBA(in_channel_low * 3, in_channel_low,
                          3, down=1, up=1, act='silu')
        self.convl4 = CBA(in_channel_low * 4, in_channel_low,
                          3, down=1, up=1, act='silu')
        self.convl5 = CBA(in_channel_low * 5, in_channel_low,
                          3, down=1, up=1, act='silu')
        self.convhl1 = CBA(in_channel_high, in_channel_low,
                           3, down=2, up=1, act='silu')
        self.convhl2 = CBA(in_channel_high, in_channel_low,
                           3, down=2, up=1, act='silu')
        self.convhl3 = CBA(in_channel_high, in_channel_low,
                           3, down=2, up=1, act='silu')
        self.convhl4 = CBA(in_channel_high, in_channel_low,
                           3, down=2, up=1, act='silu')
        self.convlh1 = CBA(in_channel_low, in_channel_high,
                           3, down=1, up=2, act='silu')
        self.convlh2 = CBA(in_channel_low, in_channel_high,
                           3, down=1, up=2, act='silu')
        self.convlh3 = CBA(in_channel_low, in_channel_high,
                           3, down=1, up=2, act='silu')
        self.convlh4 = CBA(in_channel_low, in_channel_high,
                           3, down=1, up=2, act='silu')

    def forward(self, x_high, x_low):
        x_low1 = self.convl1(x_low)
        x_high1 = self.convh1(x_high)
        x_in_high2 = torch.cat((self.convlh1(x_low1), x_high1), dim=1)
        x_in_low2 = torch.cat((self.convhl1(x_high1), x_low1), dim=1)
        x_high2 = self.convh2(x_in_high2)
        x_low2 = self.convl2(x_in_low2)
        x_in_high3 = torch.cat((self.convlh2(x_low2), x_high1, x_high2), dim=1)
        x_in_low3 = torch.cat((self.convhl2(x_high2), x_low1, x_low2), dim=1)
        x_high3 = self.convh3(x_in_high3)
        x_low3 = self.convl3(x_in_low3)
        x_in_high4 = torch.cat(
            (self.convlh3(x_low3), x_high1, x_high2, x_high3), dim=1)
        x_in_low4 = torch.cat(
            (self.convhl3(x_high3), x_low1, x_low2, x_low3), dim=1)
        x_high4 = self.convh4(x_in_high4)
        x_low4 = self.convl4(x_in_low4)
        x_in_high5 = torch.cat(
            (self.convlh4(x_low4), x_high1, x_high2, x_high3, x_high4), dim=1)
        x_in_low5 = torch.cat(
            (self.convhl4(x_high4), x_low1, x_low2, x_low3, x_low4), dim=1)
        x_high5 = self.convh5(x_in_high5) + x_high
        x_low5 = self.convl5(x_in_low5) + x_low
        return x_high5, x_low5


class MSRFSub(nn.Module):
    def __init__(self, in_x1, in_x2, in_x3, in_x4):
        super(MSRFSub, self).__init__()
        self.in_x1 = in_x1
        self.in_x2 = in_x2
        self.in_x3 = in_x3
        self.in_x4 = in_x4
        self.dsdfh1 = DSDFBlock(in_x1, in_x2)
        self.dsdfh2 = DSDFBlock(in_x1, in_x2)
        self.dsdfh3 = DSDFBlock(in_x1, in_x2)
        self.dsdfh4 = DSDFBlock(in_x1, in_x2)
        self.dsdfl1 = DSDFBlock(in_x3, in_x4)
        self.dsdfl2 = DSDFBlock(in_x3, in_x4)
        self.dsdfl3 = DSDFBlock(in_x3, in_x4)
        self.dsdfl4 = DSDFBlock(in_x3, in_x4)
        self.dsdfm1 = DSDFBlock(in_x2, in_x3)
        self.dsdfm2 = DSDFBlock(in_x2, in_x3)

    def forward(self, x1, x2, x3, x4):
        x11, x12 = self.dsdfh1(x1, x2)
        x13, x14 = self.dsdfl1(x3, x4)
        x11, x12 = self.dsdfh2(x11, x12)
        x13, x14 = self.dsdfl2(x13, x14)
        x12, x13 = self.dsdfm1(x12, x13)
        x11, x12 = self.dsdfh3(x11, x12)
        x13, x14 = self.dsdfl3(x13, x14)
        x12, x13 = self.dsdfm2(x12, x13)
        x11, x12 = self.dsdfh4(x11, x12)
        x13, x14 = self.dsdfl4(x13, x14)
        x11 = x11 + x1
        x12 = x12 + x2
        x13 = x13 + x3
        x14 = x14 + x4
        return x11, x12, x13, x14


class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, up=1, down=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, in_c, (3, 3), padding=1, bias=False))
        self.bn1 = nn.BatchNorm2d(in_c)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, (3, 3), padding=1, bias=False))
        self.bn2 = nn.BatchNorm2d(out_c)
        self.act2 = nn.ReLU(inplace=True)
        if in_c == out_c:
            self.cat = True
        else:
            self.cat = False
        self.cat_conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, (3, 3), padding=1, bias=True))
        self.resample = None
        if up == 1 and down != 1:
            self.resample = nn.MaxPool2d(kernel_size=down)
        elif up != 1:
            self.resample = UpSample(scale=up)
        if self.resample is not None:
            self.r = True
        else:
            self.r = False

    def forward(self, x_in):
        x = self.act1(self.bn1(self.conv1(x_in)))
        x = self.act2(self.bn2(self.conv2(x)))
        if self.cat:
            x = x + x_in
        else:
            x = x + self.cat_conv(x_in)
        if self.r:
            x = self.resample(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_c=3, c1=16, c2=32, c3=64, c4=128):
        super(Encoder, self).__init__()
        filters = [c1, c2, c3, c4]
        self.conv_in = nn.Sequential(nn.Conv2d(in_c, filters[0], (3, 3), padding=1, bias=False),
                                     nn.BatchNorm2d(filters[0]), nn.ReLU(
                                         inplace=True),
                                     nn.Conv2d(
                                         filters[0], filters[0], (3, 3), padding=1, bias=False),
                                     nn.BatchNorm2d(filters[0]), nn.ReLU(inplace=True))
        self.conv1 = ResBlock(filters[0], filters[1], down=2)
        self.conv2 = ResBlock(filters[1], filters[2], down=2)
        self.conv3 = ResBlock(filters[2], filters[3], down=2)

    def forward(self, x1):
        x1 = self.conv_in(x1)
        x2 = self.conv1(x1)
        x3 = self.conv2(x2)
        x4 = self.conv3(x3)
        return x1, x2, x3, x4


class Decoder_block(nn.Module):
    def __init__(self, high_c, low_c):
        super(Decoder_block, self).__init__()
        self.up = UpSample(scale=2)
        self.down_conv = nn.Sequential(
            nn.Conv2d(high_c, low_c, (1, 1)), nn.MaxPool2d(2))
        self.flat_conv = nn.Sequential(
            nn.ReLU(inplace=True), nn.Conv2d(low_c, high_c, (1, 1)))
        self.conv1 = nn.Conv2d(high_c + low_c, high_c, (3, 3), padding=1)
        self.catt = ChannelAttention(high_c)
        self.satt = SpatialAttention()
        self.conv2 = nn.Conv2d(2 * high_c + low_c, high_c, (1, 1))
        self.out_conv = nn.Sequential(nn.Conv2d(high_c, high_c, (3, 3), padding=1, bias=False),
                                      nn.BatchNorm2d(high_c), nn.ReLU(
                                          inplace=True),
                                      nn.Conv2d(high_c, high_c, (3, 3),
                                                padding=1, bias=False),
                                      nn.BatchNorm2d(high_c), nn.ReLU(inplace=True))

    def forward(self, x_high, x_low):
        x_f = self.up(torch.sigmoid(self.flat_conv(
            x_low + self.down_conv(x_high)))) * x_high
        x_up = self.up(x_low)
        x_t = self.conv1(torch.cat((x_up, x_high), dim=1))
        x_t = x_t * self.catt(x_t)
        x_t = x_t * self.satt(x_t)
        x_out = self.conv2(torch.cat((x_up, x_t, x_f), dim=1))
        x_out = x_out + self.out_conv(x_out)
        return x_out


class Decoder(nn.Module):
    def __init__(self, c_out=1, c1=16, c2=32, c3=64, c4=128):
        super(Decoder, self).__init__()
        filters = [c1, c2, c3, c4]
        self.outc = c_out
        self.db1 = Decoder_block(c3, c4)
        self.db2 = Decoder_block(c2, c3)
        self.db3 = Decoder_block(c1, c2)
        self.out_conv = nn.Sequential(nn.Conv2d(filters[0] * 2, filters[0], (3, 3), padding=1, bias=False),
                                      nn.BatchNorm2d(filters[0]), nn.ReLU(
                                          inplace=True),
                                      nn.Conv2d(filters[0], c_out, (1, 1)), )
        self.deep_conv2 = nn.Sequential(nn.Conv2d(filters[1], filters[1], (3, 3), padding=1, bias=False),
                                        nn.BatchNorm2d(filters[1]), nn.ReLU(
                                            inplace=True),
                                        nn.Conv2d(filters[1], c_out, (1, 1)), )
        self.deep_conv1 = nn.Sequential(nn.Conv2d(filters[2], filters[2], (3, 3), padding=1, bias=False),
                                        nn.BatchNorm2d(filters[2]), nn.ReLU(
                                            inplace=True),
                                        nn.Conv2d(filters[2], c_out, (1, 1)), )
        self.up1 = UpSample(scale=4)
        self.up2 = UpSample(scale=2)
        self.ssb = SSB(c1=c1, c2=c2, c3=c3, c4=c4)

    def forward(self, x1, x2, x3, x4, gard):
        ssb_out1, ssb_out2 = self.ssb(x1, x2, x3, x4, gard)
        up_flow = self.db1(x3, x4)
        x_out1 = torch.sigmoid(self.up1(self.deep_conv1(up_flow)))
        up_flow = self.db2(x2, up_flow)
        x_out2 = torch.sigmoid(self.up2(self.deep_conv2(up_flow)))
        up_flow = self.db3(x1, up_flow)
        up_flow = torch.cat((ssb_out1, up_flow), dim=1)
        x_out3 = torch.sigmoid(self.out_conv(up_flow))
        return [x_out3, x_out2, x_out1, ssb_out2]


class SSB(nn.Module):
    def __init__(self, c1=16, c2=32, c3=64, c4=128):
        super(SSB, self).__init__()
        self.conv_in = nn.Conv2d(c1, c1, (1, 1))
        self.conv1 = nn.Conv2d(c2, 1, (1, 1))
        self.conv2 = nn.Conv2d(c3, 1, (1, 1))
        self.conv3 = nn.Conv2d(c4, 1, (1, 1))
        self.out_conv = nn.Conv2d(c4 + 1, c1, (1, 1))
        self.RB1 = ResBlock(c1, c2)
        self.RB2 = ResBlock(c2, c3)
        self.RB3 = ResBlock(c3, c4)
        self.up1 = UpSample(scale=2)
        self.up2 = UpSample(scale=4)
        self.up3 = UpSample(scale=8)
        self.edge_conv = nn.Conv2d(c4, 1, (1, 1))

    def forward(self, x1, x2, x3, x4, gard):
        s1 = self.RB1(self.conv_in(x1))
        sigma = torch.sigmoid(self.conv1(s1 + self.up1(x2)))
        s1 = self.RB2(s1 * sigma)
        sigma = torch.sigmoid(self.conv2(s1 + self.up2(x3)))
        s1 = self.RB3(s1 * sigma)
        sigma = torch.sigmoid(self.conv3(s1 + self.up3(x4)))
        s1 = sigma * s1
        x_edge = torch.sigmoid(self.edge_conv(s1))
        s1 = self.out_conv(torch.cat((s1, gard), dim=1))
        return s1, x_edge


class MSRF(nn.Module):
    def __init__(self, c_in=3, c_out=1, c1=16, c2=32, c3=64, c4=128):
        super(MSRF, self).__init__()
        self.encoder = Encoder(in_c=c_in, c1=c1, c2=c2, c3=c3, c4=c4)
        self.msrf_sub = MSRFSub(c1, c2, c3, c4)
        self.decoder = Decoder(c_out=c_out, c1=c1, c2=c2, c3=c3, c4=c4)

    def gradient_1order(self, x, h_x=None, w_x=None):
        if h_x is None and w_x is None:
            h_x = x.size()[2]
            w_x = x.size()[3]
        r = F.pad(x, (0, 1, 0, 0))[:, :, :, 1:]
        l = F.pad(x, (1, 0, 0, 0))[:, :, :, :w_x]
        t = F.pad(x, (0, 0, 1, 0))[:, :, :h_x, :]
        b = F.pad(x, (0, 0, 0, 1))[:, :, 1:, :]
        xgrad = torch.pow(torch.pow((r - l) * 0.5, 2) +
                          torch.pow((t - b) * 0.5, 2), 0.5)
        return xgrad

    def forward(self, x):
        gard = self.gradient_1order(torch.mean(x, dim=1).unsqueeze(dim=1))
        x1, x2, x3, x4 = self.encoder(x)
        x1, x2, x3, x4 = self.msrf_sub(x1, x2, x3, x4)
        out1, out2, out3, ssb_out = self.decoder(x1, x2, x3, x4, gard)
        return [out1, out2, out3, ssb_out]

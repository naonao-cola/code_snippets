'''
Copyright (C) 2023 TuringVision

'''
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from ..common import SegmentationModelBase

__all__ = ['OFC']


class OFC(SegmentationModelBase):
    '''OFC (Only use Features Classification)

    Usage:
        train_dl, valid_dl = ibll.dataloader(...)
        model = OFC(ibll.labelset(), backbone='resnet50')
        trainer.fit(model, train_dl, valid_dl)

    Model input:
        img (torch.float32): (n,c,h,w)
        target (dict):
            - masks (UInt8Tensor[N, H, W]) (optional): The segmentation masks for each one of the objects
            - boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with values of x
            between 0 and W and values of y between 0 and H
            - labels (Int64Tensor[N]): the class label for each ground-truth box

    Model output (torch.float32):
        - masks ((FloatTensor[N, C, H, W]): the predicted masks (0 ~ 1.0) for each one of category
    '''

    def __init__(self, classes,
                 backbone: str = "resnet50",
                 img_c: int = 3,
                 lr: float = 1e-3,
                 backbone_out_indices=(1, 2, 3, 4),
                 fpn_out_channels=256,
                 loss_pos_balance=1.0,
                 distance_transform=False,
                 **kwargs,
                 ) -> None:
        '''
        classes (Tuple[str, ...]):
        backbone (str): timm backbone
            use this code to show all supported backbones:
            import timm
            timm.list_models(pretrained=True)
        img_c (int): img channel 1 or 3, or other positive integer
            Number of input (color) channels.
            eg: 1-channel (gray image); 25-channel image (maybe satellite image)
        lr (float): init learning rate
        backbone_out_indices (Tuple[int, ...]): timm backbone feature output indices
        fpn_out_channels (Tuple[int, ...]): fpn output channels
        loss_pos_balance (float): positive loss balance ratio
        distance_transform (bool): do distance transform for mask target when training
        '''
        super().__init__(**kwargs)
        # self.save_hyperparameters() not work after cython compile
        self.hparams.update({'classes': classes,
                             'backbone': backbone,
                             'img_c': img_c,
                             'lr': lr,
                             'backbone_out_indices': backbone_out_indices,
                             'fpn_out_channels': fpn_out_channels,
                             'loss_pos_balance': loss_pos_balance,
                             'distance_transform': distance_transform,
                             })
        self.build_model()

    def build_model(self):
        p = self.hparams
        self.num_classes = len(p.classes)
        if len(p.backbone_out_indices) > 1:
            from ..detection.backbones import TimmBackboneWithFPN

            self.backbone = TimmBackboneWithFPN(p.backbone, p.img_c,
                                                p.backbone_out_indices,
                                                p.fpn_out_channels,
                                                False, pretrained=p.pretrained)
            self.head = nn.Conv2d(p.fpn_out_channels, self.num_classes, (1, 1))
            self.fscale = self.backbone.featmap_reductions[0]
        else:
            import timm
            self.backbone = timm.create_model(p.backbone,
                                              features_only=True,
                                              in_chans=p.img_c,
                                              pretrained=p.pretrained,
                                              out_indices=p.backbone_out_indices)
            self.head = nn.Conv2d(self.backbone.feature_info.channels()[0], self.num_classes, (1, 1))
            self.fscale = self.backbone.feature_info.info[self.hparams.backbone_out_indices[0]]['reduction']

    def target_convert(self, images, targets):
        import cv2
        device = images.device
        bs, _, img_h, img_w = images.shape
        masks = np.zeros((bs, self.num_classes, img_h, img_w), dtype=np.float32)
        for i, target in enumerate(targets):
            if 'masks' not in target:
                for box, label in zip(target['boxes'], target['labels']):
                    l, t, r, b = box
                    mask = masks[i, label - 1]
                    cv2.rectangle(mask, (int(l), int(t)), (int(r), int(b)), 1, cv2.FILLED)
            else:
                for label, mask in zip(target['labels'], target['masks']):
                    h, w = mask.shape
                    mask = mask.cpu().numpy()
                    masks[i, label - 1, :h, :w] = np.maximum(masks[i, label - 1, :h, :w], mask)

            if self.hparams.distance_transform:
                for j in range(self.num_classes):
                    from scipy.ndimage.morphology import distance_transform_edt
                    mask = np.uint8(masks[i, j])
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
                    mask = cv2.dilate(mask, kernel)
                    _mask_w = distance_transform_edt(mask)
                    if _mask_w.max() > 0:
                        _mask_w = _mask_w / _mask_w.max()
                        _mask_w = _mask_w ** 2
                    masks[i, j] = _mask_w

        masks_t = torch.from_numpy(masks).to(device)
        targets = F.interpolate(masks_t, (int(img_h / self.fscale),
                                          int(img_w / self.fscale)), mode='area')
        return targets

    def loss(self, preds, targets):
        fg_mask = targets > 0.1
        bg_mask = targets == 0
        bce = nn.BCEWithLogitsLoss()
        bg_loss = bce(preds[bg_mask], targets[bg_mask])
        fg_loss = torch.zeros_like(bg_loss)
        if fg_mask.sum() > 0:
            fg_loss = bce(preds[fg_mask], targets[fg_mask])
        fg_loss = self.hparams.loss_pos_balance * fg_loss
        loss = fg_loss + bg_loss
        return loss, {'fg_loss': fg_loss, 'bg_loss': bg_loss}

    def forward(self, images, targets=None):
        x = self.backbone(images)
        # with fpn
        if len(self.hparams.backbone_out_indices) > 1:
            x = x['0']
        # no fpn
        else:
            x = x[0]
        x = self.head(x)
        if targets:
            targets = self.target_convert(images, targets)
            return self.loss(x, targets)

        img_h, img_w = images.shape[-2:]
        x = x.sigmoid()
        x = F.interpolate(x, (img_h, img_w), mode='nearest')
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
        return OFC.post_process(outputs, x_info, self.hparams.classes,
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

        loss, loss_dict = self.forward(images, targets)
        loss_dict["loss"] = loss
        self.log_dict(loss_dict, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        loss, _ = self.forward(images, targets)
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

'''
Copyright (C) 2023 TuringVision

Implementation of YOLO
'''
import torch
import pytorch_lightning as pl
from torchvision.ops import box_iou
from ..common import DetectionModelBase

__all__ = ['YOLO']


def _evaluate_iou(target, pred):
    """
    Evaluate intersection over union (IOU) for target from dataset and output prediction
    from model
    """
    if pred["boxes"].shape[0] == 0:
        # no box detected, 0 IOU
        return torch.tensor(0.0, dtype=torch.float32, device=pred["boxes"].device)
    return box_iou(target["boxes"], pred["boxes"]).diag().mean()


class YOLO(DetectionModelBase):
    '''YOLO with pre-trained models.

    Usage:
        train_dl, valid_dl = ibll.dataloader(...)
        model = YOLO(ibll.labelset(), backbone='resnet50')
        trainer.fit(model, train_dl, valid_dl)

    Model input:
        img (torch.float32): (n,c,h,w)
        target (dict):
            - boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with values of x
            between 0 and W and values of y between 0 and H
            - labels (Int64Tensor[N]): the class label for each ground-truth box


    Model output (dict):
        - boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values of x
        between 0 and W and values of y between 0 and H
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores or each prediction
    '''
    def __init__(self, classes,
                 backbone: str="resnet50",
                 img_c: int=3,
                 lr: float=1e-3,
                 backbone_out_indices=(2, 3, 4),
                 fpn_out_channels=256,
                 fpn_last_maxpool=True,
                 anchor_sizes=((32,), (64,), (128,), (256,)),
                 aspect_ratios=(0.5, 1.0, 2.0),
                 loss_box_w=0.05,
                 loss_obj_w=1.0,
                 loss_cls_w=0.1,
                 loss_pos_balance=0.5,
                 box_score_thresh=0.05,
                 box_nms_thresh=0.5,
                 box_detections_per_img=100,
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
        fpn_last_maxpool (bool): whether fpn with LastLevelMaxPool() or not
        anchor_sizes (Tuple[Tuple[int, ...], ...]): anchor sizes for each feature level
        aspect_ratios (Tuple[float, flaot, float]): anchor aspect raitos
        loss_box_w (float): box loss weight
        loss_obj_w (float): object loss weight
        loss_cls_w (float): classification loss weight
        loss_pos_balance (float): positive loss balance ratio
        box_score_thresh (float): during inference, only return proposals with a classification score
            greater than box_score_thresh
        box_nms_thresh (float): NMS threshold for the prediction head. Used during inference
        box_detections_per_img (int): maximum number of detections per image, for all classes.
        '''
        super().__init__(**kwargs)
        # self.save_hyperparameters() not work after cython compile
        self.hparams.update({'classes': classes,
                             'backbone': backbone,
                             'img_c': img_c,
                             'lr': lr,
                             'backbone_out_indices': backbone_out_indices,
                             'fpn_out_channels': fpn_out_channels,
                             'fpn_last_maxpool': fpn_last_maxpool,
                             'anchor_sizes': anchor_sizes,
                             'aspect_ratios': aspect_ratios,
                             'loss_box_w': loss_box_w,
                             'loss_obj_w': loss_obj_w,
                             'loss_cls_w': loss_cls_w,
                             'loss_pos_balance': loss_pos_balance,
                             'box_score_thresh': box_score_thresh,
                             'box_nms_thresh': box_nms_thresh,
                             'box_detections_per_img': box_detections_per_img,
                            })
        self.build_model()

    def build_model(self):
        from .backbones import TimmBackboneWithFPN
        from .head import YoloHead

        p = self.hparams
        self.backbone = TimmBackboneWithFPN(p.backbone, p.img_c,
                                            p.backbone_out_indices,
                                            p.fpn_out_channels,
                                            p.fpn_last_maxpool,
                                            pretrained=p.pretrained)
        self.head = YoloHead(p.fpn_out_channels, len(p.classes),
                             self.backbone.featmap_names,
                             self.backbone.featmap_reductions,
                             p.anchor_sizes, p.aspect_ratios,
                             p.loss_box_w, p.loss_obj_w, p.loss_cls_w,
                             p.loss_pos_balance,
                             p.box_score_thresh, p.box_nms_thresh,
                             p.box_detections_per_img)

    def target_convert(self, images, targets):
        device = images.device
        img_h, img_w = images.shape[-2:]
        total = sum([1 for target in targets for l in target['labels'] if l > 0])
        yolo_targets = torch.zeros((total, 6), dtype=torch.float32, device=device)
        j = 0
        for i, target in enumerate(targets):
            for box, label in zip(target['boxes'], target['labels']):
                if label > 0:
                    yolo_targets[j, 0] = i
                    yolo_targets[j, 1] = label - 1
                    l,t,r,b = box
                    x,y,w,h = (l+r)/2, (t+b)/2, r-l, b-t
                    yolo_targets[j, 2] = x / img_w
                    yolo_targets[j, 3] = y / img_h
                    yolo_targets[j, 4] = w / img_w
                    yolo_targets[j, 5] = h / img_h
                    j += 1
        return yolo_targets

    def forward(self, images, targets=None):
        if targets:
            targets = self.target_convert(images, targets)
        image_sizes = [images.shape[-2:]] * images.shape[0]
        x = self.backbone(images)
        x = self.head(x, targets, image_sizes)
        return x

    def predict(self, x, x_info, box_tfm=None):
        '''
        output:
            [{'labels': ['a', 'b', ...], 'bboxes': [[l,t,r,b,conf], ...]}, ...]
        '''
        x = x.to(self.device)
        outputs = self.forward(x)
        image_sizes = [x.shape[-2:]] * x.shape[0]
        p = self.hparams
        return YOLO.post_process(outputs, image_sizes, x_info, p.classes, box_tfm,
                                 p.box_score_thresh,
                                 p.box_nms_thresh,
                                 p.box_detections_per_img)

    @classmethod
    def post_process(cls, outputs, image_sizes, x_info, classes,
                     box_tfm=None,
                     box_score_thresh=0.05,
                     box_nms_thresh=0.5,
                     box_detections_per_img=100):
        from .head import YoloHead
        outputs = YoloHead.post_process(outputs, image_sizes,
                                        box_score_thresh,
                                        box_nms_thresh,
                                        box_detections_per_img)

        keys = ['boxes', 'labels', 'scores']
        outputs = [[op[k].cpu().numpy() for  k in keys] for op in outputs]
        return DetectionModelBase.post_process(outputs, x_info, classes, box_tfm)

    def training_step(self, batch, batch_idx):
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

        loss_dict = self.forward(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        loss_dict['loss'] = loss.item()
        self.log_dict(loss_dict, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outs = self.forward(images)
        image_sizes = [images.shape[-2:]] * images.shape[0]
        outs = self.head.postprocess_detections(outs, image_sizes)
        iou = torch.stack([_evaluate_iou(t, o) for t, o in zip(targets, outs)]).mean()
        return {"val_iou": iou}

    def validation_epoch_end(self, outs):
        avg_iou = torch.stack([o["val_iou"] for o in outs]).mean()
        logs = {"val_iou": avg_iou}
        self.log_dict(logs, prog_bar=True)

    def configure_optimizers(self):
        from torch import optim
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, betas=(0.5, 0.99))
        # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        return [optimizer]
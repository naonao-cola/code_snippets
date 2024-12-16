'''
Copyright (C) 2023 TuringVision

Implementation of `Faster R-CNN: Towards Real-Time Object Detection with
Region Proposal Networks <https://arxiv.org/abs/1506.01497>`_.
'''
import torch
import pytorch_lightning as pl
from torchvision.ops import box_iou
from torchvision.models.detection.image_list import ImageList
from torch.jit.annotations import List, Tuple
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from ..common import DetectionModelBase

__all__ = ['FasterRCNN']


def _evaluate_iou(target, pred):
    """
    Evaluate intersection over union (IOU) for target from dataset and output prediction
    from model
    """
    if pred["boxes"].shape[0] == 0:
        # no box detected, 0 IOU
        return torch.tensor(0.0, dtype=torch.float32, device=pred["boxes"].device)
    return box_iou(target["boxes"], pred["boxes"]).diag().mean()


class FasterRCNN(DetectionModelBase):
    '''Faster-Rcnn with pre-trained models.

    Usage:
        train_dl, valid_dl = ibll.dataloader(...)
        model = FasterRCNN(ibll.labelset(),
                           backbone='resnet50')
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
                 backbone: str = "resnet50",
                 img_c: int = 3,
                 lr: float = 1e-2,
                 backbone_out_indices=(1, 2, 3, 4),
                 fpn_out_channels=256,
                 fpn_last_maxpool=True,
                 anchor_sizes=((32,), (64,), (128,), (256,), (512,)),
                 aspect_ratios=(0.5, 1.0, 2.0),
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=512, box_positive_fraction=0.25,
                 bbox_reg_weights=None,
                 **kwargs,
                 ) -> None:
        '''
        classes (Tuple[str, ...]):
        backbone (str): timm backbone
            use this code to show all supported backbones:
            import timm
            timm.list_models(pretrained=True)
        imc_c (int): img channel 1 or 3
        lr (float): init learning rate
        backbone_out_indices (Tuple[int, ...]): timm backbone feature output indices
        fpn_out_channels (Tuple[int, ...]): fpn output channels
        fpn_last_maxpool (bool): whether fpn with LastLevelMaxPool() or not
        anchor_sizes (Tuple[Tuple[int, ...], ...]): anchor sizes for each feature level
        aspect_ratios (Tuple[float, flaot, float]): anchor aspect raitos (height / width)
        rpn_pre_nms_top_n_train (int): number of proposals to keep before applying NMS during training
        rpn_pre_nms_top_n_test (int): number of proposals to keep before applying NMS during testing
        rpn_post_nms_top_n_train (int): number of proposals to keep after applying NMS during training
        rpn_post_nms_top_n_test (int): number of proposals to keep after applying NMS during testing
        rpn_nms_thresh (float): NMS threshold used for postprocessing the RPN proposals
        rpn_fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        rpn_bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        rpn_batch_size_per_image (int): number of anchors that are sampled during training of the RPN
            for computing the loss
        rpn_positive_fraction (float): proportion of positive anchors in a mini-batch during training
            of the RPN
        box_score_thresh (float): during inference, only return proposals with a classification score
            greater than box_score_thresh
        box_nms_thresh (float): NMS threshold for the prediction head. Used during inference
        box_detections_per_img (int): maximum number of detections per image, for all classes.
        box_fg_iou_thresh (float): minimum IoU between the proposals and the GT box so that they can be
            considered as positive during training of the classification head
        box_bg_iou_thresh (float): maximum IoU between the proposals and the GT box so that they can be
            considered as negative during training of the classification head
        box_batch_size_per_image (int): number of proposals that are sampled during training of the
            classification head
        box_positive_fraction (float): proportion of positive proposals in a mini-batch during training
            of the classification head
        bbox_reg_weights (Tuple[float, float, float, float]): weights for the encoding/decoding of the
            bounding boxes
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
                             'rpn_pre_nms_top_n_train': rpn_pre_nms_top_n_train,
                             'rpn_pre_nms_top_n_test': rpn_pre_nms_top_n_test,
                             'rpn_post_nms_top_n_train': rpn_post_nms_top_n_train,
                             'rpn_post_nms_top_n_test': rpn_post_nms_top_n_test,
                             'rpn_nms_thresh': rpn_nms_thresh,
                             'rpn_fg_iou_thresh': rpn_fg_iou_thresh,
                             'rpn_bg_iou_thresh': rpn_bg_iou_thresh,
                             'rpn_batch_size_per_image': rpn_batch_size_per_image,
                             'rpn_positive_fraction': rpn_positive_fraction,
                             'box_score_thresh': box_score_thresh,
                             'box_nms_thresh': box_nms_thresh,
                             'box_detections_per_img': box_detections_per_img,
                             'box_fg_iou_thresh': box_fg_iou_thresh,
                             'box_bg_iou_thresh': box_bg_iou_thresh,
                             'box_batch_size_per_image': box_batch_size_per_image,
                             'box_positive_fraction': box_positive_fraction,
                             'bbox_reg_weights': bbox_reg_weights,
                             })
        self.build_model()

    def build_model(self):
        from .backbones import TimmBackboneWithFPN
        from .rpn import StandardRPN
        from .head import StandardRoIHeads

        p = self.hparams
        self.transform = GeneralizedRCNNTransform(0, 0, [1], [1])
        self.backbone = TimmBackboneWithFPN(p.backbone, p.img_c, p.backbone_out_indices, \
                                            p.fpn_out_channels, p.fpn_last_maxpool, pretrained=p.pretrained)

        self.rpn = StandardRPN(p.fpn_out_channels,
                               p.anchor_sizes, p.aspect_ratios,
                               p.rpn_pre_nms_top_n_train, p.rpn_pre_nms_top_n_test,
                               p.rpn_post_nms_top_n_train, p.rpn_post_nms_top_n_test,
                               p.rpn_nms_thresh,
                               p.rpn_fg_iou_thresh, p.rpn_bg_iou_thresh,
                               p.rpn_batch_size_per_image, p.rpn_positive_fraction)

        self.roi_heads = StandardRoIHeads(p.fpn_out_channels,
                                          len(p.classes), self.backbone.featmap_names,
                                          p.box_score_thresh, p.box_nms_thresh, p.box_detections_per_img,
                                          p.box_fg_iou_thresh, p.box_bg_iou_thresh,
                                          p.box_batch_size_per_image, p.box_positive_fraction,
                                          p.bbox_reg_weights)

    def forward(self, images, targets=None):
        image_sizes = [images.shape[-2:]] * images.shape[0]
        features = self.backbone(images)
        proposals, proposal_losses = self.rpn(ImageList(images, image_sizes), features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, image_sizes, targets)

        detections = self.transform.postprocess(detections, image_sizes, image_sizes)
        if targets:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return detections, losses
        return detections

    def predict(self, x, x_info, box_tfm=None):
        '''
        output:
            [{'labels': ['a', 'b', ...], 'bboxes': [[l,t,r,b,conf], ...]}, ...]
        '''
        x = x.to(self.device)
        outputs = self.forward(x)
        keys = ['boxes', 'labels', 'scores']
        outputs = [[op[k].cpu().numpy() for k in keys] for op in outputs]
        return FasterRCNN.post_process(outputs, x_info, self.hparams.classes, box_tfm)

    @classmethod
    def post_process(cls, outputs, x_info, classes, box_tfm=None):
        return DetectionModelBase.post_process(outputs, x_info, classes, box_tfm, bias=-1)

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

        # fasterrcnn takes both images and targets for training, returns
        _, loss_dict = self.forward(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        loss_dict["loss"] = loss
        self.log_dict(loss_dict, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        # fasterrcnn takes only images for eval() mode
        outs = self.forward(images)
        iou = torch.stack([_evaluate_iou(t, o) for t, o in zip(targets, outs)]).mean()
        return {"val_iou": iou}

    def validation_epoch_end(self, outs):
        avg_iou = torch.stack([o["val_iou"] for o in outs]).mean()
        logs = {"val_iou": avg_iou}
        self.log_dict(logs, prog_bar=True)

    def configure_optimizers(self):
        from torch import optim
        optimizer = optim.SGD(self.parameters(), lr=self.hparams.lr,
                              momentum=0.9, weight_decay=0.005)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        return [optimizer], [scheduler]

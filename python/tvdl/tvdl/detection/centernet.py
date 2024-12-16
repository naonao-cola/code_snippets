'''
Copyright (C) 2023 TuringVision

Implementation of CenterNet
'''
import math
import numpy as np
import torch
import pytorch_lightning as pl

from torchvision.ops import box_iou
from torch.jit.annotations import List, Dict

from ..common import DetectionModelBase

__all__ = ['CenterNet']


def _evaluate_iou(target, pred):
    """
    Evaluate intersection over union (IOU) for target from dataset and output prediction
    from model
    """
    if pred["boxes"].shape[0] == 0:
        # no box detected, 0 IOU
        return torch.tensor(0.0, device=pred["boxes"].device)
    return box_iou(target["boxes"], pred["boxes"]).diag().mean()


class CenterNet(DetectionModelBase):
    '''CENTERNET with pre-trained models.

    Usage:
        train_dl, valid_dl = ibll.dataloader(...)
        model = CenterNet(ibll.labelset(), backbone='resnet50')
        trainer.fit(model, train_dl, valid_dl)

    Model input:
        img (torch.float32): (n,c,h,w)
        target (dict):
            - boxes (FloatTensor[n,N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with values of x
            between 0 and W and values of y between 0 and H
            - labels (Int64Tensor[n,N]): the class label for each ground-truth box


    Model output (dict):
        - boxes (FloatTensor[n,N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values of x
        between 0 and W and values of y between 0 and H
        - labels (Int64Tensor[n,N]): the predicted labels for each image
        - scores (Tensor[n,N]): the scores or each prediction
    '''

    def __init__(self, classes,
                 backbone: str = "resnet50",
                 img_c: int = 3,
                 lr: float = 1e-3,
                 backbone_out_indices=(2, 3, 4),
                 fpn_out_channels=256,
                 loss_hm_weight=1.0,
                 loss_wh_weight=0.1,
                 loss_off_weight=0.1,
                 loss_pos_balance=1.0,
                 box_score_thresh=0.05,
                 box_detections_per_img=100,
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
        loss_hm_weight (float): heat loss weight
        loss_wh_weight (float): wh loss weight
        loss_off_weight (float): wg off loss weight
        loss_pos_balance (float): positive loss balance ratio
        box_score_thresh (float): during inference, only return proposals with a classification score
            greater than box_score_thresh
        box_detections_per_img (int): maximum number of detections per image, for all classes.
        '''
        super().__init__(**kwargs)
        self.hparams.update({'classes': classes,
                             'backbone': backbone,
                             'img_c': img_c,
                             'lr': lr,
                             'backbone_out_indices': backbone_out_indices,
                             'fpn_out_channels': fpn_out_channels,
                             'loss_hm_weight': loss_hm_weight,
                             'loss_wh_weight': loss_wh_weight,
                             'loss_off_weight': loss_off_weight,
                             'loss_pos_balance': loss_pos_balance,
                             'box_score_thresh': box_score_thresh,
                             'box_detections_per_img': box_detections_per_img,
                             })
        self.build_model()

    def build_model(self):
        from .backbones import TimmBackboneWithFPN
        from .head.centernet_head import CenterNetHead

        p = self.hparams
        self.num_classes = len(p.classes)
        self.box_score_thresh = p.box_score_thresh
        self.max_objects = p.box_detections_per_img
        self.backbone = TimmBackboneWithFPN(p.backbone, p.img_c,
                                            p.backbone_out_indices,
                                            p.fpn_out_channels,
                                            False, pretrained=p.pretrained)
        self.down_ratio = self.backbone.featmap_reductions[0]
        head_names = ["heatmap", "width_height", "regression"]
        head_out_channels = [len(p.classes), 2, 2]
        self.heads = CenterNetHead(self.backbone.out_channels, head_names,
                                   head_out_channels, p.loss_pos_balance)

    def _box_to_bbox(self, box):
        box = box.cpu().numpy()
        return np.array(
            [box[0], box[1], box[2], box[3]], dtype=np.float32
        )

    def _scale_point(self, point, output_size):
        x, y = point / self.down_ratio
        output_h, output_w = output_size
        x = np.clip(x, 0, output_w - 1)
        y = np.clip(y, 0, output_h - 1)
        return [x, y]

    def target_convert_centernet(self, output_w, output_h, targets, device):

        heatmap = torch.zeros(
            (len(targets), self.num_classes, output_h, output_w),
            dtype=torch.float32
        )
        width_height = torch.zeros((len(targets), self.max_objects, 2), dtype=torch.float32)
        regression = torch.zeros((len(targets), self.max_objects, 2), dtype=torch.float32)
        regression_mask = torch.zeros((len(targets), self.max_objects), dtype=torch.bool)
        indices = torch.zeros((len(targets), self.max_objects), dtype=torch.int64)
        for i, target in enumerate(targets):
            num_objects = min(len(target['boxes']), self.max_objects)
            for k in range(num_objects):
                label = target['labels'][k]
                if label <= 0:
                    continue
                bbox = self._box_to_bbox(target["boxes"][k])
                cls_id = label - 1
                # Scale to output size
                bbox[:2] = self._scale_point(bbox[:2], (output_h, output_w))
                bbox[2:] = self._scale_point(bbox[2:], (output_h, output_w))
                h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
                if h > 0 and w > 0:
                    radius = self.heads.gaussian_radius((math.ceil(h), math.ceil(w)))
                    radius = int(max(1e-5, radius))
                    ct = torch.FloatTensor(
                        [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                    )
                    ct_int = ct.to(torch.int32)
                    self.heads.draw_umich_gaussian(heatmap[i][cls_id], ct_int, radius)
                    width_height[i][k] = torch.tensor([1.0 * w, 1.0 * h])
                    indices[i][k] = ct_int[1] * output_w + ct_int[0]
                    regression[i][k] = ct - ct_int
                    regression_mask[i][k] = 1
        ret = {
            "heatmap": heatmap.to(device),
            "regression_mask": regression_mask.to(device),
            "indices": indices.to(device),
            "width_height": width_height.to(device),
            "regression": regression.to(device),
        }
        return ret

    def forward(self, images, targets=None):
        if targets:
            targets = self.target_convert(images, targets)
        image_sizes = [images.shape[-2:]] * images.shape[0]
        outputs = self.backbone(images)
        rets = self.heads(outputs['0'])
        if self.training:
            return rets
        result = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])
        # res process
        boxes, scores, labels = self.postprocess_detections(rets, image_sizes)
        num_images = len(boxes)
        for i in range(num_images):
            result.append(
                {
                    "boxes": boxes[i],
                    "labels": labels[i],
                    "scores": scores[i],
                }
            )
        return result

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
        outputs = self(images)
        loss, loss_dict = self.loss(outputs, targets)
        loss_dict["loss"] = loss
        self.log_dict(loss_dict, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        iou = torch.stack([_evaluate_iou(t, o) for t, o in zip(targets, outputs)]).mean()
        return {"val_iou": iou}

    def validation_epoch_end(self, outs):
        avg_iou = torch.stack([o["val_iou"] for o in outs]).mean()
        logs = {"val_iou": avg_iou}
        self.log_dict(logs, prog_bar=True)

    def loss(self, outputs, target):
        hm_loss, wh_loss, off_loss = 0, 0, 0
        output_h, output_w = outputs["heatmap"].shape[-2:]
        device = outputs["heatmap"].device
        target = self.target_convert_centernet(output_w, output_h, target, device)

        hm_loss += self.heads.neg_loss(outputs["heatmap"], target["heatmap"])
        wh_loss += self.heads.reg_l1_loss(
            outputs["width_height"],
            target["regression_mask"],
            target["indices"],
            target["width_height"],
        )
        off_loss += self.heads.reg_l1_loss(
            outputs["regression"],
            target["regression_mask"],
            target["indices"],
            target["regression"],
        )
        hm_loss = self.hparams.loss_hm_weight * hm_loss
        wh_loss = self.hparams.loss_wh_weight * wh_loss
        off_loss = self.hparams.loss_off_weight * off_loss
        loss = hm_loss + wh_loss + off_loss

        loss_stats = {
            "hm_loss": hm_loss,
            "wh_loss": wh_loss,
            "off_loss": off_loss,
        }
        return loss, loss_stats

    def ctdet_decode(self, heat, wh, reg=None, K=100):
        batch, cat, height, width = heat.size()
        heat = self.heads.heat_nms(heat)
        scores, inds, clses, ys, xs = self.heads.topk(heat, K=K)
        if reg is not None:
            reg = self.heads.transpose_and_gather_feat(reg, inds)
            reg = reg.view(batch, K, 2)
            xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
            ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
        else:
            xs = xs.view(batch, K, 1) + 0.5
            ys = ys.view(batch, K, 1) + 0.5
        wh = self.heads.transpose_and_gather_feat(wh, inds)
        wh = wh.view(batch, K, 2)
        clses = clses.view(batch, K, 1).float()
        scores = scores.view(batch, K, 1)
        bboxes = torch.cat(
            [
                xs - wh[..., 0:1] / 2,
                ys - wh[..., 1:2] / 2,
                xs + wh[..., 0:1] / 2,
                ys + wh[..., 1:2] / 2,
            ],
            dim=2,
        )
        detections = torch.cat([bboxes, scores, clses], dim=2)
        return detections

    def postprocess_detections(self, rets, image_shapes):
        out_heatmaps = rets["heatmap"]
        out_width_heights = rets["width_height"]
        out_regressions = rets["regression"]
        all_boxes = []
        all_scores = []
        all_labels = []
        for heatmap, width_height, regression, image_shape in zip(out_heatmaps,
                                                                  out_width_heights,
                                                                  out_regressions,
                                                                  image_shapes):
            detection = self.ctdet_decode(
                heatmap.unsqueeze(0).sigmoid_(),
                width_height.unsqueeze(0),
                reg=regression.unsqueeze(0),
                K=self.max_objects
            )
            detection = detection.reshape(-1, 6)
            boxes = detection[:, :4] * self.down_ratio
            scores = detection[:, 4]
            labels = detection[:, 5]

            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            inds = torch.nonzero(scores > self.box_score_thresh, as_tuple=False).squeeze(1)
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels

    def predict(self, x, x_info, box_tfm=None):
        '''
        output:
            [{'labels': ['a', 'b', ...], 'bboxes': [[l,t,r,b,conf], ...]}, ...]
        '''
        x = x.to(self.device)
        outputs = self.forward(x)
        keys = ['boxes', 'labels', 'scores']
        outputs = [[op[k].cpu().numpy() for k in keys] for op in outputs]
        return DetectionModelBase.post_process(outputs, x_info, self.hparams.classes, box_tfm)

    def configure_optimizers(self):
        from torch import optim
        optimizer = optim.SGD(self.parameters(), lr=self.hparams.lr,
                              momentum=0.9, weight_decay=0.005)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        return [optimizer], [scheduler]

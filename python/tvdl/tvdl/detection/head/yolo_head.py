'''
Copyright (C) 2023 TuringVision

Yolo v5 head.
'''
import math
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.jit.annotations import Optional, List, Dict, Tuple
from ..utils import bbox_iou, xywh2xyxy

__all__ = ['YoloHead']


class YoloHead(nn.Module):
    ''' Yolo v5 head
        in_channels (int): backbone output channels
        num_classes (int):
        featmap_names (Tuple[str, ...])
        featmap_strides (Tuple[int, ...])
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
    def __init__(self,
                 in_channels,
                 num_classes,
                 featmap_names,
                 featmap_strides,
                 anchor_sizes = ((32,), (64,), (128,), (256,)),
                 aspect_ratios = (0.5, 1.0, 2.0),
                 loss_box_w=0.05,
                 loss_obj_w=1.0,
                 loss_cls_w=0.5,
                 loss_pos_balance=0.5,
                 box_score_thresh=0.05,
                 box_nms_thresh=0.5,
                 box_detections_per_img=100,
                 ):
        super(YoloHead, self).__init__()
        self.loss_box_w = loss_box_w
        self.loss_obj_w = loss_obj_w
        self.loss_cls_w = loss_cls_w
        self.loss_pos_balance = loss_pos_balance

        self.box_score_thresh = box_score_thresh
        self.box_nms_thresh = box_nms_thresh
        self.box_detections_per_img = box_detections_per_img

        # 1. create conv head
        anchors = []
        h_ratios = np.sqrt(aspect_ratios)
        for anchor_size in anchor_sizes:
            for size in anchor_size:
                one_anchors = []
                for h_ratio in h_ratios:
                    w_ratio  = 1 / h_ratio
                    one_anchors += [size * w_ratio, size * h_ratio]
                anchors.append(one_anchors)

        ch = [in_channels] * len(featmap_names)

        self.featmap_names = featmap_names
        self.num_classes = num_classes  # number of classes
        self.num_output = num_classes + 5  # number of outputs per anchor
        self.num_layers = len(anchors)  # number of detection layers
        self.num_anchors = len(anchors[0]) // 2  # number of anchors

        self.grid = [torch.zeros(1, dtype=torch.float32)] * self.num_layers  # init grid
        a = torch.tensor(anchors).float().view(self.num_layers, -1, 2)
        self.register_buffer('anchors', a)  # shape(num_layers,num_anchors,2)
        self.register_buffer('anchor_grid', a.clone().view(self.num_layers, 1, -1, 1, 1, 2))

        self.m = nn.ModuleList(nn.Conv2d(x, self.num_output * self.num_anchors, 1) for x in ch)

        self.stride = featmap_strides
        self.anchors /= torch.tensor(featmap_strides).view(-1, 1, 1)

        self.balance = [4.0, 1.0, 0.25, 0.06, .02]
        self._initialize_biases()

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        for mi, s in zip(self.m, self.stride):  # from
            b = mi.bias.view(self.num_anchors, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (self.num_classes - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def loss(self, p, targets):
        device = targets.device
        lcls, lbox, lobj = [torch.zeros(1, dtype=torch.float32, device=device) for _ in range(3)]
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
            num_anchors, nh, nw = pi.shape[1:4]

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = iou.detach().clamp(0).type(tobj.dtype)  # iou ratio
                # Classification
                t = torch.full_like(ps[:, 5:], 0.0, device=device)  # targets
                t[range(n), tcls[i]] = 1.0

                fg_cnt = (t > 0).sum()
                bg_cnt = (t == 0).sum()
                pos_weight = bg_cnt * self.loss_pos_balance / max(fg_cnt, 2.0)
                BCEcls = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                lcls += BCEcls(ps[:, 5:], t)

            fg_cnt = (tobj > 0).sum()
            bg_cnt = (tobj == 0).sum()
            pos_weight = bg_cnt * self.loss_pos_balance / max(fg_cnt, 2.0)
            BCEobj = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            obji = BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]

        lbox *= self.loss_box_w
        lobj *= self.loss_obj_w
        lcls *= self.loss_cls_w
        return {'loss_box': lbox, 'loss_obj': lobj, 'loss_cls': lcls}

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        num_anchors, nt = self.num_anchors, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, dtype=torch.float32, device=targets.device)  # normalized to gridspace gain
        ai = torch.arange(num_anchors, device=targets.device).float().view(num_anchors, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(num_anchors, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.num_layers):
            anchors = self.anchors[i]
            anchors = anchors.to(targets.device)
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < 2.91 # ratio threshod
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch

    def _forward_m(self, x):
        z = []  # inference output
        for i in range(self.num_layers):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.num_anchors, self.num_output, ny, nx)
            x[i] = x[i].permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny)

                grid_i = self.grid[i].to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid_i) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.num_output))

        return x if self.training else torch.cat(z, 1)

    def forward(self, features, targets=None, image_shapes=None):
        features = [features[n] for n in self.featmap_names]
        pred = self._forward_m(features)

        if self.training:
            # compute loss
            loss_dict = self.loss(pred, targets)
            return loss_dict

        return pred


    def postprocess_detections(self, pred, image_shapes):
        return YoloHead.post_process(pred, image_shapes,
                                     self.box_score_thresh,
                                     self.box_nms_thresh,
                                     self.box_detections_per_img
                                     )

    @classmethod
    def post_process(cls, pred, image_shapes, box_score_thresh, box_nms_thresh, box_detections_per_img):
        from torchvision.ops import boxes as box_ops

        all_boxes = []
        all_scores = []
        all_labels = []

        for pred_i, image_shape in zip(pred, image_shapes):
            boxes = xywh2xyxy(pred_i[:, :4])
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
            scores = pred_i[:, 5:] * pred_i[:, 4:5]
            scores, labels = scores.max(1)

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            inds = torch.nonzero(scores > box_score_thresh, as_tuple=False).squeeze(1)
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=4)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, box_nms_thresh)
            # keep only topk scoring predictions
            keep = keep[:box_detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        # return all_boxes, all_scores, all_labels
        result = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])
        num_images = len(all_boxes)
        for i in range(num_images):
            result.append(
                {
                    "boxes": all_boxes[i],
                    "labels": all_labels[i],
                    "scores": all_scores[i],
                }
            )
        return result

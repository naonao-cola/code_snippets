'''
Copyright (C) 2023 TuringVision

Implements Mask R-CNN.
'''

import torch
import pytorch_lightning as pl
from ..detection import FasterRCNN

__all__ = ['MaskRCNN']


class MaskRCNN(FasterRCNN):
    '''Mask-Rcnn with pre-trained models.

    Usage:
        train_dl, valid_dl = ibll.dataloader(...)
        model = MaskRCNN(ibll.labelset(),
                           backbone='resnet50')
        trainer.fit(model, train_dl, valid_dl)

    Model input:
        img (torch.float32): (n,c,h,w)
        target (dict):
            - masks (UInt8Tensor[N, H, W]): The segmentation masks for each one of the objects
            - boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with values of x
            between 0 and W and values of y between 0 and H
            - labels (Int64Tensor[N]): the class label for each ground-truth box

    Model output (dict):
        - boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values of x
        between 0 and W and values of y between 0 and H
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores or each prediction
        - masks ((FloatTensor[N, 1, H, W]): the predicted masks (0 ~ 1.0) for each one of objects
    '''
    def build_model(self):
        from .head import StandardMaskRoIHeads
        super(MaskRCNN, self).build_model()
        p = self.hparams
        self.roi_heads = StandardMaskRoIHeads(p.fpn_out_channels,
                                              len(p.classes), self.backbone.featmap_names,
                                              p.box_score_thresh, p.box_nms_thresh,
                                              p.box_detections_per_img,
                                              p.box_fg_iou_thresh, p.box_bg_iou_thresh,
                                              p.box_batch_size_per_image, p.box_positive_fraction,
                                              p.bbox_reg_weights)

    @classmethod
    def post_process(cls, outputs, infos, classes,
                     polygon_tfm=None,
                     mask_threshold=0.5):
        from tvlab.utils.basic import mask_to_polygon
        ypred = []
        for o, info in zip(outputs, infos):
            boxes, labels, scores, masks = o
            yp_polygons = []
            yp_labels = []
            for mask, label, s in zip(masks, labels, scores):
                polygon = mask_to_polygon(mask[0] > mask_threshold)
                if polygon is None:
                    polygon = []
                if len(polygon) > 6:
                    if polygon_tfm:
                        polygon = polygon_tfm(polygon, info['ori_shape'])
                    yp_polygons.append(polygon + [s])
                    yp_labels.append(classes[label-1])
            yp = {'labels': yp_labels, 'polygons': yp_polygons}
            ypred.append(yp)
        return ypred

    def predict(self, x, y, polygon_tfm=None, mask_threshold=0.5):
        '''
        output:
            [{'labels': ['a', 'b', ...], 'polygons': [[x1,y1,x2,y2,x3,y3,...,conf], ...]}, ...]
        '''
        x = x.to(self.device)
        outputs = self.forward(x)
        keys = ['boxes', 'labels', 'scores', 'masks']
        outputs = [[op[k].cpu().numpy() for  k in keys] for op in outputs]
        return MaskRCNN.post_process(outputs, y, self.hparams.classes, polygon_tfm, mask_threshold)

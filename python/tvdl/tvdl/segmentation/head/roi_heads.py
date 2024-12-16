'''
Copyright (C) 2023 TuringVision

standard mask roi heads
'''
from torch import nn
from torchvision.ops import MultiScaleRoIAlign
from collections import OrderedDict
from ...detection.head import StandardRoIHeads

__all__ = ['StandardMaskRoIHeads']


class MaskRCNNHeads(nn.Sequential):
    def __init__(self, in_channels, layers, dilation):
        """
        Arguments:
            in_channels (int): number of input channels
            layers (list): feature dimensions of each FCN layer
            dilation (int): dilation rate of kernel
        """
        d = OrderedDict()
        next_feature = in_channels
        for layer_idx, layer_features in enumerate(layers, 1):
            d["mask_fcn{}".format(layer_idx)] = nn.Conv2d(
                next_feature, layer_features, kernel_size=3,
                stride=1, padding=dilation, dilation=dilation)
            d["relu{}".format(layer_idx)] = nn.ReLU(inplace=True)
            next_feature = layer_features

        super(MaskRCNNHeads, self).__init__(d)
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
            # elif "bias" in name:
            #     nn.init.constant_(param, 0)


class MaskRCNNPredictor(nn.Sequential):
    def __init__(self, in_channels, dim_reduced, num_classes):
        super(MaskRCNNPredictor, self).__init__(OrderedDict([
            ("conv5_mask", nn.ConvTranspose2d(in_channels, dim_reduced, 2, 2, 0)),
            ("relu", nn.ReLU(inplace=True)),
            ("mask_fcn_logits", nn.Conv2d(dim_reduced, num_classes, 1, 1, 0)),
        ]))

        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
            # elif "bias" in name:
            #     nn.init.constant_(param, 0)


class StandardMaskRoIHeads(StandardRoIHeads):
    ''' standard roi heads
        in_channels (int): backbone output channels
        num_classes (int):
        featmap_names (Tuple[str, ...])
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
    def __init__(self,
                 in_channels,
                 num_classes,
                 featmap_names,
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=512, box_positive_fraction=0.25,
                 bbox_reg_weights=None
                ):
        super(StandardMaskRoIHeads, self).__init__(
                 in_channels, num_classes, featmap_names,
                 box_score_thresh, box_nms_thresh, box_detections_per_img,
                 box_fg_iou_thresh, box_bg_iou_thresh,
                 box_batch_size_per_image, box_positive_fraction,
                 bbox_reg_weights)

        mask_roi_pool = MultiScaleRoIAlign(
            featmap_names=featmap_names,
            output_size=14,
            sampling_ratio=2)

        mask_layers = (256, 256, 256, 256)
        mask_dilation = 1
        mask_head = MaskRCNNHeads(in_channels, mask_layers, mask_dilation)

        mask_predictor_in_channels = 256  # == mask_layers[-1]
        mask_dim_reduced = 256
        mask_predictor = MaskRCNNPredictor(mask_predictor_in_channels,
                                           mask_dim_reduced, num_classes+1)

        self.mask_roi_pool = mask_roi_pool
        self.mask_head = mask_head
        self.mask_predictor = mask_predictor

'''
Copyright (C) 2023 TuringVision

standard region proposal network
'''
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork

__all__ = ['StandardRPN']


class StandardRPN(RegionProposalNetwork):
    ''' standard region proposal network
    inchannels (int): backbone output channels

    anchor_sizes (Tuple[Tuple[int, ...], ...]): anchor sizes for each feature level
    aspect_ratios (Tuple[float, flaot, float]): anchor aspect raitos
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
    '''
    def __init__(self,
                 in_channels,
                 anchor_sizes = ((32,), (64,), (128,), (256,), (512,)),
                 aspect_ratios = (0.5, 1.0, 2.0),
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                ):
        aspect_ratios = (aspect_ratios,) * len(anchor_sizes)
        rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

        rpn_head = RPNHead(
            in_channels, rpn_anchor_generator.num_anchors_per_location()[0]
        )

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        super(StandardRPN, self).__init__(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)

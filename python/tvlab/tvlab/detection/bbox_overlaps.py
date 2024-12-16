'''
Copyright (C) 2023 TuringVision

'''
import numpy as np

__all__ = ['bbox_overlaps', 'nms', 'y_nms']


def bbox_overlaps(bboxes1, bboxes2, mode='iou'):
    """Calculate the ious between each bbox of bboxes1 and bboxes2.

    Args:
        bboxes1(ndarray): shape (n, 4)
        bboxes2(ndarray): shape (k, 4)
        mode(str): iou (intersection over union) or iof (intersection
            over foreground)

    Returns:
        ious(ndarray): shape (n, k)
    """

    assert mode in ['iou', 'iof']

    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = np.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = np.zeros((cols, rows), dtype=np.float32)
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
        bboxes1[:, 3] - bboxes1[:, 1] + 1)
    area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
        bboxes2[:, 3] - bboxes2[:, 1] + 1)
    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        overlap = np.maximum(x_end - x_start + 1, 0) * np.maximum(
            y_end - y_start + 1, 0)
        if mode == 'iou':
            union = area1[i] + area2 - overlap
        else:
            union = area1[i] if not exchange else area2
        ious[i, :] = overlap / union
    if exchange:
        ious = ious.T
    return ious


def nms(bboxes, scores=None, iou_threshold=0.5):
    '''
    Args:
        bboxes(ndarray): shape (n, 4)
        iou_threshold(float)
    Returns:
        keep(ndarray): int64 with the indices of the elements that
            have been kept by NMS, sorted in decreasing order of scores
    '''
    from torchvision.ops import nms as tnms
    import torch


    bboxes = np.array(bboxes, np.float)

    _scores = scores
    if scores is None:
        _scores = np.ones(len(bboxes))

    _scores = np.array(_scores, np.float)

    keep = tnms(torch.from_numpy(bboxes),
                torch.from_numpy(_scores),
                iou_threshold=iou_threshold)
    return keep.numpy()


def y_nms(y, iou_threshold=0.5):
    '''
    Args:
        y: {'labels': ['A', 'B'], 'bboxes': [[10, 20, 100, 200], [20, 40, 50, 80]]}
        or
        y: {'labels': ['A', 'B'], 'bboxes': [[10, 20, 100, 200, 0.5], [20, 40, 50, 80, 1.0]]}
        iou_threshold(float)
    Returns:
        y
    '''
    import warnings
    warnings.warn('`y_nms` is deprecated and will be removed, use `ibll.y.nms() or BBoxLabel(y).nms()` instead.')

    import copy
    y = copy.deepcopy(y)

    bboxes = y['bboxes']
    if not bboxes:
        return y

    bboxes = np.array(bboxes)
    scores = None
    if bboxes.shape[1] == 5:
        scores = bboxes[:, -1]
        bboxes = bboxes[:, :4]

    keep = nms(bboxes, scores, iou_threshold=iou_threshold)
    new_y = dict()
    for key, item in y.items():
        new_y[key] = [item[i] for i in keep]
    return new_y

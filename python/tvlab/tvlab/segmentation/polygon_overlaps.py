'''
Copyright (C) 2023 TuringVision

'''
import numpy as np

__all__ = ['polygon_overlaps', 'polygon_nms']


def polygon_overlaps(polygons1, polygons2, mode='iou'):
    """Calculate the ious between each polygon of polygons1 and polygons2.

    Args:
        polygons1(list): (n, X) [[x1,y1, x2,y2, ...] , [....], ...]
        polygons2(list): (k, X) [[x1,y1, x2,y2, ...] , [....], ...]
        mode(str): iou (intersection over union) or iof (intersection
            over foreground)

    Returns:
        ious(ndarray): shape (n, k)
    """
    from shapely.geometry import Polygon

    assert mode in ['iou', 'iof']

    rows = len(polygons1)
    cols = len(polygons2)
    ious = np.zeros((rows, cols), dtype=np.float32)
    polygons1 = [Polygon(np.array(polygon).reshape(-1, 2)).buffer(0) for polygon in polygons1]
    polygons2 = [Polygon(np.array(polygon).reshape(-1, 2)).buffer(0) for polygon in polygons2]
    area1 = np.array([p.area for p in polygons1])
    area2 = np.array([p.area for p in polygons2])
    for i in range(rows):
        p1 = polygons1[i]
        overlap = np.array([p1.intersection(p2).area for p2 in polygons2])
        if mode == 'iou':
            union = area1[i] + area2 - overlap
        else:
            union = area1[i]
        ious[i, :] = overlap / union
    return ious


def polygon_nms(polygons, scores=None, iou_threshold=0.5):
    from shapely.geometry import Polygon
    if len(polygons) == 0:
        return []

    if scores is None:
        scores = np.ones(len(polygons))

    assert len(scores) == len(polygons)
    polygons = [Polygon(np.array(polygon).reshape(-1, 2)).buffer(1) for polygon in polygons]

    pick = []
    areas = np.array([p.area for p in polygons])

    idxs = np.argsort(scores)

    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # compute the ratio of overlap
        overlap = []
        for j in range(last):
            k = idxs[j]
            pa = polygons[i]
            pb = polygons[k]
            pc = pa.intersection(pb)
            overlap.append(pc.area/areas[k])

        overlap = np.array(overlap)

        # delete all indexes from the index list that have
        need_del = np.where(overlap > iou_threshold)[0]
        idxs = np.delete(idxs, np.concatenate(([last], need_del)))

    return pick

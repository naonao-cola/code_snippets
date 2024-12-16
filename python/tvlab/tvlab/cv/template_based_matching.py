"""
Copyright (C) 2023 TuringVision

a vision tool  for configuring and performing template based matching.
"""

__all__ = ['TemplateBasedMatching']

import numpy as np
import cv2
import yaml


class TemplateBasedMatching:
    """
    Using template_match to check whether there have the object ot not,
    """
    def __init__(self):
        self.templates = {}

    def add(self, template, class_id, method=cv2.TM_CCOEFF_NORMED):
        """
        templates: 8 bit ndarray1
        class_ids: str 'a'
        method:
            cv2.TM_SQDIFF_NORMED
            cv2.TM_CCORR_NORMED
            cv2.TM_CCOEFF_NORMED
        """
        assert len(template.shape) == 2
        self.templates[class_id] = [template, method]

    def save(self, save_path):
        with open(save_path, 'wt', encoding='utf-8') as fp:
            config = {'templates': self.templates}
            yaml.dump(config, fp)

    def load(self, load_path):
        with open(load_path, 'rt', encoding='utf-8') as fp:
            config = yaml.load(fp, Loader=yaml.UnsafeLoader)
            self.templates = config['templates']

    def find(self, img, class_ids='default', score_threshold=90, iou_threshold=0.5, topk=-1):
        """
        img: 8 bit ndarray
        class_ids: list of class_ids, ['class_id0', class_id1, ...]
        score_threshold: only returns score > score_threshold
        iou_threshold: using nms to delete some bboxes
        topk: return resluts which the top k bboxes (descending order), if topk = -1, returns all bboxes

        output:
        a python dictionary
        key：class_id {'class_id0':[...], 'class_id1':[...], ...}
        value: list of ndarray: [[x y w h socre] ....]
        """
        from torchvision.ops import nms
        import torch

        assert len(img.shape) == 2
        if class_ids == 'default':
            class_ids = [class_id for class_id in self.templates]

        result = dict()
        for class_id in class_ids:
            result[class_id] = []
            template, method = self.templates[class_id]
            w, h = template.shape[::-1]
            res = cv2.matchTemplate(img, template, method)
            loc = np.where(res >= score_threshold / 100)
            for pt in zip(*loc[::-1]):
                result[class_id].append(
                    [pt[0], pt[1], pt[0] + w, pt[1] + h, res[pt[0]][pt[1]]])

            if len(result[class_id]) == 0:
                continue
            dets = np.array(result[class_id])
            bboxes = dets[:, :4]
            scores = dets[:, 4]
            keep = nms(torch.from_numpy(bboxes),
                       torch.from_numpy(scores), iou_threshold)
            keep = keep.numpy()
            dets = dets[keep]
            dets[:, 2], dets[:, 3] = dets[:, 2] - dets[:, 0], dets[:, 3] - dets[:, 1]
            dets[:, 0], dets[:, 1] = dets[:, 0] + dets[:, 2] / 2, dets[:, 1] + dets[:, 3] / 3
            result[class_id] = dets

            if topk != -1 and len(result[class_id]) > topk:
                result[class_id] = result[class_id][:topk]

        return result

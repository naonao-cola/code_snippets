'''
Copyright (C) 2023 TuringVision

do perspective
'''
import numpy as np
import cv2
from .geometry import *

__all__ = ['Perspective']


class Perspective:
    def __init__(self, src_pts, dst_wh):
        '''
        do perspective
        In:
            src_pts: (lt, rt, lb, rb)
                    lt,rt,lb,rb: (x, y)
            wh: (width, height)
        '''
        self.dst_wh = dst_wh
        dst_w, dst_h = dst_wh
        pts1 = np.array(src_pts, dtype=np.float32)
        pts2 = np.float32([[0, 0], [dst_w, 0], [0, dst_h], [dst_w, dst_h]])
        self.m = cv2.getPerspectiveTransform(pts1, pts2)
        self.invert_m = cv2.getPerspectiveTransform(pts2, pts1)

    @classmethod
    def from_points(cls, src_pts, dst_wh):
        '''
         do perspective
        :param src_pts: vector of 2D points
            list [(x1,y1), (x2,y2), ...]
            or
            np.array (N, 2)
        :param dst_wh: (width, height)
        :return:
        '''
        dst_w, dst_h = dst_wh
        rect = cv2.minAreaRect(src_pts)
        box = cv2.boxPoints(rect)
        bbox = Region.from_polygon(box).min_bbox()
        l1 = Point(bbox.coords()[0]).distance(Point(bbox.coords()[1]))
        l2 = Point(bbox.coords()[1]).distance(Point(bbox.coords()[2]))
        if l2 > l1:
            if dst_w > dst_h:
                pts1 = np.vstack([bbox.coords()[1:4, :], bbox.coords()[0]])
            else:
                pts1 = bbox.coords()
        else:
            if dst_w > dst_h:
                pts1 = bbox.coords()
            else:
                pts1 = np.vstack([bbox.coords()[1:4, :], bbox.coords()[0]])
        pts1 = [pts1[0], pts1[1], pts1[3], pts1[2]]
        return cls(pts1, dst_wh)

    def tfm_xy(self, xy, invert=False):
        '''
        In:
            xy: (x, y)
            invert: True: convert from dst to src
                    False: convert from src to dst
        Out:
            x,y
        '''
        m = self.invert_m if invert else self.m
        u, v = xy
        c = m.flatten()
        den = c[6] * u + c[7] * v + 1.0
        x = (c[0] * u + c[1] * v + c[2]) / den
        y = (c[3] * u + c[4] * v + c[5]) / den
        return x, y

    def tfm_pts(self, pts, invert=False):
        '''
        In:
            pts: [(x, y), (x, y), ...]
            invert: True: convert from dst to src
                    False: convert from src to dst
        Out:
            pts [(x, y), (x, y), ...]
        '''
        m = self.invert_m if invert else self.m
        pts = np.array(pts).reshape(-1, 2)
        u = pts[:, 0]
        v = pts[:, 1]
        c = m.flatten()
        den = c[6] * u + c[7] * v + 1.0
        x = (c[0] * u + c[1] * v + c[2]) / den
        y = (c[3] * u + c[4] * v + c[5]) / den
        pts[:, 0] = x
        pts[:, 1] = y
        return pts.tolist()

    def tfm_bbox_label(self, bbox_label, invert=False):
        '''
        in:
            bbox_label: BBoxLabel
            invert: True: convert from dst to src
                    False: convert from src to dst
        out:
            BBoxLabel
        '''

        def _tfm(box):
            l, t, r, b = box[:4]
            lt = self.tfm_xy((l, t), invert)
            lb = self.tfm_xy((l, b), invert)
            rt = self.tfm_xy((r, t), invert)
            rb = self.tfm_xy((r, b), invert)
            l = min([lt[0], lb[0], rt[0], rb[0]])
            t = min([lt[1], lb[1], rt[1], rb[1]])
            r = max([lt[0], lb[0], rt[0], rb[0]])
            b = max([lt[1], lb[1], rt[1], rb[1]])
            return [l, t, r, b] + box[4:]

        return bbox_label.tfm_bbox(_tfm)

    def tfm_polygon_label(self, polygon_label, invert=False):
        '''
        in:
            polygon_label: PolygonLabel
            invert: True: convert from dst to src
                    False: convert from src to dst
        out:
            PolygonLabel
        '''

        def _tfm(p):
            all_xy = p
            c = []
            if len(all_xy) % 2 == 1:
                all_xy = all_xy[:-1]
                c = all_xy[-1:]
            all_xy = self.tfm_pts(all_xy, invert)
            return np.array(all_xy).flatten().tolist() + c

        return polygon_label.tfm_polygon(_tfm)

    def tfm_img(self, img, **kwargs):
        '''
        In:
            img (np.ndarray)
        Out:
            img (np.ndarray)
        '''
        return cv2.warpPerspective(img, self.m, self.dst_wh, **kwargs)

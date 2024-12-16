'''
Copyright (C) 2023 TuringVision

List of label suitable for computer vision instance segmentation task.
'''
import os
import os.path as osp
import numpy as np
from .polygon_overlaps import polygon_nms, polygon_overlaps
from ..utils import polygon_to_bbox, obj_from_json, obj_to_json
from ..ui import plot_stack_bar, plot_bokeh_histogram

__all__ = ['PolygonLabel', 'PolygonLabelList']


class PolygonLabel(dict):
    '''
    polygon_label (dict): {'labels': ['A', 'B'], 'polygons': [[10, 20, 100, 150, 50, 60, ...], [...]]}
        polygons: [x1, y1, x2, y2, x3, y3, ...]
    '''
    def __init__(self, polygon_label=None):
        if polygon_label is None:
            polygon_label = {'labels': [], 'polygons': []}
        super().__init__(polygon_label)

    def __add__(self, polygon_label):
        import copy
        polygon_label = PolygonLabel(polygon_label)
        if len(self.labels()) == 0:
            return copy.deepcopy(polygon_label)
        new = copy.deepcopy(self)
        for key, v in new.items():
            if key in polygon_label:
                v += polygon_label[key]
            else:
                v += [None] * len(polygon_label['labels'])
        return new

    def __getitem__(self, y):
        if isinstance(y, int):
            item = {}
            for k, v in self.items():
                k = k[:-1]
                item[k] = v[y]
            return item

        return super().__getitem__(y)

    def __setitem__(self, key, value):
        if isinstance(key, int):
            item = {}
            for k, v in self.items():
                k = k[:-1]
                v[key] = value[k]
            return;

        return super().__setitem__(key, value)

    def pop(self, key):
        if isinstance(key, int):
            item = {}
            for k, v in self.items():
                k = k[:-1]
                item[k] = v.pop(key)
            return item;

        return super().pop(key)

    def append(self, value):
        for k, v in self.items():
            k = k[:-1]
            v.append(value[k])

    def deepcopy(self):
        import copy
        return copy.deepcopy(self)

    def labels(self):
        return self['labels']

    def polygons(self):
        return self['polygons']

    def bboxes(self):
        return [polygon_to_bbox(polygon) for polygon in self.polygons()]

    def nms(self, iou_threshold=0.5):
        '''
        iou_threshold (float)
        '''
        polygons = self.polygons()
        if not polygons:
            return self

        scores = []
        pure_polygons = []
        for polygon in polygons:
            score = 1.0
            if len(polygon) % 2 == 1:
                score = polygon[-1]
                polygon = polygon[:-1]
            scores.append(score)
            pure_polygons.append(polygon)

        keep = polygon_nms(pure_polygons, scores, iou_threshold=iou_threshold)
        new_label = dict()
        for key, item in self.items():
            new_label[key] = [item[i] for i in keep]
        return PolygonLabel(new_label)

    def tfm(self, tfm):
        '''
        tfm (callable):
            eg:
                def tfm(label, polygon):
                    ...
                    return label, polygon
        '''
        labels, polygons = [], []
        for label, polygon in zip(self.labels(), self.polygons()):
            label, polygon = tfm(label, polygon)
            labels.append(label)
            polygons.append(polygon)

        new_poly_label = self.deepcopy()
        new_poly_label["labels"] = labels
        new_poly_label["polygons"] = polygons
        return new_poly_label

    def tfm_item(self, tfm):
        '''
        tfm (callable):
            eg:
                def tfm(item):
                    item['label'] = xxx
                    item['polygon'] = xxx
                    ...
                    return item
        '''
        new = self.deepcopy()
        for i in range(len(self.labels())):
            new[i] = tfm(new[i])
        return new

    def tfm_label(self, tfm):
        '''
        tfm (callable):
            eg:
                def tfm(label):
                    ...
                    return label
        '''
        labels = []
        for label in self.labels():
            label = tfm(label)
            labels.append(label)

        new_poly_label = self.deepcopy()
        new_poly_label["labels"] = labels
        return new_poly_label

    def tfm_polygon(self, tfm, without_score=False):
        '''
        tfm (callable):
            eg:
                def tfm(polygon):
                    ...
                    return polygons
        '''
        polygons = []
        for polygon in self.polygons():
            if without_score:
                score = []
                if len(polygon)%2 == 1:
                    polygon = polygon[:-1]
                    score = polygon[-1:]
                polygon = tfm(polygon) + score
            else:
                polygon = tfm(polygon)
            polygons.append(polygon)

        new_poly_label = self.deepcopy()
        new_poly_label["polygons"] = polygons
        return new_poly_label

    def crop(self, box, iof_threshold=0.3):
        polygons = self.polygons()
        if not polygons:
            return PolygonLabel()

        labels = self.labels()
        l,t,r,b = box
        tpolygons = [[l,t,r,t,r,b,l,b]]
        ppolygons = []
        for p in polygons:
            if len(p) % 2 == 1:
                p = p[:-1]
            ppolygons.append(p)
        overlaps = polygon_overlaps(ppolygons, tpolygons, 'iof')
        overlaps = overlaps[:, 0]
        new_labels = []
        new_polygons = []
        crop_w = box[2] - box[0]
        crop_h = box[3] - box[1]
        if overlaps.max() > iof_threshold:
            for j, iof in enumerate(overlaps.tolist()):
                if iof > iof_threshold:
                    new_labels.append(labels[j])
                    new_polygons.append(polygons[j])

        def _clip(p):
            p = np.array(p).reshape(-1, 2)
            p[:, 0] = (p[:, 0] - box[0]).clip(0, crop_w)
            p[:, 1] = (p[:, 1] - box[1]).clip(0, crop_h)
            return p.flatten().tolist()

        new_p = PolygonLabel({'labels': new_labels, 'polygons': new_polygons})
        return new_p.tfm_polygon(_clip, True)

    def to_labelme(self, img_path, out_dir):
        from PIL import Image

        file_name = osp.basename(img_path)
        os.makedirs(out_dir, exist_ok=True)

        labels = self.labels()
        polygons = self.polygons()
        try:
            with Image.open(img_path) as im:
                image_shape = (im.width, im.height)

            shapes = []
            for l, polygon in zip(labels, polygons):
                if len(polygon)%2 == 1:
                    polygon = polygon[:-1]
                shape_info = {'label': l,
                              'lineColor': None,
                              'fillColor': None,
                              'points': np.array(polygon).reshape(-1, 2).tolist(),
                              'shape_type': 'polygon',
                             }
                shapes.append(shape_info)

            info = {'version': '3.16.4',
                    'flags': {},
                    'shapes': shapes,
                    'lineColor': [0, 255, 0, 128],
                    'fillColor': [255, 0, 0, 128],
                    'imagePath': img_path,
                    'imageHeight': image_shape[1],
                    'imageWidth': image_shape[0]}
            out_path = osp.join(out_dir, file_name[:file_name.rindex('.')]+'.json')
            obj_to_json(info, out_path, ensure_ascii=False)
        except Exception as e:
            print(e)

    def to_disk(self, img_path, out_dir):
        return self.to_labelme(img_path, out_dir)

    @classmethod
    def from_labelme(cls, path):
        import math
        polygons = list()
        labels = list()
        info = obj_from_json(path)
        shapes = info.get('shapes', [])
        for shape in shapes:
            points = shape.get('points', [])
            polygon = np.array(points).flatten().tolist()
            if polygon:
                polygons.append(polygon)
                label = shape.get('label', 'object')
                labels.append(label)
        return PolygonLabel({'labels': labels, 'polygons': polygons})

    @classmethod
    def from_disk(cls, path):
        return cls.from_labelme(path)

    def offset(self, x_off, y_off):
        def _add_offset(p):
            p = np.array(p).reshape(-1, 2)
            p[:, 0] += x_off
            p[:, 1] += y_off
            return p.flatten().tolist()
        return self.tfm_polygon(_add_offset, True)

    def scale(self, x_scale, y_scale):
        def _scale(p):
            p = np.array(p).reshape(-1, 2)
            p[:, 0] *= x_scale
            p[:, 1] *= y_scale
            return p.flatten().tolist()
        return self.tfm_polygon(_scale, True)

    def filter(self, key):
        '''
        key (callable):
            eg:
                def key(label, polygon):
                    ...
                    return True or False (True for keep)
        '''
        keep = []
        for i, (label, polygon) in enumerate(zip(self.labels(), self.polygons())):
            ret = key(label, polygon)
            if ret:
                keep.append(i)

        new_label = dict()
        for key, item in self.items():
            new_label[key] = [item[i] for i in keep]

        return PolygonLabel(new_label)

    def filter_item(self, key):
        '''
        key (callable):
            eg:
                def key(item):
                    label = item['label']
                    polygon = item['polygon']
                    ...
                    return True or False (True for keep)
        '''
        keep = []
        for i in range(len(self.labels())):
            ret = key(self[i])
            if ret:
                keep.append(i)

        new_label = dict()
        for key, item in self.items():
            new_label[key] = [item[i] for i in keep]

        return PolygonLabel(new_label)

    def filter_by_label(self, key):
        '''
        key (callable):
            eg:
                def key(label):
                    ...
                    return True or False (True for keep)
        '''
        def _label_filter(label, polygon):
            return key(label)

        return self.filter(_label_filter)

    def filter_by_polygon(self, key):
        '''
        key (callable):
            eg:
                def key(polygon):
                    ...
                    return True or False (True for keep)
        '''
        def _polygon_filter(label, polygon):
            return key(polygon)

        return self.filter(_polygon_filter)

    def filter_by_bbox(self, key):
        '''
        key (callable):
            eg:
                def key(box):
                    ...
                    return True or False (True for keep)
        '''
        def _polygon_filter(label, polygon):
            return key(polygon_to_bbox(polygon))

        return self.filter(_polygon_filter)


class PolygonLabelList(list):
    '''
    polygon_label_list (list): [{'labels': ['A', 'B'], 'polygons': [[10, 20, 100, 150, 50, 60, ...], [...]]}, ...]

    '''
    def __init__(self, polygon_label_list):
        super().__init__([PolygonLabel(polygon_label) for polygon_label in polygon_label_list])

    def labels(self):
        return [l for polygon_label in self for l in polygon_label['labels']]

    def polygons(self):
        return [polygon for polygon_label in self for polygon in polygon_label['polygons']]

    def labelset(self):
        return sorted(list(set(self.labels())))

    def nms(self, iou_threshold=0.5):
        '''
        iou_threshold (float)
        '''
        return PolygonLabelList([polygon_label.nms(iou_threshold=iou_threshold) for polygon_label in self])

    def tfm(self, tfm):
        '''
        tfm (callable):
            eg:
                def tfm(label, polygon):
                    ...
                    return label, polygon
        '''
        return PolygonLabelList([polygon_label.tfm(tfm=tfm) for polygon_label in self])

    def tfm_item(self, tfm):
        '''
        tfm (callable):
            eg:
                def tfm(item):
                    item['label'] = xxx
                    item['polygon'] = xxx
                    ...
                    return item
        '''
        return PolygonLabelList([polygon_label.tfm_item(tfm=tfm) for polygon_label in self])

    def tfm_label(self, tfm):
        '''
        tfm (callable):
            eg:
                def tfm(label):
                    ...
                    return label
        '''
        return PolygonLabelList([polygon_label.tfm_label(tfm=tfm) for polygon_label in self])

    def tfm_polygon(self, tfm):
        '''
        tfm (callable):
            eg:
                def tfm(polygon):
                    ...
                    return polygons
        '''
        return PolygonLabelList([polygon_label.tfm_polygon(tfm=tfm) for polygon_label in self])

    def filter(self, key):
        '''
        key (callable):
            eg:
                def key(label, polygon):
                    ...
                    return True or False (True for keep)
        '''
        return PolygonLabelList([polygon_label.filter(key=key) for polygon_label in self])

    def filter_item(self, key):
        '''
        key (callable):
            eg:
                def key(item):
                    label = item['label']
                    polygon = item['polygon']
                    ...
                    return True or False (True for keep)
        '''
        return PolygonLabelList([polygon_label.filter_item(key=key) for polygon_label in self])

    def filter_by_label(self, key):
        '''
        key (callable):
            eg:
                def key(label):
                    ...
                    return True or False (True for keep)
        '''
        return PolygonLabelList([polygon_label.filter_by_label(key=key) for polygon_label in self])

    def filter_by_polygon(self, key):
        '''
        key (callable):
            eg:
                def key(polygon):
                    ...
                    return True or False (True for keep)
        '''
        return PolygonLabelList([polygon_label.filter_by_polygon(key=key) for polygon_label in self])

    def show_bbox_dist(self, labelset=None, x='size', y='ratio',
                       xbins=20, ybins=20, need_show=True):
        ''' show distribution of polygons bbox
        labelset (list): only show class label in labelset
        x (str): one of ('ratio', 'size', 'area', 'w', 'h', 'x', 'y')
        y (str): one of ('ratio', 'size', 'area', 'w', 'h', 'x', 'y')
        '''
        polygon_label_list = self
        if labelset:
            polygon_label_list = self.filter_by_label(lambda label: label in labelset)

        polygons = [polygon for polygon in polygon_label_list.polygons() if polygon]
        if len(polygons) <= 0:
            print('no polygons!')
            return

        bboxes = np.array([polygon_to_bbox(polygon) for polygon in polygons])

        all_w = bboxes[:, 2] - bboxes[:, 0]
        all_h = bboxes[:, 3] - bboxes[:, 1]
        all_x = (bboxes[:, 2] + bboxes[:, 0])/2
        all_y = (bboxes[:, 3] + bboxes[:, 1])/2

        _data_map = {'w': all_w, 'h': all_h, 'area': all_h * all_w,
                     'size': np.sqrt(all_h*all_w), 'ratio': all_h/all_w,
                     'x': all_x, 'y': all_y}

        data = {x: _data_map[x], y: _data_map[y]}
        return plot_bokeh_histogram('area=h*w, ratio=h/w, size=sqrt(h*w)',
                                    data, xlabel=x, ylabel=y,
                                    xbins=xbins, ybins=ybins)

    def show_dist(self, need_show=True):
        '''show distribution
        '''
        labelset = self.labelset()
        labelset.append('backgroud')

        all_labels = self.labels()
        all_labels += ['backgroud' for polygon_label in self if not polygon_label['labels']]
        count_data = {'labelset': labelset,
                      'total': [all_labels.count(l) for l in labelset]}

        return plot_stack_bar('label distribution', count_data, 'labelset', ['total'],
                              need_show=need_show)

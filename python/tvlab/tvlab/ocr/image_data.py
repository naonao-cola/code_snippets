'''
Copyright (C) 2023 TuringVision

List of image and label suitable for computer vision instances segmentation task.
'''
import os
import os.path as osp
import sys
import numpy as np
from PIL import Image
import math
import copy
import cv2

from ..category import ImageLabelList, get_image_files, get_files
from ..segmentation.polygon_label import PolygonLabelList
from ..segmentation.image_data import ImagePolygonLabelList

__all__ = ['ImageOCRPolygonLabelList', 'imgaug_img_ocr_polygon_tfm']


def load_one_mtwi(path, image_name):
    item = {
        'labels': [],
        'polygons': []
    }

    reader = open(path, 'r').readlines()
    for line in reader:
        try:
            parts = line.strip().split(',')
            label = ""
            if len(parts) == 9:
                line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in parts]
                poly = np.array(list(map(float, line[:8]))).tolist()
                label = parts[-1]
            else:
                num_points = math.floor((len(line) - 1) / 2) * 2
                poly = np.array(list(map(float, line[:num_points]))).tolist()
                label = ",".join(parts[num_points:])

            if label == '1':
                label = '###'

            item['polygons'].append(poly)
            item['labels'].append(label)
        except:
            print("Load image error! {}".format(path))
    return item

def imgaug_ocr_polygon_transform(aug, polygons, shape, fraction=0.5):
    from shapely import geometry
    import imgaug
    from imgaug.augmentables.polys import Polygon, PolygonsOnImage

    conf_list = []
    true_polygons = []
    for polygon in polygons:
        conf = []
        if len(polygon) % 2 == 1:
            conf = polygon[-1:]
            polygon = polygon[:-1]
        conf_list.append(conf)
        true_polygons.append(polygon)

    polys = [Polygon(np.array(polygon).reshape(-1, 2)) for polygon in true_polygons]
    psoi = PolygonsOnImage(polys, shape)
    psoi_aug = aug.augment_polygons([psoi])[0]

    mask = []
    polys = []
    h, w = psoi_aug.shape[:2]
    poly_image = geometry.Polygon([(0, 0), (w, 0), (w, h), (0, h)])

    for i, poly in enumerate(psoi_aug.polygons):
        poly_shapely = poly.to_shapely_polygon().buffer(0)
        multipoly_inter_shapely = poly_shapely.intersection(poly_image)
        ignore_types = (geometry.LineString,
                        geometry.MultiLineString,
                        geometry.point.Point,
                        geometry.MultiPoint,
                        geometry.GeometryCollection)
        if isinstance(multipoly_inter_shapely, geometry.Polygon):
            multipoly_inter_shapely = geometry.MultiPolygon(
                [multipoly_inter_shapely])
        elif isinstance(multipoly_inter_shapely,
                        geometry.MultiPolygon):
            pass
        elif isinstance(multipoly_inter_shapely, ignore_types):
            # polygons that become (one or more) lines/points after clipping
            # are here ignored
            multipoly_inter_shapely = geometry.MultiPolygon([])
        else:
            raise Exception(
                "Got an unexpected result of type %s from Shapely for "
                "image (%d, %d) and polygon %s. This is an internal error. "
                "Please report." % (
                    type(multipoly_inter_shapely), h, w, poly.exterior)
            )

        max_area = 0
        clip_poly = None
        for poly_inter_shapely in multipoly_inter_shapely.geoms:
            area = poly_inter_shapely.area
            if area > max_area:
                max_area = area
                clip_poly = Polygon.from_shapely(poly_inter_shapely, label=poly.label)
        if clip_poly and (1 - max_area / poly_shapely.area) < fraction:
            polys.append(clip_poly)
            mask.append(i)

    conf_list = [conf_list[i] for i in mask]
    ret_polys = [np.array(poly.exterior).flatten().tolist() + c
                 for poly, c in zip(polys, conf_list)]
    return ret_polys, mask

def imgaug_img_ocr_polygon_tfm(x, y, aug, fraction=0.5):
    '''
    Do img and polygons transform for imgaug augmenter.

    In:
        x: image array
        y: {'labels': ['A', 'B'], 'polygons': [[x1, y1, x2, y2, ...], [x1, y1, x2, y2, ...]]}
        aug: augmenter for imgaug
        fraction [0.0 ~ 1.0]: remove all polygons with an out of image fraction of at least fraction
    Out:
        x, y
    '''
    import copy

    img_shape = x.shape
    y = copy.deepcopy(y)
    aug_det = aug.to_deterministic()
    x = aug_det.augment_images([x])[0]
    if y and y['polygons'] and y['labels']:
        y['polygons'], mask = imgaug_ocr_polygon_transform(aug_det,
                       y['polygons'],
                       img_shape,
                       fraction)
        for k, v in y.items():
            if k in ["labels", "ignore_tags"]:
                y[k] = [v[i] for i in mask]
    return x, y


class ImageOCRPolygonLabelList(ImagePolygonLabelList):
    '''
    label_list: (list) list of label
    label_list (list): [
        {
            'labels': ['A', 'B'],
            'polygons': [[10, 20, 100, 150, 50, 60, ...], [...]],
        },
        ...
    ]
    polygons: [x1, y1, x2, y2, x3, y3, ...]
    '''
    def __init__(self, img_path_list, label_list=None):
        super().__init__(img_path_list)
        if label_list is None:
            y = PolygonLabelList([None for _ in img_path_list])
        else:
            y = PolygonLabelList(label_list)
        self.y = y

    @classmethod
    def from_label_info(cls, image_dir, label_info_dict):
        raise NotImplementedError

    def to_label_info(self, class_list=None):
        raise NotImplementedError

    @classmethod
    def from_pascal_voc(cls, image_dir, xml_dir=None, check_ext=True, recurse=True, followlinks=False):
        '''
        image_dir:
        xml_dir: directory of xml, find xml in image_dir if xml_dir is None
        '''
        raise NotImplementedError

    def to_pascal_voc(self, out_path):
        raise NotImplementedError

    def to_mmdet_pkl(self, pkl_path):
        raise NotImplementedError

    def tile(self, tile_shape, overlap, out_dir, img_format='png',
             img_quality=95, iof_threshold=0.3, workers=8):
        raise NotImplementedError

    def show_sample(self, count=4, idx=None, ncols=2, figsize=(9,9), **kwargs):
        raise NotImplementedError

    def limit_bboxes_min_size(self, min_size=32):
        ''' limit min of width and height to min_size
        '''
        raise NotImplementedError

    @classmethod
    def from_mtwi(cls, image_dir, txt_dir=None, check_ext=True, recurse=True, followlinks=False):
        '''
        image_dir:
        txt_dir: directory of txt, find txt in image_dir if txt_dir is None
        '''

        if not txt_dir:
            txt_dir = image_dir

        image_dir = osp.normpath(image_dir)
        txt_dir = osp.normpath(txt_dir)

        img_path_list = get_image_files(image_dir, check_ext=check_ext,
                                        recurse=recurse, followlinks=followlinks)
        txt_path_list = get_files(txt_dir, extensions=['.txt'],
                                  recurse=recurse, followlinks=followlinks)
        txt_name_list = [osp.splitext(osp.basename(txt))[0] for txt in txt_path_list]

        label_info = list()
        label_set = set()
        for img_path in img_path_list:
            img_name = osp.splitext(osp.basename(img_path))[0]
            match_cnt = txt_name_list.count(img_name)
            txt_path = None
            if match_cnt == 1:
                txt_path = txt_path_list[txt_name_list.index(img_name)]
            elif match_cnt > 1:
                all_find_txt = [txt_path_list[i] for i, txt_name in enumerate(txt_name_list)
                                if txt_name == img_name]
                for find_txt_path in all_find_txt:
                    txt_path_suffix = osp.dirname(find_txt_path[len(txt_dir):])
                    img_path_suffix = osp.dirname(img_path[len(image_dir):])
                    if txt_path_suffix in img_path_suffix or img_path_suffix in txt_path_suffix:
                        txt_path = find_txt_path
                        break
            bboxes = list()
            item = load_one_mtwi(txt_path, img_name)
            label_info.append(item)

        return cls(img_path_list, label_info)

    def dbnet_data(self, cfg, train_tfms=None, valid_tfms=None, path='.',
                   bs=4, num_workers=2, show_dist=True):
        from .db_process import reader_db_data
        os.makedirs(path, exist_ok=True)

        self.clear_tfms()
        train, valid = self.split(show=show_dist)
        if train_tfms:
            train.set_tfms(train_tfms)

        if valid_tfms:
            valid.set_tfms(valid_tfms)

        train_dataset = reader_db_data(train, mode="train", config={'num_workers': num_workers, "train_batch_size_per_card": bs,})
        valid_dataset = reader_db_data(valid, mode="eval", config={"test_batch_size_per_card": bs,})
        return [train_dataset, valid_dataset]

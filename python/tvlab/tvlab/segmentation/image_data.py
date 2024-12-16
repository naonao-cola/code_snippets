'''
Copyright (C) 2023 TuringVision

List of image and label suitable for computer vision instances segmentation task.
'''
import os
import os.path as osp
import math
import numpy as np
from PIL import Image
from ..utils import *
from ..category import ImageLabelList, get_image_files, get_files, save_image, get_image_shape
from ..ui import show_images, plot_stack_bar
from .polygon_label import PolygonLabelList
from ..detection.image_data import ImageBBoxLabelList

__all__ = ['ImagePolygonLabelList', 'imgaug_img_polygon_tfm']



def _rgb2bgr(x):
    return x[:, :, ::-1]


def imgaug_polygon_transform(aug, polygons, shape, fraction=0.5):
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
        try:
            poly_shapely = poly.to_shapely_polygon().buffer(0)
        except ValueError:

            continue
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


def imgaug_img_polygon_tfm(x, y, aug, fraction=0.5):
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
        y['polygons'], mask = imgaug_polygon_transform(aug_det,
                                                       y['polygons'],
                                                       img_shape,
                                                       fraction)
        for k, v in y.items():
            if k != 'polygons':
                y[k] = [v[i] for i in mask]
    return x, y


class MMCustomDataset:
    def __init__(self, ipll, labelset, pipeline, test_mode=False):
        from mmseg.datasets.pipelines import Compose
        self.CLASSES = labelset
        self.ipll = ipll
        self.test_mode = test_mode
        self.img_infos = self.load_annotations()
        self.pipeline = Compose(pipeline)

    def load_annotations(self):
        pkl_label_info = []
        labelset = self.CLASSES
        for image_path in self.ipll.x:
            height, width = get_image_shape(image_path, self.ipll.cache_x, self.ipll.cache_img)[0:2]

            pkl_one = {
                'filename': osp.basename(image_path),
                'width': width,
                'height': height,
            }
            pkl_label_info.append(pkl_one)
        return pkl_label_info

    def __len__(self):
        return len(self.img_infos)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_img(idx)
        while True:
            data = self.prepare_img(idx)
            return data

    def pre_pipeline(self, results):
        results['seg_fields'] = []
        results['flip'] = False
        results['flip_direction'] = 'horizontal'

    def prepare_img(self, idx):
        from PIL import Image, ImageDraw
        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        self.pre_pipeline(results)

        img, label = self.ipll[idx]
        results['filename'] = self.ipll.x[idx]
        results['ori_filename'] = osp.basename(results['filename'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['pad_shape'] = img.shape
        ori_shape = list(img.shape)
        ori_shape[0] = img_info['height']
        ori_shape[1] = img_info['width']
        results['ori_shape'] = tuple(ori_shape)
        h_scale = img.shape[0] / ori_shape[0]
        w_scale = img.shape[1] / ori_shape[1]
        results['scale_factor'] = np.array([w_scale, h_scale, w_scale, h_scale],
                                           dtype=np.float32)
        labels = label['labels']
        polygons = label['polygons']
        labelset = self.CLASSES
        img = Image.new('L', (img.shape[1], img.shape[0]), 0)
        mask = np.array(img)
        img_draw = ImageDraw.Draw(img)
        for l, p in zip(labels, polygons):
            img_draw.polygon(p, outline=1, fill=labelset.index(l) + 1)
            mask = np.array(img)
        results['gt_semantic_seg'] = mask
        results['seg_fields'].append('gt_semantic_seg')
        results = self.pipeline(results)
        return results


def _label_idx(classes, label):
    return classes.index(label) + 1 if label in classes and label else 0


def _xy_to_tensor(img, gt, idx, img_path, labelset, cache_x, cache_img):
    import torch
    from torchvision.transforms.functional import to_tensor
    from PIL import Image, ImageDraw

    num_objs = len(gt['labels'])
    img_h, img_w = img.shape[:2]

    if num_objs > 0:
        masks = np.zeros((num_objs, img_h, img_w), dtype=np.uint8)
        for i in range(num_objs):
            mask = Image.fromarray(masks[i])
            mask_draw = ImageDraw.Draw(mask)
            mask_draw.polygon(gt['polygons'][i], outline=1, fill=1)
            masks[i] = np.array(mask)
        masks = torch.from_numpy(masks)

        bboxes = [polygon_to_bbox(polygon) for polygon in gt['polygons']]
        boxes = torch.as_tensor([box[:4] for box in bboxes],
                                dtype=torch.float32)
        # there is only one class
        labels = torch.as_tensor(
            [_label_idx(labelset, l) for l in gt['labels']], dtype=torch.int64)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    else:
        masks = torch.zeros((0, img_h, img_w), dtype=torch.uint8)
        boxes = torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.zeros((0, ), dtype=torch.int64)
        area = torch.zeros((0, ), dtype=torch.float32)
    # suppose all instances are not crowd
    iscrowd = torch.zeros((num_objs, ), dtype=torch.int64)

    image_id = torch.tensor([idx])
    target = {}
    target["boxes"] = boxes
    target["labels"] = labels
    target["image_id"] = image_id
    target["area"] = area
    target["iscrowd"] = iscrowd
    target["masks"] = masks
    target["ori_shape"] = get_image_shape(img_path, cache_x, cache_img)

    img = to_tensor(img.copy())
    return img, target


class ImagePolygonLabelList(ImageBBoxLabelList):
    '''
    label_list: (list) list of label
    label_list (list): [{'labels': ['A', 'B'], 'polygons': [[10, 20, 100, 150, 50, 60, ...], [...]]}, ...}
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
        '''create ImageLabelList from label info dict
        '''
        classes = label_info_dict.get('classList', None)
        img_path_list = []
        label_list = []
        for label_info in label_info_dict['labelSet']:
            if 'localImagePath' in label_info:
                img_path = label_info['localImagePath']
            else:
                img_path = osp.join(image_dir, label_info['imageName'])
            labels = label_info['labels']
            polygons_info_list = label_info['polygons']
            polygons = [[int(x) for x in polygon_str.split(",")] for polygon_str in polygons_info_list]
            labels = ['Other' if classes and l not in classes else l for l in labels]
            img_path_list.append(img_path)
            label_list.append({'labels': labels, 'polygons':polygons})
        return cls(img_path_list, label_list)

    def to_label_info(self, class_list=None):
        '''convert ImageLabelList to label info dict
        '''
        label_info_dict = {}
        label_info_dict['classList'] = class_list if class_list is not None else self.labelset()
        labelset = []
        for img_path, label in zip(self.x, self.y):
            polygons = label['polygons']
            polygons_info_list = list()
            for polygon in polygons:
                if len(polygon)%2 == 1:
                    polygon = polygon[:-1]
                polygon_str = ','.join(str(round(x, 1)) for x in polygon)
                polygons_info_list.append(polygon_str)
            label_info = {
                'imageName': osp.basename(img_path),
                'labels': label['labels'],
                'polygons': polygons_info_list
            }
            labelset.append(label_info)
        label_info_dict['labelSet'] = labelset
        return label_info_dict

    @classmethod
    def from_turbox_data(cls, turbox_data):
        '''create ImagePolygonLabelList from turbox format data
        '''
        classes = turbox_data.get('classList', None)
        img_path_list = []
        label_list = []
        for img_info in turbox_data['labelSet']:
            img_path = img_info['imagePath']
            img_path_list.append(img_path)
            polygons = []
            labels = []
            for shape in img_info['shapes']:
                if shape['shapeType'] == 'polygon':
                    polygons.append(shape['points'])
                    labels.append(shape['label'])
                elif shape['shapeType'] == 'rectangle':
                    rc = shape['points']
                    polygons.append([rc[0],rc[1],rc[2],rc[1],rc[2],rc[3],rc[0],rc[3]])
                    labels.append(shape['label'])
                elif shape['shapeType'] == 'circle':
                    c = shape['points']
                    r = np.linalg.norm(np.array(c[:2]) - np.array(c[2:]))
                    polygon = [[c[0]+r*np.cos(deg*np.pi/180), c[1]+r*np.sin(deg*np.pi/180)] for deg in range(360)]
                    polygons.append(sum(polygon, []))
                    labels.append(shape['label'])
            labels = ['Other' if classes and l not in classes else l for l in labels]
            label_list.append({'labels': labels, 'polygons': polygons})
        return cls(img_path_list, label_list)

    def to_turbox_data(self, class_list=None):
        '''convert ImagePolygonLabelList to turbox data
        '''
        turbox_data = {}
        turbox_data['classList'] = class_list if class_list is not None else self.labelset()
        turbox_data['labelSet'] = []
        for img_path, label in zip(self.x, self.y):
            img_info = {
                'imageName': osp.basename(img_path),
                'imagePath': img_path,
                'shapes': []
            }
            for label, polygon in zip(label['labels'], label['polygons']):
                shape = {
                    'label': label,
                    'points': polygon,
                    'confidence': 1,
                    'shapeType': 'polygon'
                }
                img_info['shapes'].append(shape)

            turbox_data['labelSet'].append(img_info)
        return turbox_data

    def get_to_tensor_tfm(self):
        from functools import partial
        labelset = self.labelset()
        cache_x = self.cache_x
        cache_img = self.cache_img
        return partial(_xy_to_tensor, labelset=labelset, cache_x=cache_x, cache_img=cache_img)

    def mmdet_data(self, cfg, train_tfms=None, valid_tfms=None, path='.',
                   bs=2, num_workers=1, show_dist=True):
        raise NotImplementedError

    def mmseg_data(self, cfg, train_tfms=None, valid_tfms=None, path='.',
                   bs=2, num_workers=1, show_dist=True):
        from mmseg.datasets.pipelines import (Normalize, Pad, Collect, ImageToTensor)
        from mmseg.datasets.pipelines.formating import DefaultFormatBundle
        train_pipline = [Normalize(**cfg.img_norm_cfg),
                         Pad(size_divisor=32),
                         DefaultFormatBundle(),
                         Collect(keys=['img', 'gt_semantic_seg'])]

        valid_pipline = [Normalize(**cfg.img_norm_cfg),
                         Pad(size_divisor=32),
                         ImageToTensor(keys=['img']),
                         Collect(keys=['img'])]
        cfg.data.samples_per_gpu = bs
        cfg.data.workers_per_gpu = num_workers
        cfg.work_dir = path
        cfg.gpus = 1
        cfg.seed = None
        os.makedirs(path, exist_ok=True)

        self.clear_tfms()
        train, valid = self.split(show=show_dist)
        if train_tfms:
            train.set_tfms(train_tfms)
        train.add_tfm(_rgb2bgr)
        if valid_tfms:
            valid.set_tfms(valid_tfms)
        valid.add_tfm(_rgb2bgr)

        labelset = self.labelset()
        train_dataset = MMCustomDataset(train, labelset, train_pipline, test_mode=False)
        valid_dataset = MMCustomDataset(valid, labelset, valid_pipline, test_mode=True)
        return [train_dataset, valid_dataset]

    @classmethod
    def from_pascal_voc(cls, image_dir, xml_dir=None, check_ext=True, recurse=True, followlinks=False):
        '''
        image_dir:
        xml_dir: directory of xml, find xml in image_dir if xml_dir is None
        '''
        raise NotImplementedError

    def to_pascal_voc(self, out_path, keep_img_dir=0):
        raise NotImplementedError

    def to_mmdet_pkl(self, pkl_path):
        raise NotImplementedError

    @classmethod
    def from_disk(cls, image_dir, lbl_dir=None, check_ext=True, recurse=True, followlinks=False):
        return cls.from_labelme(image_dir, json_dir=lbl_dir, check_ext=check_ext,
                recurse=recurse, followlinks=followlinks)

    @classmethod
    def from_labelme(cls, image_dir, json_dir=None, check_ext=True, recurse=True, followlinks=False):
        if not json_dir:
            json_dir = image_dir

        image_dir = osp.normpath(image_dir)
        json_dir = osp.normpath(json_dir)

        img_json_match_result = img_label_path_match(image_dir, json_dir, ext='.json',
                check_ext=check_ext, recurse=recurse, followlinks=followlinks)
        label_info = list()
        label_set = set()
        for img_path, json_path in img_json_match_result.items():
            polygons = list()
            labels = list()
            if json_path:
                info = obj_from_json(json_path)
                shapes = info.get('shapes', [])
                for shape in shapes:
                    points = shape.get('points', [])
                    polygon = np.array(points).flatten().tolist()
                    if polygon:
                        polygons.append(','.join(str(math.ceil(x)) for x in polygon))
                        label = shape.get('label', 'object')
                        labels.append(label)
                        label_set.add(label)
            label_info.append({'localImagePath': img_path, 'polygons': polygons, 'labels': labels})

        class_list = sorted(list(label_set))
        label_info_dict = {'classList': class_list, 'labelSet': label_info}
        return cls.from_label_info(image_dir, label_info_dict)

    def to_labelme(self, out_path, keep_img_dir=0):
        '''
        keep_img_dir (int):
            eg: img_path: "a/b/c/img_name.jpg"
            0:  "out_path/img_name.json"
            1:  "out_path/c/img_name.json"
            2:  "out_path/b/c/img_name.json"
            3:  "out_path/a/b/c/img_name.json"
            ...
        '''
        self.to_disk(out_path, keep_img_dir=keep_img_dir)

    def show_sample(self, count=4, idx=None, ncols=2, figsize=(9,9), **kwargs):
        if idx is not None:
            if isinstance(idx, int):
                idx = [idx]
        else:
            count = min(count, len(self))
            idx = np.random.randint(0, len(self), size=count).tolist()
        if len(idx) > 0:
            img_list = list()
            for i in idx:
                img, label = self[i]
                img = draw_polygons_on_img_pro(img, label['polygons'], label['labels'], **kwargs)
                img_list.append(img)
            show_images(img_list, None, ncols, figsize)
        else:
            print('No image!')

    def limit_bboxes_min_size(self, min_size=32):
        ''' limit min of width and height to min_size
        '''
        raise NotImplementedError

    def do_tfm_imgaug(self, tfm, img, label):
        img, label = imgaug_img_polygon_tfm(img, label, tfm)
        return img, label

    def do_tfm_albumentations(self, tfm, img, label):
        raise NotImplementedError('albumentations not support polygon aug')

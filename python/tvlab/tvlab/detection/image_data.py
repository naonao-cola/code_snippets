'''
Copyright (C) 2023 TuringVision

List of image and label suitable for computer vision object detection task.
'''
import os
import math
import os.path as osp
import numpy as np
from PIL import Image
from ..utils import *
from ..category import ImageLabelList, get_image_files, get_files, save_image, get_image_shape
from ..ui import show_images, plot_stack_bar
from .bbox_label import BBoxLabel, BBoxLabelList

__all__ = ['ImageBBoxLabelList', 'imgaug_img_bbox_tfm',
           'limit_bbox_min_size', 'limit_bboxes_min_size', 'ybbox_filter']


def _rgb2bgr(x):
    return x[:, :, ::-1]


def imgaug_bbox_transform(aug, bboxes, shape, fraction=0.5):
    import imgaug
    bboxes_arr = np.array([box[:4] for box in bboxes], dtype=np.float32)
    bbs = imgaug.BoundingBoxesOnImage.from_xyxy_array(bboxes_arr, shape=shape)
    bbs_aug = aug.augment_bounding_boxes([bbs])[0]
    bbs_aug_on = bbs_aug.remove_out_of_image_fraction(fraction)
    mask = []
    for i, box in enumerate(bbs_aug.bounding_boxes):
        if box in bbs_aug_on.bounding_boxes:
            mask.append(i)
    bboxes = [bboxes[i] for i in mask]

    bbs_aug_on = bbs_aug_on.clip_out_of_image()
    ret_bboxes = bbs_aug_on.to_xyxy_array(dtype=np.float32)
    ret_bboxes = ret_bboxes.tolist()
    ret_bboxes = [rb + sb[4:] for sb, rb in zip(bboxes, ret_bboxes)]
    return ret_bboxes, mask


def imgaug_img_bbox_tfm(x, y, aug, fraction=0.5):
    '''
    Do img and bboxes transform for imgaug augmenter.

    In:
        x: image array
        y: {'labels': ['A', 'B'], 'bboxes': [[10, 20, 100, 200], [20, 40, 50, 80]]}
        aug: augmenter for imgaug
        fraction [0.0 ~ 1.0]: remove all bboxes with an out of image fraction of at least fraction
    Out:
        x, y
    '''
    import copy

    img_shape = x.shape
    y = copy.deepcopy(y)
    aug_det = aug.to_deterministic()
    x = aug_det.augment_images([x])[0]
    if y and y['bboxes'] and y['labels']:
        y['bboxes'], mask = imgaug_bbox_transform(aug_det,
                                                  y['bboxes'],
                                                  img_shape,
                                                  fraction)
        for k, v in y.items():
            if k != 'bboxes':
                y[k] = [v[i] for i in mask]
    return x, y


def limit_bbox_min_size(box, min_size):
    x = (box[0] + box[2]) / 2
    y = (box[1] + box[3]) / 2
    w = max(min_size, (box[2] - box[0]))
    h = max(min_size, (box[3] - box[1]))

    return (x-w/2, y-h/2, x+w/2, y+h/2)


def limit_bboxes_min_size(y, min_size=32):
    import warnings
    warnings.warn('`limit_bboxes_min_size` is deprecated and will be removed, use `ibll.y.limit_min_size()` instead.')
    if y and y['bboxes']:
        y['bboxes'] = [limit_bbox_min_size(box, min_size) for box in y['bboxes']]
    return y


def ybbox_filter(y, key):
    '''
    y (list): [{'labels': ['A', 'B'], 'bboxes': [[10, 20, 100, 200], [20, 40, 50, 80]]}, ...]
    key (callable):
        eg: def filter_func(l, box):
                if l == 'xxx':
                    return None
                box = limit_bbox_min_size(box, 32)
                return l, box
    '''
    import warnings
    warnings.warn('`ybbox_filter` is deprecated and will be removed, use `BBoxLabelList(y).filter() or ibll.y.filter()` instead.')

    new_y = list()
    for yi in y:
        labels, bboxes = [], []
        for l, box in zip(yi['labels'], yi['bboxes']):
            ret = key(l, box)
            if ret is None:
                continue
            if isinstance(ret, bool):
                if ret is False:
                    continue
            else:
                l, box = ret
            if l and box:
                labels.append(l)
                bboxes.append(box)
        new_y.append({'labels': labels, 'bboxes': bboxes})
    return new_y


def _y_to_d2_data_obj(y, class_list):
    from detectron2.structures import BoxMode

    if y is None or not y['labels']:
        return []

    bboxes = y.get('bboxes', None)
    polygons = y.get('polygons', None)
    if polygons is None:
        polygons = []
        for box in bboxes:
            l,t,r,b = box[:4]
            polygons.append([l,t,r,t,r,b,l,b])

    if bboxes is None:
        bboxes = [polygon_to_bbox(polygon) for polygon in polygons]

    objs = []
    for box, polygon, label in zip(bboxes, polygons, y['labels']):
        obj = {
            "bbox": box[:4],
            "bbox_mode": BoxMode.XYXY_ABS,
            "category_id": class_list.index(label),
            "segmentation": [polygon],
            "iscrowd": 0
        }
        objs.append(obj)
    return objs


def _get_detectron2_dataset_dicts(ibll, class_list):
    dataset_dicts = []
    for i, x in enumerate(ibll.x):
        height, width = get_image_shape(x, ibll.cache_x, ibll.cache_img)[0:2]
        record = {}
        record["file_name"] = x
        record["image_id"] = i
        record["height"] = height
        record["width"] = width

        record["annotations"] = _y_to_d2_data_obj(ibll.y[i], class_list)
        dataset_dicts.append(record)
    return dataset_dicts


def _detectron2_dataset_mapper(dataset_dict, ibll, with_gt=True, class_list=None):
    """
    Args:
        dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
    """
    import copy
    import torch
    from detectron2.data import detection_utils as utils

    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    idx = dataset_dict['image_id']

    img, y = ibll[idx]

    dataset_dict["image"] = torch.as_tensor(img.transpose(2, 0, 1).astype("float32"))
    if not with_gt:
        dataset_dict.pop("annotations")
        return dataset_dict

    image_shape = img.shape[:2]
    objs = _y_to_d2_data_obj(y, class_list)

    dataset_dict.pop("annotations")
    instances = utils.annotations_to_instances(
        objs, image_shape, mask_format="polygon",
    )
    dataset_dict["instances"] = utils.filter_empty_instances(instances)

    return dataset_dict


class MMCustomDataset:
    def __init__(self, ibll, labelset, pipeline, test_mode=False):
        from mmdet.datasets.pipelines import Compose
        self.CLASSES = labelset
        self.ibll = ibll
        self.test_mode = test_mode
        self.img_infos = self.load_annotations()
        if not self.test_mode:
            self._set_group_flag()
        self.pipeline = Compose(pipeline)

    def load_annotations(self):
        pkl_label_info = []
        labelset = self.CLASSES
        for image_path, yi in zip(self.ibll.x, self.ibll.y):
            if not yi:
                continue
            labels = yi['labels']
            bboxes = yi['bboxes']
            labels = [labelset.index(l) for l in labels]
            try:
                height, width = get_image_shape(image_path, self.ibll.cache_x, self.ibll.cache_img)[0:2]

                if not bboxes:
                    bboxes = [[0, 0, 0, 0]]
                    labels = [0]

                pkl_one = {
                    'filename': osp.basename(image_path),
                    'width': width,
                    'height': height,
                    'ann': {
                        'bboxes': np.array(bboxes, dtype=np.float32),
                        'labels': np.array(labels, dtype=np.int64),
                        'bboxes_ignore': np.zeros((0, 4), dtype=np.float32)
                    }
                }
                pkl_label_info.append(pkl_one)
            except Exception as e:
                print(e)
        return pkl_label_info

    def get_ann_info(self, idx):
        return self.img_infos[idx]['ann']

    def get_cat_ids(self, idx):
        return self.img_infos[idx]['ann']['labels'].astype(np.int64).tolist()

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.img_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    def _rand_another(self, idx):
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __len__(self):
        return len(self.img_infos)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_img(idx)
        while True:
            data = self.prepare_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def pre_pipeline(self, results):
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        results['flip'] = False
        results['flip_direction'] = 'horizontal'

    def prepare_img(self, idx):
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)

        img, label = self.ibll[idx]
        results['filename'] = self.ibll.x[idx]
        results['ori_filename'] = osp.basename(results['filename'])
        results['img'] = img
        results['img_shape'] = img.shape
        ori_shape = list(img.shape)
        ori_shape[0] = img_info['height']
        ori_shape[1] = img_info['width']
        results['ori_shape'] = tuple(ori_shape)
        h_scale = img.shape[0] / ori_shape[0]
        w_scale = img.shape[1] / ori_shape[1]
        results['scale_factor'] = np.array([w_scale, h_scale, w_scale, h_scale],
                                           dtype=np.float32)
        labels = label['labels']
        bboxes = label['bboxes']
        labelset = self.CLASSES
        labels = [labelset.index(l) for l in labels]
        if not bboxes:
            bboxes = [[0, 0, 0, 0]]
            labels = [0]
        results['gt_bboxes'] = np.array(bboxes, dtype=np.float32)
        results['gt_labels'] = np.array(labels, dtype=np.int64)
        results['bbox_fields'].append('gt_bboxes')
        results = self.pipeline(results)
        return results


def _label_idx(classes, label):
    return classes.index(label) + 1 if label in classes and label else 0


def _xy_to_tensor(img, gt, idx, img_path, labelset, cache_x, cache_img):
    import torch
    from torchvision.transforms.functional import to_tensor

    num_objs = len(gt['labels'])

    if num_objs > 0:
        boxes = torch.as_tensor([box[:4] for box in gt['bboxes']],
                                dtype=torch.float32)
        # there is only one class
        labels = torch.as_tensor(
            [_label_idx(labelset, l) for l in gt['labels']], dtype=torch.int64)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    else:
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
    target["ori_shape"] = get_image_shape(img_path, cache_x, cache_img)

    img = to_tensor(img.copy())
    return img, target


def _collate_fn(batch):
    import torch
    x, y = tuple(zip(*batch))
    return torch.stack(x), y


class ImageBBoxLabelList(ImageLabelList):
    '''
    label_list: (list) list of label
    label: {'labels': ['A', 'B'], 'bboxes': [[10, 20, 100, 200], [20, 40, 50, 80]]}
    '''
    def __init__(self, img_path_list, label_list=None):
        if label_list is None:
            y = BBoxLabelList([None for _ in img_path_list])
        else:
            y = BBoxLabelList(label_list)

        super().__init__(img_path_list, y)

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
            boxes_info_list = label_info['boxs']
            boxes = [[int(x) for x in box_str.split(",")] for box_str in boxes_info_list]
            labels = ['Other' if classes and l not in classes else l for l in labels]
            img_path_list.append(img_path)
            if "extern_infos" in label_info:
                label_list.append({'labels': labels, 'bboxes': boxes, "extern_infos": label_info["extern_infos"]})
            else:
                label_list.append({'labels': labels, 'bboxes': boxes})
        return cls(img_path_list, label_list)

    def to_label_info(self, class_list=None):
        '''convert ImageLabelList to label info dict
        '''
        label_info_dict = {}
        label_info_dict['classList'] = class_list if class_list is not None else self.labelset()
        labelset = []
        for img_path, label in zip(self.x, self.y):
            bboxes = label['bboxes']
            boxes_info_list = [','.join(str(math.ceil(x)) for x in box[:4]) for box in bboxes]
            label_info = {
                'imageName': osp.basename(img_path),
                'labels': label['labels'],
                'boxs': boxes_info_list
            }
            if "extern_infos" in label:
                label_info["extern_infos"] = label["extern_infos"]
            labelset.append(label_info)
        label_info_dict['labelSet'] = labelset
        return label_info_dict

    @classmethod
    def from_turbox_data(cls, turbox_data):
        '''create ImageBBoxLabelList from turbox format data
        '''
        classes = turbox_data.get('classList', None)
        img_path_list = []
        label_list = []
        for img_info in turbox_data['labelSet']:
            img_path = img_info['imagePath']
            img_path_list.append(img_path)
            boxes = []
            labels = []
            for shape in img_info['shapes']:
                if shape['shapeType'] == 'polygon':
                    polygons = shape['points']
                    xmin = np.min(polygons[::2])
                    xmax = np.max(polygons[::2])
                    ymin = np.min(polygons[1::2])
                    ymax = np.max(polygons[1::2])
                    boxes.append([xmin,ymin,xmax,ymax])
                    labels.append(shape['label'])
                elif shape['shapeType'] == 'rectangle':
                    boxes.append(shape['points'])
                    labels.append(shape['label'])
                elif shape['shapeType'] == 'circle':
                    c = shape['points']
                    r = np.linalg.norm(np.array(c[:2]) - np.array(c[2:]))
                    boxes.append([c[0]-r, c[1]-r, c[0]+r, c[1]+r])
                    labels.append(shape['label'])
            labels = ['Other' if classes and l not in classes else l for l in labels]
            label_list.append({'labels': labels, 'bboxes': boxes})
        return cls(img_path_list, label_list)

    def to_turbox_data(self, class_list=None):
        '''convert ImageBBoxLabelList to turbox data
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
            for label, box in zip(label['labels'], label['bboxes']):
                shape = {
                    'label': label,
                    'points': box,
                    'confidence': 1,
                    'shapeType': 'rectangle'
                }
                img_info['shapes'].append(shape)

            turbox_data['labelSet'].append(img_info)
        return turbox_data

    def databunch(self):
        raise NotImplementedError

    def get_to_tensor_tfm(self):
        from functools import partial
        labelset = self.labelset()
        cache_x = self.cache_x
        cache_img = self.cache_img
        return partial(_xy_to_tensor, labelset=labelset, cache_x=cache_x, cache_img=cache_img)

    def get_collate_fn(self):
        return _collate_fn

    def mmdet_data(self, cfg, train_tfms=None, valid_tfms=None, path='.',
                   bs=2, num_workers=1, show_dist=True):
        from mmdet.datasets.pipelines import (Normalize, Pad, Collect, ImageToTensor)
        from mmdet.datasets.pipelines.formating import DefaultFormatBundle
        train_pipline = [Normalize(**cfg.img_norm_cfg),
                         Pad(size_divisor=32),
                         DefaultFormatBundle(),
                         Collect(keys=['img', 'gt_bboxes', 'gt_labels'])]

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

    def detectron2_data(self, cfg, train_tfms=None, valid_tfms=None,
                        path='.', bs=2, num_workers=1,
                        dataset_dicts_func=None,
                        show_dist=True):
        ''' to detectron2 data_loader
        # Arguments
            cfg: detectron2 config
            train_tfms: transform list for train dataset
            valid_tfms: transform list for valid dataset
            path: work path for Detectron2
            bs: batch size
            dataset_dicts_func (callable): see _get_detectron2_dataset_dicts
            num_workers:
        # Returns
            cfg, (train_data_loader, valid_data_loader)
        '''
        from detectron2.data import (DatasetCatalog, MetadataCatalog,\
                                     build_detection_train_loader,\
                                     build_detection_test_loader)
        from functools import partial

        class_list = self.labelset()

        cfg.INPUT.FORMAT = 'BGR'
        cfg.INPUT.MASK_FORMAT = "polygon"
        cfg.DATASETS.TRAIN = ("train_dataset",)
        cfg.DATASETS.TEST = ("valid_dataset",)
        cfg.OUTPUT_DIR = path
        cfg.DATALOADER.NUM_WORKERS = num_workers
        cfg.SOLVER.IMS_PER_BATCH = bs
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_list)

        os.makedirs(path, exist_ok=True)

        self.clear_tfms()
        train, valid = self.split(show=show_dist)
        if train_tfms:
            train.set_tfms(train_tfms)
        train.add_tfm(_rgb2bgr)
        if valid_tfms:
            valid.set_tfms(valid_tfms)
        valid.add_tfm(_rgb2bgr)

        with_gt = False
        for y in self.y:
            if y != None and y['labels']:
                with_gt = True
                break

        if dataset_dicts_func is None:
            dataset_dicts_func = _get_detectron2_dataset_dicts

        _train_func = partial(dataset_dicts_func, ibll=train, class_list=class_list)
        DatasetCatalog.clear()
        DatasetCatalog.register("train_dataset", _train_func)

        _valid_func = partial(dataset_dicts_func, ibll=valid, class_list=class_list)
        DatasetCatalog.register("valid_dataset", _valid_func)
        if with_gt:
            metadata = MetadataCatalog.get("train_dataset")
            try:
                metadata.__delattr__('thing_classes')
            except AttributeError:
                pass
            metadata.set(thing_classes=class_list)
            metadata = MetadataCatalog.get("valid_dataset")
            try:
                metadata.__delattr__('thing_classes')
            except AttributeError:
                pass
            metadata.set(thing_classes=class_list)

        _valid_mapper = partial(_detectron2_dataset_mapper, ibll=valid,
                                with_gt=with_gt, class_list=class_list)
        data_loader_valid = build_detection_test_loader(cfg, 'valid_dataset',
                                                        mapper=_valid_mapper)
        data_loader_valid.batch_sampler.batch_size = bs

        data_loader_train = data_loader_valid
        if len(train) > 0:
            _train_mapper = partial(_detectron2_dataset_mapper, ibll=train,
                                    with_gt=with_gt, class_list=class_list)
            if with_gt:
                data_loader_train = build_detection_train_loader(cfg, mapper=_train_mapper)
            else:
                data_loader_train = build_detection_test_loader(cfg, 'train_dataset',
                                                                mapper=_train_mapper)

        return data_loader_train, data_loader_valid

    @classmethod
    def from_disk(cls, image_dir, lbl_dir=None, check_ext=True, recurse=True, followlinks=False):
        return cls.from_pascal_voc(image_dir, xml_dir=lbl_dir, check_ext=check_ext,
                recurse=recurse, followlinks=followlinks)

    @classmethod
    def from_pascal_voc(cls, image_dir, xml_dir=None, check_ext=True, recurse=True, followlinks=False):
        '''
        image_dir:
        xml_dir: directory of xml, find xml in image_dir if xml_dir is None
        '''
        from .pascal_voc_io import PascalVocReader

        if not xml_dir:
            xml_dir = image_dir

        image_dir = osp.normpath(image_dir)
        xml_dir = osp.normpath(xml_dir)

        img_xml_match_result = img_label_path_match(image_dir, xml_dir, ext='.xml',
                check_ext=check_ext, recurse=recurse, followlinks=followlinks)

        label_info = list()
        label_set = set()
        for img_path, xml_path in img_xml_match_result.items():
            bboxes = list()
            labels = list()
            extern_infos = list()
            if xml_path:
                reader = PascalVocReader(xml_path)
                shapes = reader.getShapes()
                extern_infos = reader.getExternInfos()
                for lb_info in shapes:
                    label = lb_info["name"]
                    points = lb_info["points"]
                    lt = points[0]
                    rb = points[2]
                    bboxes.append([lt[0], lt[1], rb[0], rb[1]])
                    labels.append(label)
                    label_set.add(label)

            boxes_info_list = [','.join(str(math.ceil(x)) for x in box[:4]) for box in bboxes]
            exter_info_len = sum([len(extern_info) for extern_info in extern_infos])
            if exter_info_len > 0:
                label_info.append({'localImagePath': img_path, 'boxs': boxes_info_list, 'labels': labels, "extern_infos": extern_infos})
            else:
                label_info.append({'localImagePath': img_path, 'boxs': boxes_info_list, 'labels': labels})

        class_list = sorted(list(label_set))
        label_info_dict = {'classList': class_list, 'labelSet': label_info}
        return cls.from_label_info(image_dir, label_info_dict)

    def to_disk(self, out_path, keep_img_dir=0):
        '''
        keep_img_dir (int):
            eg: img_path: "a/b/c/img_name.jpg"
            0:  "out_path/img_name.[xml/json]"
            1:  "out_path/c/img_name.[xml/json]"
            2:  "out_path/b/c/img_name.[xml/json]"
            3:  "out_path/a/b/c/img_name.[xml/json]"
            ...
        '''
        os.makedirs(out_path, exist_ok=True)
        for img_path, yi in zip(self.x, self.y):
            if not yi:
                continue
            img_dir = img_path.split(osp.sep)[-1-keep_img_dir:-1]
            out_label_dir = os.path.join(out_path, *img_dir)
            yi.to_disk(img_path, out_label_dir)

    def to_pascal_voc(self, out_path, keep_img_dir=0):
        '''
        keep_img_dir (int):
            eg: img_path: "a/b/c/img_name.jpg"
            0:  "out_path/img_name.xml"
            1:  "out_path/c/img_name.xml"
            2:  "out_path/b/c/img_name.xml"
            3:  "out_path/a/b/c/img_name.xml"
            ...
        '''
        self.to_disk(out_path, keep_img_dir=keep_img_dir)

    def to_mmdet_pkl(self, pkl_path):
        assert pkl_path is not None
        pkl_label_info = []
        labelset = self.labelset()
        for image_path, yi in zip(self.x, self.y):
            if not yi:
                continue
            labels = yi['labels']
            bboxes = yi['bboxes']
            extern_infos = yi.get("extern_infos", [dict()] * len(labels))
            labels = [labelset.index(l) for l in labels]
            try:
                with Image.open(image_path) as im:
                    if not bboxes:
                        bboxes = [[0, 0, 0, 0]]
                        labels = [0]
                    pkl_one = {
                        'filename': osp.basename(image_path),
                        'width': im.width,
                        'height': im.height,
                        'ann': {
                            'bboxes': np.array(bboxes, dtype=np.float32),
                            'labels': np.array(labels, dtype=np.int64),
                            'bboxes_ignore': np.zeros((0, 4), dtype=np.float32),
                            "extern_infos": extern_infos,
                        }
                    }
                    pkl_label_info.append(pkl_one)
            except Exception as e:
                print(e)

        obj_to_pkl(pkl_label_info, pkl_path)

    def find_idxs(self, key=None):
        ''' Return list of index for same image.
            eg:
            def key(img_path, label):
                return osp.basename(img_path)

            Returns:
                idxs: [1,3,5 ...]
        '''
        if key is None:
            key = lambda x,y: y and y['labels']
        find_idxs = [i for i, (x, y) in enumerate(zip(self.x, self.y))
                     if key(x, y)]
        return find_idxs

    def filter_similar_img(self, similar):
        return super().filter_similar_img(similar)

    def labelset(self):
        return self.y.labelset()

    def get_main_labels(self):
        def _get_max_cnt_label(info):
            labels = info['labels']
            if not labels:
                return 'backgroud'
            l = max([(i, labels.count(i)) for i in set(labels)], key=lambda x:x[1])[0]
            return l

        return [_get_max_cnt_label(info) for info in self.y]

    def get_main_labelset(self):
        return self.labelset()

    def show_split(self, train, valid, need_show=True):
        '''show split distribution
        '''
        labelset = self.labelset()
        labelset.append('backgroud')

        train_y = [l for info in train.y for l in info['labels']]
        valid_y = [l for info in valid.y for l in info['labels']]
        train_y += ['backgroud' for info in train.y if not info['labels']]
        valid_y += ['backgroud' for info in valid.y if not info['labels']]

        count_data = {'labelset': labelset,
                      'train': [train_y.count(l) for l in labelset],
                      'valid': [valid_y.count(l) for l in labelset]}

        return plot_stack_bar('Split distribution', count_data, 'labelset', ['valid', 'train'],
                              need_show=need_show)

    def split(self, valid_pct=None, seed=None, show=False):
        return super().split(valid_pct=valid_pct, per_label=False,
                             seed=seed, show=show)

    def kfold(self, fold=5, valid_pct=0.2, seed=None):
        ''' To split the dataset into K fold
        # Arguments:
            fold: (int) the number of K in K-fold
            valid_pct: percent of valid dataset.(0 ~ 1.0)
        # Returns
            list of ImageBBoxLabelList, A total of K
        '''
        return super().kfold(fold=fold, valid_pct=valid_pct,
                             per_label=False, seed=seed)

    def tile(self, tile_shape, overlap, out_dir, keep_img_dir=0, img_format='png',
             img_quality=95, iof_threshold=0.3, workers=8):
        ''' construct new ImageBBoxLabelList/ImagePolygonLabelList with
            tile operation for small objdet
        demo:
            - origin img shape: (9, 9)
            - tile shape: (3, 3)
            - overlap: (0, 0)
         _________            ___    ___    ___
        | A       |          | A |  |   |  |   |
        |         |          |   |  |   |  |   |
        |         |          |___|  |___|  |___|
        |    B    |   ---->   ___    ___    ___
        |         |          |   |  | B |  |   |
        |_________|          |   |  |   |  |   |
                             |___|  |___|  |___|

        tile_shape (tuple): (h, w)
        overlap (float or int or (h, w)):
        out_dir (str): output path for save crop image and xml
            -- outdir
                 |--- img
                 |     |--- 0_0  -->  "crop_left"_"crop_top"
                 |     |     |--- xxx1.jpg.jpg  --> ori_img_name + .crop_img_format
                 |     |     |--- xxx2.jpg.jpg
                 |     |----100_100
                 |     |     |--- xxx1.jpg.jpg
                 |     |     |--- xxx2.jpg.jpg
                 |--- lbl
                 |     |--- 0_0
                 |     |     |--- xxx1.jpg.xml  --> ori_img_name + .xml or .json
                 |     |     |--- xxx2.jpg.xml
                 |     |----100_100
                 |     |     |--- xxx1.jpg.xml
                 |     |     |--- xxx2.jpg.xml

        img_format (str): one of 'jpeg', 'png', 'bmp'
        img_quality (int): only for 'jpeg' format
        iof_threshold (float): iof threshold for keep ground truth in tile block

         _________            ___    ___
        | O       |          | O |  |   |
        |/|--     |          |/|-|  |-  |
        |/ \      |          |/_\|  |___|
        |         |   ---->    |      |
        |         |            |      V
        |_________|            |     drop label in this block
                               V
                             keep label is this block
        '''
        img_dir = osp.join(out_dir, 'img')
        lbl_dir = osp.join(out_dir, 'lbl')
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        tile_h, tile_w = tile_shape

        if isinstance(overlap, tuple):
            overlap_h, overlap_w = overlap
        else:
            overlap_h, overlap_w = overlap, overlap
        if isinstance(overlap_h, float):
            overlap_h = int(tile_h * overlap_h)
        if isinstance(overlap_w, float):
            overlap_w = int(tile_w * overlap_w)

        assert (overlap_h < tile_h and overlap_w < tile_w)

        tile_list = list()
        def _tile_one(idx):
            img_path = self.x[idx]
            img, gt = self[idx]
            ys = np.arange(0, img.shape[0] - tile_h - overlap_h, tile_h - overlap_h).tolist()
            ys += [img.shape[0]-tile_h]
            xs = np.arange(0, img.shape[1] - tile_w - overlap_w, tile_w - overlap_w).tolist()
            xs += [img.shape[1]-tile_w]
            fname = osp.basename(img_path)
            _keep_img_dir = img_path.split(osp.sep)[-1-keep_img_dir:-1]
            for y in ys:
                for x in xs:
                    out_img_path = osp.join(img_dir, *_keep_img_dir, f'{x}_{y}', fname)
                    out_img_path = out_img_path + '.' + img_format
                    img_block = np.array(Image.fromarray(img).crop((x,y,x+tile_w,y+tile_h)))
                    img_mode = 'RGB'
                    if len(img_block.shape) == 2:
                        img_mode = 'L'
                    save_image(img_block, out_img_path, img_mode,
                            img_quality=img_quality)
                    out_gt = gt.crop([x, y, x+tile_w, y+tile_h])
                    out_lbl_dir = osp.join(lbl_dir, *_keep_img_dir, f'{x}_{y}')
                    out_gt.to_disk(out_img_path, out_lbl_dir)
                    tile_list.append({'x': out_img_path, 'y': out_gt})

        thread_pool(_tile_one, list(range(len(self))), max(1, workers))
        tile_x_list = [item['x'] for item in tile_list]
        tile_y_list = [item['y'] for item in tile_list]
        return self.__class__(tile_x_list, tile_y_list)

    def label_from_tile(self, tile_ibll, iou_threshold=0.3):
        '''
         ___    ___    ___            _________
        | A |  |   |  |   |         | A       |
        |   |  |   |  |   |         |         |
        |___|  |___|  |___|         |         |
         ___    ___    ___   ---->  |    B    |
        |   |  | B |  |   |         |         |
        |   |  |   |  |   |         |_________|
        |___|  |___|  |___|

        tile_ibll: from ibll.tile(...)
        iou_threshold:
        '''
        new_ibll = self.copy()
        fname_list = [osp.basename(x) for x in self.x]
        new_ibll.y = self.y.__class__([None for _ in self.x])
        for x, y in zip(tile_ibll.x, tile_ibll.y):
            fname = osp.splitext(osp.basename(x))[0]
            idx = fname_list.index(fname)
            crop_x_dir = osp.dirname(osp.dirname(x))
            if fname_list.count(fname) > 1:
                match_idxs = [i for i, fn in enumerate(fname_list) if fn == fname]
                match_scores = [path_compare(crop_x_dir, osp.dirname(self.x[i]), True) for i in match_idxs]
                idx = match_idxs[np.argmax(match_scores)]
            x_off, y_off = [int(item) for item in osp.basename(osp.dirname(x)).split('_')][:2]
            new_y = y.offset(x_off, y_off)
            new_ibll.y[idx] += new_y
        new_ibll.y = new_ibll.y.nms(iou_threshold)
        return new_ibll

    def crop(self, out_dir, border=64, keep_img_dir=0,
             img_format='png', img_quality=95, workers=8):
        ''' construct new ImageBBoxLabelList/ImagePolygonLabelList with defect crop operation
        demo:
         _________            ___    ___
        | A       |          |   |  |   |
        | A       |          | A |  | B |
        |         |          | A |  |___|
        |    B    |   ---->  |___|
        |         |
        |_________|


        out_dir (str): output path for save crop image and xml
            -- outdir
                 |--- img
                 |     |--- 0_0  -->  "crop_left"_"crop_top"
                 |     |     |--- xxx1.jpg.jpg  --> ori_img_name + .crop_img_format
                 |     |     |--- xxx2.jpg.jpg
                 |     |----100_100
                 |     |     |--- xxx1.jpg.jpg
                 |     |     |--- xxx2.jpg.jpg
                 |--- xml
                 |     |--- 0_0
                 |     |     |--- xxx1.jpg.xml  --> ori_img_name + .xml or .json
                 |     |     |--- xxx2.jpg.xml
                 |     |----100_100
                 |     |     |--- xxx1.jpg.xml
                 |     |     |--- xxx2.jpg.xml

        border (int): width/height expand border size for defect crop.
        img_format (str): one of 'jpeg', 'png', 'bmp'
        img_quality (int): only for 'jpeg' format
        '''
        img_dir = osp.join(out_dir, 'img')
        lbl_dir = osp.join(out_dir, 'lbl')
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        crop_list = list()
        def _crop_one(idx):
            if not self.y[idx]['labels']:
                return

            img_path = self.x[idx]
            img, gt = self[idx]
            fname = osp.basename(img_path)
            _keep_img_dir = img_path.split(osp.sep)[-1-keep_img_dir:-1]
            for i, box in enumerate(gt.bboxes()):
                l,t,r,b = box[:4]
                xs = int(l - border)
                ys = int(t - border)
                xe = int(r + border)
                ye = int(b + border)
                out_img_path = osp.join(img_dir, *_keep_img_dir, f'{xs}_{ys}_{border}', fname)
                out_img_path = out_img_path + '.' + img_format
                img_block = np.array(Image.fromarray(img).crop((xs,ys,xe,ye)))
                img_mode = 'RGB'
                if len(img_block.shape) == 2:
                    img_mode = 'L'
                save_image(img_block, out_img_path, img_mode,
                        img_quality=img_quality)

                _gt = gt.__class__({k:[v[i]] for k, v in gt.items()})
                out_gt = _gt.crop([xs, ys, xe, ye])
                out_lbl_dir = osp.join(lbl_dir, *_keep_img_dir, f'{xs}_{ys}_{border}')
                out_gt.to_disk(out_img_path, out_lbl_dir)
                crop_list.append({'x': out_img_path, 'y': out_gt})

        thread_pool(_crop_one, list(range(len(self))), max(1, workers))
        crop_x_list = [item['x'] for item in crop_list]
        crop_y_list = [item['y'] for item in crop_list]
        return self.__class__(crop_x_list, crop_y_list)

    def label_from_crop(self, crop_ibll, iou_threshold=0.7):
        '''
         ___                         _________
        |   |                       | A       |
        | A |                       | A       |
        | A |                       |         |
        |___|   ____         ---->  |    BB   |
               |    |               |         |
               | BB |               |_________|
               |____|

        all label info from crop_ibll.
        in:
            crop_ibll: from ibll.crop(...)
            iou_threshold:
        '''
        return self.label_from_tile(crop_ibll, iou_threshold=iou_threshold)

    def label_update_from_error_crop(self, err_crop_ibll, iou_threshold=0.7):
        '''
           old                err_crop                new
         _________        ___           ___           _________
        | C       |      |   |         |   |         | A     E |
        | C       |      | A |         | E |         | A     E |
        |         |      | A |   ____  | E |         |         |
        |    BB   | -->  |___|  |    | |___|  ---->  |         |
        | D       |        ^    |    |   ^           | D       |
        |_________|        |    |____|   |           |_________|
          ^                |       ^     |
          |                |       |     overshot crop and relabel to real defect E
          |                |       |
          |                |       B is cropped and relabel to no defect
          D is not cropped |
                           C is cropped and relabel to A

        update label info from err_crop_ibll.
        '''
        new_ibll = self.copy()
        fname_list = [osp.basename(x) for x in self.x]
        err_crop_ibll.y = err_crop_ibll.y.filter_by_label(lambda l: ':' not in l)
        for x, y in zip(err_crop_ibll.x, err_crop_ibll.y):
            fname = osp.splitext(osp.basename(x))[0]
            idx = fname_list.index(fname)
            crop_x_dir = osp.dirname(osp.dirname(x))
            if fname_list.count(fname) > 1:
                match_idxs = [i for i, fn in enumerate(fname_list) if fn == fname]
                match_scores = [path_compare(crop_x_dir, osp.dirname(self.x[i]), True) for i in match_idxs]
                idx = match_idxs[np.argmax(match_scores)]
            x_off, y_off, border = [int(item) for item in
                    osp.basename(osp.dirname(x)).split('_')]
            new_y = y.offset(x_off, y_off)
            def _bbox_match(bbox):
                box_lt = (int(bbox[0]), int(bbox[1]))
                crop_lt = (int(x_off + border), int(y_off + border))
                return box_lt != crop_lt

            new_ibll.y[idx] = new_ibll.y[idx].filter_by_bbox(_bbox_match)
            new_ibll.y[idx] += new_y
        return new_ibll

    @classmethod
    def from_error_crop(cls, data_dir):
        '''
        data_dir is eval.export_error_crop out_dir
        '''
        _ibll = cls.from_disk(osp.join(data_dir, 'img'), osp.join(data_dir, 'lbl'))
        for yi in _ibll.y:
            yi['colors'] = ['gold' if ':' not in l else 'red' for l in yi['labels']]
        return _ibll

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
                img = draw_bboxes_on_img_pro(img, label['bboxes'], label['labels'], **kwargs)
                img_list.append(img)
            show_images(img_list, None, ncols, figsize)
        else:
            print('No image!')

    def limit_bboxes_min_size(self, min_size=32):
        ''' limit min of width and height to min_size
        '''
        import warnings
        warnings.warn('`limit_bboxes_min_size` is deprecated and will be removed, use `ibll.y.limit_min_size()` instead.')
        self.y = self.y.limit_min_size(size=min_size)

    def show_bbox_dist(self, labelset=None, x='size', y='ratio', xbins=20, ybins=20, need_show=True):
        ''' show distribution of bboxes
        labelset (list): only show class label in labelset
        x (str): one of ('ratio', 'size', 'area', 'w', 'h', 'x', 'y')
        y (str): one of ('ratio', 'size', 'area', 'w', 'h', 'x', 'y')
        '''
        return self.y.show_bbox_dist(labelset=labelset, x=x, y=y,
                                     xbins=xbins, ybins=ybins,
                                     need_show=need_show)

    def label_dist(self, reverse=True):
        labelset = self.labelset()
        labelset.append('backgroud')
        all_y = self.y.labels()
        all_y += ['backgroud' for info in self.y if not info['labels']]
        label_dist = {l: all_y.count(l) for l in labelset}
        label_dist = sorted(label_dist.items(), key=lambda data: data[1], reverse=reverse)
        label_dist = {l: num for l, num in label_dist}
        return label_dist

    def get_dist(self, sublabel_func=None, reverse=True):
        '''
        Note:
            1) suggest use just right arg_cnt.
                lambda x: get_image_res. (OK)
                lambda x,y: get_image_res. (Not Suggest)
        # Arguments
            sublabel_func:
                1) arg_cnt 1 (img_path):
                    eg1: lambda x: osp.basename(x)[5:7])
                    eg2: get_image_res
                    eg3: def sublabel_func(img_path):
                        return str
                2) arg_cnt 2 (img_path, y):
                    eg1: to get box area level.
                    def sublabel_func(img_path, label):
                        xxx = label['box'] # Note: 'box', not 'bboxes'
                        xxx = area_level_of_box(box)
                        return xxx

                    eg2: to get defect level.
                    sublabel_func = lambda x,y: y['level']

                3) arg_cnt 3 (img_path, y, i):
                    def sublabel_func(img_path, label, idx):
                        return str
        Return:
            in: sublabel_func, to get_image_res.
            out: dict()
                {
                    'labelset': ['P1U', 'N1U', 'OK'],
                    'sublabel_total': {'(1024, 768)': 1402,
                                        '(2456, 2058)': 397} # image resolution
                    '(1024, 768)': { 'P1U': 327, 'N1U': 428, 'OK': 647,},
                    '(2456, 2058)': {'P1U': 173, 'N1U': 71, 'OK': 153,}
                }
        '''
        label_dist_ordered = self.get_main_label_dist(reverse)
        label_dist = dict()
        if sublabel_func is None:
            label_dist = label_dist_ordered
            return label_dist

        labelset = list(label_dist_ordered.keys())
        label_dist['labelset'] = labelset

        # get_sublabel_list
        label_list = list()
        sublabel_list = list()
        for i, img_path in enumerate(tqdm(self.x)):
            yi = self.y[i]
            if not yi['labels']:
                if sublabel_func.__code__.co_argcount == 1:
                    sublabel = sublabel_func(img_path)
                else:
                    sublabel = 'None'
                sublabel_list.append(str(sublabel))
                label_list.append('backgroud')
            else:
                for j in range(len(yi.labels())):
                    label = yi['labels'][j]
                    if sublabel_func.__code__.co_argcount == 1:
                        sublabel = sublabel_func(img_path)
                    elif sublabel_func.__code__.co_argcount == 2:
                        sublabel = sublabel_func(img_path, yi[j])
                    else:
                        sublabel = sublabel_func(img_path, yi[j], i)
                    sublabel_list.append(str(sublabel))
                    label_list.append(label)

        sublabel_set = sorted(list(set(sublabel_list)))
        sub_dist_info = {sub: {label: 0 for label in labelset} for sub in sublabel_set}
        for sub_label, label in zip(sublabel_list, label_list):
            sub_dist_info[sub_label][label] += 1

        label_dist['sublabel_total'] = {sub: sublabel_list.count(sub) for sub in sublabel_set}
        label_dist.update(sub_dist_info)

        return label_dist

    def do_tfm_imgaug(self, tfm, img, label):
        img, label = imgaug_img_bbox_tfm(img, label, tfm)
        return img, label

    def do_tfm_albumentations(self, tfm, img, label):
        import albumentations as A
        if isinstance(tfm, A.Compose):
            tfm = tfm.transforms
        else:
            tfm = [tfm]
        label_fields = [l for l in label.keys() if l != 'bboxes']
        tfm = A.Compose(tfm, bbox_params=A.BboxParams('pascal_voc',
                                                      label_fields=label_fields))
        tfed = tfm(image=img, **label)
        img = tfed['image']
        for k, v in tfed.items():
            if k != 'image':
                label[k] = v
        label['bboxes'] = [list(box) for box in label['bboxes']]
        return img, label


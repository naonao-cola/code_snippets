'''
Copyright (C) 2023 TuringVision

List of image and label suitable for computer vision classfication task.
'''

import copy
import os
import mimetypes
import os.path as osp
import random
from random import shuffle, sample
import numpy as np
from abc import ABC, abstractmethod
from tqdm.auto import tqdm, trange
from ..utils import *
from ..ui import *
import uuid

__all__ = ['ImageLabelList', 'get_files', 'get_image_files',
           'open_image', 'save_image', 'get_image_res',
           'get_image_mode', 'get_image_shape',
           'ZipImageLabelList']

IMAGE_EXTENSIONS = set(k for k, v in mimetypes.types_map.items() if v.startswith('image/'))


def _get_files(p, f, extensions):
    if isinstance(extensions, str):
        extensions = [extensions]
    low_extensions = [e.lower() for e in extensions] if extensions is not None else None
    res = [osp.join(p, o) for o in f if not o.startswith('.')
           and (extensions is None or '.' + o.split(".")[-1].lower() in low_extensions)]
    return res


def get_files(path, extensions=None, recurse=False, followlinks=False):
    "Return list of files in `path` that have a suffix in `extensions`; optionally `recurse`."
    if recurse:
        res = []
        for p, _, f in os.walk(path, followlinks=followlinks):
            res += _get_files(p, f, extensions)
        return res
    else:
        f = [o for o in os.listdir(path)]
        return _get_files(path, f, extensions)


def get_image_files(c, check_ext=True, recurse=False, followlinks=False):
    "Return list of files in `c` that are images. `check_ext` will filter to `image_extensions`."
    return get_files(c, extensions=(IMAGE_EXTENSIONS if check_ext else None),
                     recurse=recurse, followlinks=followlinks)


def get_image_res(fn):
    "Return (W, H) of image in file `fn`."
    from PIL import Image
    x = Image.open(fn)
    return (x.width, x.height)


def get_image_mode(fn):
    ''' RGB or L '''
    from PIL import Image
    x = Image.open(fn)
    return x.mode


def get_image_shape(fn, cache_x=None, cache_img=None):
    ''' get image shape. (H, W, C) '''
    if cache_x and cache_img and fn in cache_x:
        x = cache_img[cache_x.index(fn)]
        img_shape = x.shape
    else:
        from PIL import Image
        x = Image.open(fn)
        if x.mode == 'L':
            img_shape = (x.height, x.width)
        elif x.mode == 'RGB':
            img_shape = (x.height, x.width, 3)
        else:
            img_shape = (x.height, x.width, x.mode)
    return img_shape


def open_image(fn, convert_mode='RGB'):
    '''
    In:
        fn: image path
        convert_mode: one of [None, 'L', ‘RGB'], use origin mode if convert_mode is None
    Out: np.ndarray
    '''
    from PIL import Image
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # EXIF warning from TiffPlugin
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)
        x = Image.open(fn)
        if convert_mode and x.mode != convert_mode:
            x = x.convert(convert_mode)
    return np.asarray(x)


def save_image(img_arr, fn, img_mode='RGB', img_quality=95):
    '''
    Saves this image under the given filename.
    img_arr (np.ndarray): (h, w, 3) for RGB or (h, w) for 'L'
    fn: filename
    img_mode: 'L' or 'RGB'
    '''
    from PIL import Image
    os.makedirs(osp.dirname(fn), exist_ok=True)
    Image.fromarray(img_arr, img_mode).save(fn, quality=img_quality)


def _label_idx(classes, label):
    return classes.index(label) if label in classes and label else -1


def _xy_to_tensor(img, gt, idx, img_path, labelset):
    import torch
    from torchvision.transforms.functional import to_tensor

    if isinstance(img, np.ndarray):
        img = to_tensor(img)
    if isinstance(gt, (list, tuple)):
        gt = torch.Tensor([_label_idx(labelset[i], gti) for i, gti in enumerate(gt)])
    elif isinstance(gt, str):
        gt = _label_idx(labelset, gt)
    elif gt is None:
        gt = -1
    return img, gt


def _fastai_format_convert(x):
    from fastai.vision import Image as FImage
    import torch

    if isinstance(x, (list, tuple)):
        xall = list()
        for xi in x:
            if isinstance(xi, np.ndarray):
                if xi.ndim == 3:
                    xi = np.transpose(xi, (2, 0, 1))
                xi = torch.from_numpy(xi.astype(np.float32, copy=False))
                if xi.dim() >= 2:
                    xi.div_(255)
                    xi = FImage(xi)
            xall.append(xi)
        return xall
    else:
        if isinstance(x, np.ndarray):
            x = np.transpose(x, (2, 0, 1))
            x = torch.from_numpy(x.astype(np.float32, copy=False))
            x.div_(255)
            x = FImage(x)
        return x


class ImageLabelList(ABC):
    ''' List of image data and annotated information, simplify dataset preprocessing.

    # cache image
    ill = ImageLabelList.from_folder(x).filter(x).label_from_func(x).filter(x).add_tfm(x)
    train, valid = ill.split(x)
    ill.export(pkl, h5)

    # train/inference
    ill = ImageLabelList.load(pkl, h5).clear_tfms().add_tfm(x)
    train, valid = ill.split(x) # split from cached idx
    '''

    def __init__(self, img_path_list, label_list=None):
        '''
        # Arguments
            img_path_list: (list) list of image path
            label_list: (list) list of label
        '''

        self.x = img_path_list
        if label_list is None:
            y = [None] * len(self.x)
        else:
            assert len(img_path_list) == len(label_list),\
                "img_path_list length(%d) do not match label_list length(%d)" % (len(img_path_list), len(label_list))
            y = label_list
        self.y = y
        self._tfms = []
        self._train_idx = None
        self._valid_idx = None
        # cache_f is not been used anymore , but used to adapt for former projects. Please do not delete it!
        self.cache_f = None
        self.cache_x = None
        self.cache_img = None
        self.img_mode = 'RGB'

    def set_img_mode(self, mode='RGB'):
        '''
        mode: 'RGB' or 'L' or None , use origin mode if it's None
        '''
        self.img_mode = mode

    def cache_images(self, workers=8):
        '''
        Cache read data into memory.
        '''
        assert self.x, "Image list is not none and not be empty"
        self.cache_x = self.x.copy()

        if self.cache_img is None:
            self.cache_img = []

        image_dict = dict()

        def read_image_data(img_path):
            image_dict[img_path] = open_image(img_path, self.img_mode)

        thread_pool(read_image_data, list(self.x), max(1, workers))

        for img_path in self.cache_x:
            self.cache_img.append(image_dict[img_path])

        return self

    def copy(self):
        ''' Return one copy of ImageLabelList
        '''
        return self.from_ill(self)

    @classmethod
    def from_ill(cls, ill):
        new_ill = cls(list(ill.x).copy(), copy.deepcopy(list(ill.y)))
        new_ill._tfms = ill._tfms.copy()
        # cache_f is not been used anymore , but used to adapt for former projects. Please do not delete it!
        new_ill.cache_f = getattr(ill, 'cache_f', None)
        new_ill.cache_x = ill.cache_x
        new_ill.cache_img = ill.cache_img
        if hasattr(ill, 'img_mode'):
            new_ill.img_mode = ill.img_mode
        else:
            new_ill.img_mode = 'RGB'
        return new_ill

    @classmethod
    def from_memory(cls, imgs, label_list=None):
        '''
        create ImageLabelList from image memory.
            imgs: images in memory.
            label_list: labels
        '''
        pre = "memory_mode_"
        img_path_list = [pre + str(uuid.uuid4()) for _ in range(len(imgs))]
        ill = cls(img_path_list, label_list)
        ill.cache_x = img_path_list.copy()
        ill.cache_img = imgs.copy()
        return ill

    @classmethod
    def from_folder(cls, path, check_ext=True, recurse=True, followlinks=False):
        '''create ImageLabelList from image folder.
        '''
        img_path_list = get_image_files(path, check_ext=check_ext,
                                        recurse=recurse, followlinks=followlinks)
        return cls(img_path_list)

    @classmethod
    def from_label_info(cls, image_dir, label_info_dict):
        '''create ImageLabelList from label info dict
        '''
        classes = label_info_dict.get('classList', None)
        img_path_list = []
        label_list = []
        for label_info in label_info_dict['labelSet']:
            img_path = osp.join(image_dir, label_info['imageName'])
            label = label_info['labels'][0]
            if classes and label not in classes:
                label = 'Other'
            img_path_list.append(img_path)
            label_list.append(label)
        return cls(img_path_list, label_list)

    def to_label_info(self, class_list=None):
        '''convert ImageLabelList to label info dict
        '''
        label_info_dict = {}
        label_info_dict['classList'] = class_list if class_list is not None else self.labelset()
        labelset = []
        for img_path, label in zip(self.x, self.y):
            label_info = {
                'imageName': osp.basename(img_path),
                'labels': [label],
                'boxs': ["0.0,0.0,1.0,1.0"]
            }
            labelset.append(label_info)
        label_info_dict['labelSet'] = labelset
        return label_info_dict

    @classmethod
    def from_turbox_data(cls, turbox_data):
        '''create ImageLabelList from turbox format data
        '''
        classes = turbox_data.get('classList', None)
        img_path_list = []
        label_list = []
        for img_info in turbox_data['labelSet']:
            img_path = img_info['imagePath']
            img_path_list.append(img_path)
            label_list.append(img_info['shapes'][0]['label'])
        label_list = ['Other' if classes and l not in classes else l for l in label_list]
        return cls(img_path_list, label_list)

    def to_turbox_data(self, class_list=None):
        '''convert ImageLabelList to turbox data
        '''
        turbox_data = {}
        turbox_data['classList'] = class_list if class_list is not None else self.labelset()
        turbox_data['labelSet'] = []
        for img_path, label in zip(self.x, self.y):
            img_info = {
                'imageName': osp.basename(img_path),
                'imagePath': img_path,
                'shapes': [{'label': label}]
            }
            turbox_data['labelSet'].append(img_info)
        return turbox_data

    def databunch(self, train_tfms=None, valid_tfms=None, path='.',
                  bs=64, num_workers=1, show_dist=True,
                  label_cls=None, batch_sampler=None, **kwargs):
        '''to fastai databunch

        # Arguments
            train_tfms: transform list for train dataset
            valid_tfms: transform list for valid dataset
            path: work path for fastai
            bs: batch size
            num_workers:
            batch_sampler: see torch.utils.data.BatchSampler (training only).
        # Returns
            fastai databunch
        '''
        from fastai.vision import ImageList, ItemLists, CategoryList
        from .fastai_custom_image_list import CImageList

        self.clear_tfms()
        train, valid = self.split(show=show_dist)

        if train_tfms:
            train.set_tfms(train_tfms)
        train.add_tfm(_fastai_format_convert)
        if valid_tfms:
            valid.set_tfms(valid_tfms)
        valid.add_tfm(_fastai_format_convert)

        if len(train) == 0:
            train = valid
        else:
            train = train.shuffle()

        tcil = CImageList(train)
        vcil = CImageList(valid)

        if label_cls is None:
            label_cls = CategoryList

        ll = ItemLists(path, tcil, vcil).label_from_lists(train.y, valid.y,
                                                          label_cls=label_cls,
                                                          classes=self.labelset())
        data = ll.databunch(path, bs=bs, num_workers=num_workers, **kwargs)
        if batch_sampler:
            from torch.utils.data import DataLoader
            from fastai.basic_data import DeviceDataLoader

            get_y_func = None
            if hasattr(self, 'main_label_idx'):
                get_y_func = lambda y: y[self.main_label_idx]
            dl = DataLoader(data.train_dl.dataset,
                            batch_sampler=batch_sampler(data.train_dl.dl.sampler,
                                                        get_y_func=get_y_func),
                            num_workers=num_workers)
            data.train_dl = DeviceDataLoader(dl, data.train_dl.device,
                                             data.train_dl.tfms,
                                             data.train_dl.collate_fn)
        return data

    def get_to_tensor_tfm(self):
        from functools import partial
        labelset = self.labelset()
        return partial(_xy_to_tensor, labelset=labelset)

    def get_collate_fn(self):
        return None

    def dataloader(self, train_tfms=None, valid_tfms=None,
                   bs=8, num_workers=8, **kwargs):
        '''to torch DataLoader

        # Arguments
            train_tfms: transform list for train dataset
            valid_tfms: transform list for valid dataset
            bs: batch size
            num_workers:
        # Returns
            train_dataloader, valid_dataloader
        '''
        from torch.utils.data import DataLoader

        xy_to_tensor_p = self.get_to_tensor_tfm()
        self.clear_tfms()
        train, valid = self.split(show=False)

        if train_tfms:
            train.set_tfms(train_tfms)
        train.add_tfm(xy_to_tensor_p)
        if valid_tfms:
            valid.set_tfms(valid_tfms)
        valid.add_tfm(xy_to_tensor_p)

        collate_fn = self.get_collate_fn()
        if 'collate_fn' in kwargs:
            collate_fn = kwargs['collate_fn']
            kwargs.pop('collate_fn')

        valid_loader = DataLoader(dataset=valid,
                                  batch_size=bs,
                                  num_workers=num_workers,
                                  shuffle=False,
                                  collate_fn=collate_fn,
                                  **kwargs)
        if len(train) == 0:
            train_loader = valid_loader
        else:
            train_loader = DataLoader(dataset=train,
                                      batch_size=bs,
                                      num_workers=num_workers,
                                      drop_last=True,
                                      shuffle=True,
                                      collate_fn=collate_fn,
                                      **kwargs)

        return train_loader, valid_loader

    def shuffle(self, seed=None):
        '''shuffle image list
        '''
        if self._train_idx is not None:
            self._train_idx = np.random.permutation(self._train_idx)
        elif self.x:
            if seed is not None:
                np.random.seed(seed)
            rand_idx = np.random.permutation(list(range(len(self.x))))
            y_cls = self.y.__class__
            self.x, self.y = zip(*[(self.x[i], self.y[i]) for i in rand_idx])
            self.y = y_cls(self.y)
        return self

    def label_from_func(self, func):
        '''get label info from func
        eg:
            def label_func(img_path):
                return label
        '''
        y_cls = self.y.__class__
        label_list = [func(fn) for fn in self.x]
        self.y = y_cls(label_list)
        return self

    def label_from_folder(self):
        '''give a label to each filename depending on its folder.
        '''
        return self.label_from_func(func=lambda o: (o.split(osp.sep))[-2])

    def sorted(self, key=None, reverse=False):
        ''' Return a new list containing all items from the iterable in ascending order.
            A custom key function can be supplied to customize the sort order, and the
            reverse flag can be set to request the result in descending order.

            eg:
            def key(img_path, label):
                return osp.basename(img_path)
        '''
        def _sort_key(item):
            if key:
                return key(item[0], item[1])
            return osp.basename(item[0])
        new_x, new_y = zip(*sorted(zip(self.x, self.y), key=_sort_key, reverse=reverse))
        y_cls = self.y.__class__
        ill = self.copy()
        ill.x = new_x
        ill.y = y_cls(new_y)
        return ill

    def find_idxs(self, key=None):
        ''' Return list of index for same image.
            eg:
            def key(img_path, label):
                return osp.basename(img_path)

            Returns:
                idxs: [1,3,5 ...]
        '''
        if key is None:
            key = lambda x, y: y is not None
        find_idxs = [i for i, (x, y) in enumerate(zip(self.x, self.y))
                     if key(x, y)]
        return find_idxs

    def filter(self, func):
        '''filter unwanted items, func return False for unwanted item
        eg:
            def filter_func(img_path, label):
                return True of False
        '''
        def warp_func(item):
            return func(item[0], item[1])
        try:
            img_path_list, label_list = zip(*filter(warp_func, zip(self.x, self.y)))
        except ValueError:
            print('The number of items after filter is 0!')
            img_path_list = []
            label_list = []
        y_cls = self.y.__class__
        ill = self.copy()
        ill.x = img_path_list
        ill.y = y_cls(label_list)
        return ill

    def filter_invalid_img(self, check_img_data=False, percent_cb=None, workers=8,
                           ret_invalids=False):
        '''filter invalid image
        in: check_img_data: try to decode jpeg/png.
            percent_cb: callback for progress
            ret_invalids: to return invalid_img_list
        Out:
            if ret_invalids is False, return filtered_ill
            if ret_invalids is True, return (filtered_ill, invalid_list)
        '''
        from PIL import Image
        from threading import RLock

        status = {'total': len(self), 'index': 0, 'percent': -1}
        filter_info = dict()
        invalid_list = list()

        rlock = RLock()

        def check_img_func(img_path):
            with rlock:
                if percent_cb:
                    percent = int((status['index'] + 1) * 100 / status['total'])
                    status['index'] += 1
                    if percent != status['percent']:
                        status['percent'] = percent
                        percent_cb(percent)
            filter_info[img_path] = True
            try:
                with Image.open(img_path) as im:
                    if check_img_data:
                        _ = im.getdata()
            except Exception as e:
                filter_info[img_path] = False
                if ret_invalids:
                    invalid_list.append(img_path)

        thread_pool(check_img_func, list(self.x), max(1, workers))
        new_ill = self.filter(lambda x, y: filter_info[x] if x in filter_info else False)
        if ret_invalids:
            return new_ill, invalid_list
        else:
            return new_ill

    def filter_similar_img(self, similar, prio_labels=None):
        '''
        similar: ImageSimilarPro
        prio_labels: (list of labels) The image's label in the priority list will be used first
        '''
        group_idxs, _ = similar.group()
        remove_idxs = []
        for idxs in tqdm(group_idxs):
            imgs_path = [similar.img_path_list[i] for i in idxs]
            valid_idxs = []
            for im_path in imgs_path:
                try:
                    i = self.x.index(im_path)
                    valid_idxs.append(i)
                except ValueError:
                    pass
            idxs = valid_idxs
            main_y = self.get_main_labels()
            labels = [main_y()[j] for j in idxs]
            label_set = list(set(labels))
            if len(label_set) == 1:
                if len(idxs) > 1:
                    remove_idxs.extend(idxs[1:])
            else:
                pick_label = None
                if prio_labels:
                    for label in prio_labels:
                        if label in label_set:
                            pick_label = label
                            break
                if pick_label is None:
                    label_cnts = [labels.count(label) for label in label_set]
                    pick_label = label_set[np.argmax(label_cnts)]
                pick_idx = labels.index(pick_label)
                remove_idxs.extend([i for i in idxs if i != pick_idx])
        pick_idxs = set(list(range(len(self)))) - set(remove_idxs)
        new_x = [self.x[i] for i in pick_idxs]
        new_y = [self.y[i] for i in pick_idxs]
        return self.__class__(new_x, new_y)

    def resample(self, label_weights=None, max_count=None, seed=None):
        '''resample with label weights
        # Arguments
            label_weights: resample weight for each label
            eg: {'default': 1.0,
                 'aaa': 3.0,
                 'bbb': -1, # -1 is auto resample to max_count
                 'backgroud': 0.5}
            max_count: max count for each label
        '''
        if seed is not None:
            random.seed(seed)

        if self._train_idx is not None:
            main_y = self.get_main_labels()
            y_list = [main_y[i] for i in self._train_idx]
        else:
            y_list = self.get_main_labels()

        # 1. get origin count for each class
        labelset = self.get_main_labelset()
        if 'backgroud' in y_list and 'backgroud' not in labelset:
            labelset += ['backgroud']
        origin_count = [y_list.count(l) for l in labelset]
        max_count = max_count if max_count else max(origin_count)

        # 2. apply label weights for each label
        label_weights = label_weights or dict()
        new_count = []
        default_weight = label_weights['default'] if 'default' in label_weights else 1.0
        for label, count in zip(labelset, origin_count):
            weight = default_weight
            if label in label_weights:
                weight = label_weights[label]
            if weight < 0:
                weight = max_count / count
            elif weight == 1.0 and count > max_count:
                weight = max_count / count
            new_count.append(int(count * weight))

        # 3. resample with new count
        new_idx = []
        for label, count in zip(labelset, new_count):
            label_idx = [i for i, l in enumerate(y_list) if l == label]
            ori_count = len(label_idx)
            if ori_count > 0:
                while count >= ori_count:
                    new_idx = new_idx + label_idx
                    count -= ori_count
            if count > 0:
                new_idx = new_idx + sample(label_idx, k=count)

        if self._train_idx is None:
            if new_idx:
                img_path_list = [self.x[i] for i in new_idx]
                label_list = [copy.deepcopy(self.y[i]) for i in new_idx]
            else:
                img_path_list, label_list = [], []
            y_cls = self.y.__class__
            ill = self.copy()
            ill.x = img_path_list
            ill.y = y_cls(label_list)
        else:
            new_idx = [self._train_idx[i] for i in new_idx]
            self._train_idx = new_idx
            ill = self
        return ill

    def clear_tfms(self):
        '''clear all transform
        '''
        self._tfms = []
        return self

    def set_tfms(self, tfms):
        '''set all transform

        tfms (list): list of transform
        supported transform:
            def tfm_func1(img):
                return new_img

            def tfm_func2(img, label):
                return new_img, new_label

            def tfm_func3(img, label, idx, img_path):
                return new_img, new_label

            # import imgaug.augmenters as iaa
            imgaug_aug = iaa.Sequential([
                iaa.CropToFixedSize(224, 224, position='normal'),
                iaa.GaussianBlur(),
                ])

            # import albumentations as A
            a_aug = A.Compose([
                A.RandomCrop(768, 384),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                ])

            eg: xxx.set_tfms([a_aug, imgaug_aug, ..., tfm_funcx, ...]
        '''
        self._tfms = tfms.copy()

    def get_tfms(self):
        return self._tfms

    def add_tfm(self, func):
        '''add one transform
        eg:
            def tfm_func(img, label):
                return new_img, new_label

            or:

            def tfm_func(img, label, idx, img_path):
                return new_img, new_label
        '''
        self._tfms.append(func)
        return self

    def do_tfm_imgaug(self, tfm, img, label):
        img = tfm.augment_image(img)
        return img, label

    def do_tfm_albumentations(self, tfm, img, label):
        img = tfm(image=img)['image']
        return img, label

    def do_tfms(self, tfms, img, label, idx, img_path):
        '''do all transform for one image
        '''
        label = copy.deepcopy(label)

        for tfm in tfms:
            if hasattr(tfm, '__module__'):
                if tfm.__module__.startswith('imgaug.'):
                    img, label = self.do_tfm_imgaug(tfm, img, label)
                    continue
                elif tfm.__module__.startswith('albumentations.'):
                    img, label = self.do_tfm_albumentations(tfm, img, label)
                    continue
            if hasattr(tfm, '__code__'):
                if tfm.__code__.co_argcount == 1:
                    img = tfm(img)
                    continue
                elif tfm.__code__.co_argcount == 2:
                    result = tfm(img, label)
                    if isinstance(result, tuple):
                        img, label = result
                    else:
                        img = result
                    continue
            result = tfm(img, label, idx, img_path)
            if isinstance(result, tuple):
                img, label = result
            else:
                img = result
        return img, label

    def _split_by_idx(self, idx):
        '''split the dataset to get a subset
        # Arguments
            idx: list of pick index

        # Returns
            new ImageLabelList
        '''
        if len(idx) > 0:
            img_path_list, label_list = zip(*[(self.x[i], self.y[i]) for i in idx])
        else:
            img_path_list, label_list = [], []
        y_cls = self.y.__class__
        ill = self.copy()
        ill.x = img_path_list
        ill.y = y_cls(label_list)
        return ill

    def split_by_idxs(self, train_idx, valid_idx):
        '''split the dataset to two subset(train and valid)
        # Arguments
            train_idx: list of pick index for train dataset
            valid_idx: list of pick index for valid dataset

        # Returns
            ImageLabelList for train, ImageLabelList for valid
        '''
        self._train_idx = train_idx
        self._valid_idx = valid_idx
        train_list = self._split_by_idx(train_idx)
        valid_list = self._split_by_idx(valid_idx)
        return train_list, valid_list

    def labelset(self):
        '''get label set
        '''
        labelset = list(set(self.y))
        if None in labelset and len(labelset) != 1:
            labelset.remove(None)
        return sorted(labelset)

    def get_main_labels(self):
        return self.y

    def get_main_labelset(self):
        return self.labelset()

    def show_split(self, train, valid, need_show=True):
        '''show split distribution
        '''
        labelset = self.get_main_labelset()
        train_y = train.get_main_labels()
        valid_y = valid.get_main_labels()
        if None in train_y or None in valid_y:
            labelset.append(None)
        count_data = {'labelset': [str(l) for l in labelset],
                      'train': [train_y.count(l) for l in labelset],
                      'valid': [valid_y.count(l) for l in labelset]}

        return plot_stack_bar('Split distribution', count_data, 'labelset', ['train', 'valid'],
                              need_show=need_show)

    def get_split(self):
        return self._train_idx, self._valid_idx

    def clear_split(self):
        '''clear all split
        '''
        self._train_idx = None
        self._valid_idx = None
        return self

    def split(self, valid_pct=None, per_label=True, seed=None, show=False):
        '''split the dataset to two subset(train and valid)
        # Arguments
            valid_pct: percent of valid dataset.(0 ~ 1.0)
            If valid_pct is None use split information from records.
            per_label: split up by the number of each category

        # Returns
            ImageLabelList for train, ImageLabelList for valid
        '''
        assert valid_pct is None or (0.0 <= valid_pct <= 1.0)

        if valid_pct is None:
            if self._train_idx is not None and self._valid_idx is not None:
                train_idx = self._train_idx
                valid_idx = self._valid_idx
            else:
                print('There is no split record. Please provide valid_pct!')
                return None, None
        elif valid_pct == 1.0:
            train_idx = []
            valid_idx = list(range(len(self.x)))
        else:
            if seed is not None:
                np.random.seed(seed)
            if per_label:
                labelset = self.get_main_labelset()
                train_idx = []
                valid_idx = []
                main_y = self.get_main_labels()
                for label in labelset:
                    label_idx = np.random.permutation([i for i, l in enumerate(main_y)
                                                       if l == label])
                    label_idx = label_idx.tolist()
                    cut = int(valid_pct * len(label_idx))
                    train_idx += label_idx[cut:]
                    valid_idx += label_idx[:cut]
            else:
                rand_idx = np.random.permutation(list(range(len(self.x))))
                cut = int(valid_pct * len(self.x))
                train_idx = rand_idx[cut:]
                valid_idx = rand_idx[:cut]

        train, valid = self.split_by_idxs(train_idx, valid_idx)
        if show:
            self.show_split(train, valid)
        return train, valid

    def kfold(self, fold=5, valid_pct=0.2, per_label=True, seed=None):
        ''' To split the dataset into K fold
        # Arguments:
            fold: (int) the number of K in K-fold
            valid_pct: percent of valid dataset.(0 ~ 1.0)
            per_label: split up by the number of each category
        # Returns
            list of ImageLabelList, A total of K
        '''
        slice_num = 10
        valid_slice_num = int(slice_num * valid_pct)
        if seed is not None:
            np.random.seed(seed)
        if per_label:
            labelset = self.get_main_labelset()
            k_fold_train_idx = [[] for i in range(fold)]
            k_fold_valid_idx = [[] for i in range(fold)]
            main_y = self.get_main_labels()
            for label in labelset:
                rand_idx = np.random.permutation([i for i, l in enumerate(main_y) if l == label])
                kf_code_train_idx, kf_code_valid_idx = kfold_split(rand_idx, fold,
                                                                   slice_num, valid_slice_num)
                for i in range(fold):
                    k_fold_train_idx[i] += kf_code_train_idx[i]
                    k_fold_valid_idx[i] += kf_code_valid_idx[i]
        else:
            rand_idx = np.random.permutation(list(range(len(self.x))))
            k_fold_train_idx, k_fold_valid_idx = kfold_split(rand_idx, fold,
                                                             slice_num, valid_slice_num)

        k_fold_ill = list()
        for train_idx, valid_idx in zip(k_fold_train_idx, k_fold_valid_idx):
            new_ill = self.copy()
            new_ill.split_by_idxs(train_idx, valid_idx)
            k_fold_ill.append(new_ill)
        return k_fold_ill

    @classmethod
    def merge(cls, train, valid):
        ''' merge two ImageLabelList
        # Arguments:
            train: train dataset
            valid: valid dataset
        '''
        merge_x = list(train.x) + list(valid.x)
        merge_y = list(copy.deepcopy(train.y)) + list(copy.deepcopy(valid.y))
        ill = cls(merge_x, merge_y)
        train_cnt = len(train)
        valid_cnt = len(valid)
        train_idx = list(range(train_cnt))
        valid_idx = list(range(train_cnt, train_cnt + valid_cnt))
        _, _ = ill.split_by_idxs(train_idx, valid_idx)
        return ill

    def export(self, pkl_path, img_cache_path=None):
        '''export this to pickle db.

        # Arguments
            pkl_path: pickle file name
            img_cache_path: h5py file name for image cache.
        '''
        import h5py
        if img_cache_path:
            os.makedirs(osp.dirname(img_cache_path), exist_ok=True)
            dst_img_path_list = []
            dst_label_list = []
            with h5py.File(img_cache_path, 'w') as f:
                dest_img = None
                for i in tqdm(range(len(self))):
                    try:
                        img, label = self[i]
                    except Exception as e:
                        continue
                    img_path = self.x[i]
                    if dest_img is None:
                        dest_img = f.create_dataset('img',
                                                    (len(self), *img.shape),
                                                    maxshape=(None, *img.shape),
                                                    dtype=img.dtype, chunks=True)
                    dest_img[i] = img
                    dst_img_path_list.append(img_path)
                    dst_label_list.append(label)

                img_path_list = f.create_dataset('img_path_list',
                                                 (len(self), ),
                                                 h5py.special_dtype(vlen=str))
                img_path_list[:] = dst_img_path_list
            y_cls = self.y.__class__
            self.x = dst_img_path_list
            self.y = y_cls(dst_label_list)
        tfms = self._tfms
        self._tfms = []
        obj_to_pkl(self, pkl_path)
        self._tfms = tfms

    def load_img_cache(self, img_cache_path):
        '''load image cache from h5py

        img_cache_path: h5py file name
        '''
        import h5py
        try:
            if img_cache_path:
                cache_f = h5py.File(img_cache_path, 'r')
                self.cache_x = cache_f['img_path_list'][:].tolist()
                self.cache_img = cache_f['img']
        except Exception as e:
            print('Failed to load img cache!', e)
            self.cache_x = None
            self.cache_img = None
        return self

    @classmethod
    def load(cls, pkl_path, img_cache_path=None):
        '''load ImageLabelList from pkl

        pkl_path: pickle file name
        img_cache_path: h5py file name
        '''
        old_ill = obj_from_pkl(pkl_path)
        ill = cls.from_ill(old_ill)
        ill._train_idx = old_ill._train_idx
        ill._valid_idx = old_ill._valid_idx
        ill.load_img_cache(img_cache_path)
        return ill

    def load_image(self, img_path):
        '''load one image from disk
        '''
        if self.cache_img and self.cache_x and img_path in self.cache_x:
            idx = self.cache_x.index(img_path)
            return self.cache_img[idx]
        return open_image(img_path, self.img_mode)

    def __len__(self):
        '''len of this dataset
        '''
        return len(self.x)

    def __getitem__(self, idx):
        '''get one item

        # Arguments
            idx: index of dataset

        # Returns
            img, label
        '''
        img_path, label = self.x[idx], self.y[idx]
        img = self.load_image(img_path)
        img, label = self.do_tfms(self._tfms, img, label, idx, img_path)
        return img, label

    def show_sample(self, count=4, label=None, idx=None, ncols=2, figsize=(9, 9)):
        '''show sample of image/label

        # Arguments
            count: number of random sample
            label: only show this label
            idx: image index
        '''
        if idx is not None:
            if isinstance(idx, int):
                idx = [idx]
        else:
            if label is None:
                count = min(count, len(self))
                idx = np.random.randint(0, len(self), size=count).tolist()
            else:
                main_y = self.get_main_labels()
                label_idx = [i for i, l in enumerate(main_y) if l == label]
                count = min(count, len(label_idx))
                idx = np.random.choice(label_idx, size=count)

        if len(idx) > 0:
            img_list, label_list = zip(*[self[i] for i in idx])
            show_images(img_list, label_list, ncols, figsize)
        else:
            print('No image!')

    def label_dist(self, reverse=True):
        ''' get each label's dist
        Out: dict()
        '''
        labelset = self.labelset()
        if None in self.y:
            labelset.append(None)
        label_dist = {l: self.y.count(l) for l in labelset}
        label_dist = sorted(label_dist.items(), key=lambda data: data[1], reverse=reverse)
        label_dist = {str(l): num for l, num in label_dist}
        return label_dist

    def get_main_label_dist(self, reverse=True):
        return self.label_dist(reverse)

    def get_dist(self, sublabel_func=None, reverse=True):
        '''
        # Arguments
            sublabel_func:
                1) arg_cnt 1 (img_path):
                    eg1: lambda x: osp.basename(x)[5:7])
                    eg2: func 'get_image_res'
                    eg3: def sublabel_func(img_path):
                        return str

                2) arg_cnt 2 (img_path, y):
                    def sublabel_func(img_path, label):
                        return xxx

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
        label_list = self.get_main_labels()
        sublabel_list = list()
        for i, img_path in enumerate(tqdm(self.x)):
            yi = self.y[i]
            if sublabel_func.__code__.co_argcount == 1:
                sublabel = sublabel_func(img_path)
            elif sublabel_func.__code__.co_argcount == 2:
                sublabel = sublabel_func(img_path, yi)
            else:
                sublabel = sublabel_func(img_path, yi, i)
            sublabel_list.append(str(sublabel))

        sublabel_set = sorted(list(set(sublabel_list)))
        sub_dist_info = {sub: {label: 0 for label in labelset} for sub in sublabel_set}
        for sub_label, label in zip(sublabel_list, label_list):
            sub_dist_info[sub_label][label] += 1

        label_dist['sublabel_total'] = {sub: sublabel_list.count(sub) for sub in sublabel_set}
        label_dist.update(sub_dist_info)

        return label_dist

    def show_dist(self, sublabel_func=None, need_show=True, reverse=True):
        '''show distribution of dataset
        Note:
            1) ill vs ibll/ipll, is a little different when arg_cnt > 1
            2) suggest use just right arg_cnt.
                lambda x: get_image_res. (OK)
                lambda x,y: get_image_res. (Not Suggest)
        In:
            sublabel_func:
                support 3 kinds of args
                1) arg_cnt 1 (img_path)
                    def sublabel_func(img_path):
                        return str
                2) arg_cnt 2 (img_path, label)
                    def sublabel_func(img_path, label):
                        return str
                3) arg_cnt 3 (img_path, label, i)
                    def sublabel_func(img_path, label, i):
                        return str
        Example:
            1. ill
                sublabel_func:
                1) arg_cnt 1 (img_path):
                    eg1: get image type (.jpg/.bmp)
                        ill.show_dist(lambda x: osp.basename(x)[-4:]))
                    eg2: func 'get_image_res'
                        ill.show_dist(get_image_res)
                    eg3:
                        def sublabel_func(img_path):
                            return str
                        ill.show_dist(sublabel_func)

                2) arg_cnt 2 (img_path, y):
                    def sublabel_func(img_path, label):
                        return str

                3) arg_cnt 3 (img_path, y, i):
                    def sublabel_func(img_path, label, idx):
                        return str
            2. ibll/ipll
                sublabel_func:
                1) arg_cnt 1 (img_path):
                    similar to ill.show_dist()

                2) arg_cnt 2 (img_path, y):
                    eg1: to get box area level.
                        def box_area_level_func(img_path, label):
                            xxx = label['box'] # Note: 'box', not 'bboxes'
                            xxx = area_level_of_box(box)
                            return xxx
                        ibll.show_dist(sublabel_func=box_area_level_func)

                    eg2: to get defect level.
                        # Note: 'level', not 'levels'
                        ibll.show_dist(lambda x,y: y['level'])

                3) arg_cnt 3 (img_path, y, i):
                    def sublabel_func(img_path, label, idx):
                        return str
                    ibll.show_dist(sublabel_func)
        '''
        label_dist = self.get_dist(sublabel_func, reverse)
        if sublabel_func is None:
            count_data = {'labelset': list(label_dist.keys()),
                          'total': list(label_dist.values())}
            return plot_stack_bar('label distribution', count_data, 'labelset', ['total'],
                                  need_show=need_show)
        count_data = label_dist
        sublabels = list(count_data['sublabel_total'].keys())
        del count_data['sublabel_total']
        for k, v in count_data.items():
            if k == 'labelset':
                continue
            count_data[k] = list(v.values())
        return plot_stack_bar('label distribution', count_data, 'labelset', sublabels,
                              need_show=need_show)


class ZipImageLabelList:
    ''' (deprecated) zip multi ImageLabelList

    '''

    def __init__(self, *ill_list, key=None, reverse=False):
        '''
        ill_list (list): list of ill
        key (callable): A custom key function can be supplied to customize the sort order.
            eg: def func1(x):
                    return osp.basename(x)
        '''

        y_cls = ill_list[0].y.__class__
        if key:
            ill_id_set = [{key(x) for x in ill.x} for ill in ill_list]
            key_id_set = ill_id_set[0]
            for id_set in ill_id_set[1:]:
                key_id_set = key_id_set & id_set

            key_id_set = sorted(key_id_set, reverse=reverse)

            new_ill_list = list()
            for ill in ill_list:
                new_x, new_y = list(), list()
                for key_id in tqdm(key_id_set):
                    for x, y in zip(ill.x, ill.y):
                        _key_id = key(x)
                        if key_id == _key_id:
                            new_x.append(x)
                            new_y.append(y)
                            break
                new_ill = ill.copy()
                new_ill.x = new_x
                new_ill.y = y_cls(new_y)
                new_ill_list.append(new_ill)
            ill_list = new_ill_list

        # All ill must be the same length
        ill_lens = [len(ill) for ill in ill_list]
        assert ill_lens.count(ill_lens[0]) == len(ill_lens)

        self.x = [_x for x in zip(*[ill.x for ill in ill_list]) for _x in x]
        self.y = y_cls([_y for y in zip(*[ill.y for ill in ill_list]) for _y in y])
        self.ill_idx = list(range(len(ill_list))) * ill_lens[0]
        self.img_in_ill_idx = [_i for i in zip(*[list(range(ill_lens[0]))] * len(ill_list)) for _i in i]

        self.ill_list = ill_list
        self._tfms = []

    def get_ill_list(self):
        new_y_list = [self.y[i::len(self.ill_list)] for i in range(len(self.ill_list))]
        new_ill_list = list()
        for ill, new_y in zip(self.ill_list, new_y_list):
            y_cls = ill.y.__class__
            ill = ill.copy()
            ill.y = y_cls(new_y)
            new_ill_list.append(ill)
        return new_ill_list

    def __len__(self):
        return len(self.x)

    def copy(self):
        ''' Return one copy of ZipImageLabelList
        '''
        return copy.deepcopy(self)

    def labelset(self):
        return sorted(list(set([c for ill in self.ill_list for c in ill.labelset()])))

    def clear_tfms(self):
        '''clear all transform
        '''
        self._tfms = []
        return self

    def set_tfms(self, tfms):
        '''set all transform
        '''
        self._tfms = tfms.copy()

    def add_tfm(self, func):
        '''add one transform
        eg:
            def tfm_func(img, label):
                return new_img, new_label

            or:

            def tfm_func(img, label, idx, img_path):
                return new_img, new_label
        '''
        self._tfms.append(func)
        return self

    def do_tfms(self, tfms, img, label, idx, img_path):
        '''do all transform for one image
        '''
        label = copy.deepcopy(label)

        for tfm in tfms:
            if hasattr(tfm, '__code__'):
                if tfm.__code__.co_argcount == 1:
                    img = tfm(img)
                    continue
                elif tfm.__code__.co_argcount == 2:
                    result = tfm(img, label)
                    if isinstance(result, tuple):
                        img, label = result
                    else:
                        img = result
                    continue
            result = tfm(img, label, idx, img_path)
            if isinstance(result, tuple):
                img, label = result
            else:
                img = result
        return img, label

    def __getitem__(self, idx):
        ill = self.ill_list[self.ill_idx[idx]]
        img, _ = ill[self.img_in_ill_idx[idx]]
        img_path, label = self.x[idx], self.y[idx]
        img, label = self.do_tfms(self._tfms, img, label, idx, img_path)
        return img, label

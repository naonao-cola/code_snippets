'''
Copyright (C) 2023 TuringVision

List of image and label suitable for computer vision multi-task classfication task.
'''
from .image_data import ImageLabelList
from ..utils import *
from ..ui import *

__all__ = ['ImageMultiLabelList']


class ImageMultiLabelList(ImageLabelList):
    def __init__(self, img_path_list, label_list=None):
        '''
        # Arguments
            img_path_list: (list) list of image path
            label_list: (list) list of label
                    eg: [[main_code, sub_code], [main_code, sub_code], ...]
        '''
        super().__init__(img_path_list, label_list)
        self.main_label_idx = 0

    def set_main_label_idx(self, idx=0):
        assert idx < self.task_num()
        self.main_label_idx = idx

    def copy(self):
        ''' Return one copy of ImageMultiLabelList
        '''
        new_imll = super().copy()
        new_imll.main_label_idx = self.main_label_idx
        return new_imll

    @classmethod
    def from_label_info(cls, image_dir, label_info_dict):
        raise NotImplementedError

    @classmethod
    def from_turbox_data(cls, turbox_data):
        '''create from turbox format data
        '''
        classes = turbox_data.get('classList', None)
        img_path_list = []
        label_list = []
        for img_info in turbox_data['labelSet']:
            img_path = img_info['imagePath']
            img_path_list.append(img_path)
            ml = []
            for shape in img_info['shapes']:
                ml.append('Other' if classes and shape['label'] not in classes else shape['label'])
            label_list.append(ml)

        labelset = sorted(list(set([l for ll in label_list for l in ll])))
        imll_y = []
        for y in label_list:
            my = [l + '_0' for l in labelset]
            for yi in y:
                if yi in labelset:
                    my[labelset.index(yi)] = yi + '_1'
            imll_y.append(my)
        return cls(img_path_list, imll_y)

    def to_turbox_data(self, class_list=None):
        '''convert to turbox data
        '''
        turbox_data = {}
        org_labelset = [ls[0].rsplit('_', 1)[0] for ls in self.labelset()]
        turbox_data['classList'] = class_list if class_list is not None else org_labelset
        turbox_data['labelSet'] = []
        for img_path, labels in zip(self.x, self.y):
            img_info = {
                'imageName': osp.basename(img_path),
                'imagePath': img_path,
                'shapes': [{'label':l.rsplit('_', 1)[0]} for l in labels if l.endswith('_1')]
            }

            turbox_data['labelSet'].append(img_info)
        return turbox_data

    def to_label_info(self, class_list=None):
        raise NotImplementedError

    def databunch(self, train_tfms=None, valid_tfms=None, path='.',
                  bs=64, num_workers=1, show_dist=True,
                  batch_sampler=None, **kwargs):
        '''to fastai databunch

        # Arguments
            train_tfms: transform list for train dataset
            valid_tfms: transform list for valid dataset
            path: work path for fastai
            bs: batch size
            num_workers:
        # Returns
            fastai databunch
        '''
        from .multi_label_category_list import MultiLabelCategoryList

        data = super().databunch(train_tfms=train_tfms, valid_tfms=valid_tfms,
                                 path=path, bs=bs, num_workers=num_workers,
                                 show_dist=show_dist,
                                 label_cls=MultiLabelCategoryList,
                                 batch_sampler=batch_sampler,
                                 **kwargs)
        return data

    def label_from_folder(self):
        raise NotImplementedError

    def task_num(self):
        ''' number of category task
        '''
        return 1 if self.y[0] is None else len(self.y[0])

    def labelset(self):
        label_set_list = list()
        for i in range(self.task_num()):
            labelset = sorted(list(set({None if l is None else l[i] for l in self.y})))
            label_set_list.append(labelset)
        return label_set_list

    def get_main_labels(self):
        return [None if l is None else l[self.main_label_idx] for l in self.y]

    def get_main_labelset(self):
        return self.labelset()[self.main_label_idx]

    def label_dist(self, reverse=True):
        labelset_list = self.labelset()
        label_dist_list = list()
        for i, labelset in enumerate(labelset_list):
            main_y = [None if l is None else l[i] for l in self.y]
            label_dist = {l: main_y.count(l) for l in labelset}
            label_dist = sorted(label_dist.items(), key=lambda data: data[1], reverse=reverse)
            label_dist = {l: num for l, num in label_dist}
            label_dist_list.append(label_dist)
        return label_dist_list

    def get_main_label_dist(self, reverse=True):
        return self.label_dist(reverse)[self.main_label_idx]

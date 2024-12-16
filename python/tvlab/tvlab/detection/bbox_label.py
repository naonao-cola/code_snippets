'''
Copyright (C) 2023 TuringVision

List of label suitable for computer vision object detection task.
'''
import os
import os.path as osp
import numpy as np
from .bbox_overlaps import nms, bbox_overlaps
from ..ui import plot_stack_bar, plot_bokeh_histogram

__all__ = ['BBoxLabel', 'BBoxLabelList']


class BBoxLabel(dict):
    '''
    bbox_label (dict): {'labels': ['A', 'B'], 'bboxes': [[10, 20, 100, 200], [20, 40, 50, 80]]}

        bboxes: [l, t, r, b] or [l, t, r, b, conf]
    '''
    def __init__(self, bbox_label=None):
        if bbox_label is None:
            bbox_label = {'labels': [], 'bboxes': []}
        super().__init__(bbox_label)

    def __add__(self, bbox_label):
        import copy
        bbox_label = BBoxLabel(bbox_label)
        if len(self.labels()) == 0:
            return copy.deepcopy(bbox_label)
        new = copy.deepcopy(self)
        for key, v in new.items():
            if key in bbox_label:
                v += bbox_label[key]
            else:
                v += [None] * len(bbox_label['labels'])
        return new

    def __getitem__(self, y):
        if isinstance(y, int):
            item = {}
            for k, v in self.items():
                k = 'box' if k == 'bboxes' else k[:-1]
                item[k] = v[y]
            return item

        return super().__getitem__(y)

    def __setitem__(self, key, value):
        if isinstance(key, int):
            item = {}
            for k, v in self.items():
                k = 'box' if k == 'bboxes' else k[:-1]
                v[key] = value[k]
            return;

        return super().__setitem__(key, value)

    def pop(self, key):
        if isinstance(key, int):
            item = {}
            for k, v in self.items():
                k = 'box' if k == 'bboxes' else k[:-1]
                item[k] = v.pop(key)
            return item;

        return super().pop(key)

    def append(self, value):
        for k, v in self.items():
            k = 'box' if k == 'bboxes' else k[:-1]
            v.append(value[k])

    def labels(self):
        return self['labels']

    def bboxes(self):
        return self['bboxes']

    def deepcopy(self):
        import copy
        return copy.deepcopy(self)

    def nms(self, iou_threshold=0.5):
        '''
        iou_threshold (float or dict)
            eg1: 0.5
            eg2: {'label1': 0.6, 'label2': 0.3, default:0.5}, other key such as
            label3, label4 will use the value of the default key; beyond that if
            missing the default key, all other lables will use default value 1.0
            (meaning no nms)
        '''
        bboxes = self.bboxes()
        if not bboxes:
            return BBoxLabel()
        bboxes = np.array(bboxes)

        if isinstance(iou_threshold, dict):
            labels = self.labels()
            label_types = list(set(labels))
            default_value = iou_threshold.get('default', 1.0)
            keep = []
            for lt in label_types:
                value = iou_threshold.get(lt, default_value)
                idx_list = [i for i, la in enumerate(labels) if la == lt]
                if len(idx_list) == 0:
                    continue

                if value >= 1.0:
                    keep.extend(idx_list)
                    continue

                bb_list = [bb for i, (la, bb) in enumerate(zip(labels, bboxes)) if la == lt]
                bb_list = np.array(bb_list)
                scores = None
                if bb_list.shape[1] >= 5:
                    scores = bb_list[:, 4]
                    bb_list = bb_list[:, :4]

                cur_keep = nms(bboxes=bb_list, scores=scores, iou_threshold=value)
                keep.extend([idx_list[k] for k in cur_keep])

            keep.sort()
        else:
            scores = None
            if bboxes.shape[1] >= 5:
                scores = bboxes[:, 4]
                bboxes = bboxes[:, :4]
            keep = nms(bboxes, scores, iou_threshold=iou_threshold)

        new_label = dict()
        for key, item in self.items():
            new_label[key] = [item[i] for i in keep]
        return BBoxLabel(new_label)

    def tfm(self, tfm):
        '''
        tfm (callable):
            eg:
                def tfm(label, box):
                    ...
                    return label, box
        '''
        labels, bboxes = [], []
        for label, box in zip(self.labels(), self.bboxes()):
            label, box = tfm(label, box)
            labels.append(label)
            bboxes.append(box)

        new_bbox_label = self.deepcopy()
        new_bbox_label["labels"] = labels
        new_bbox_label["bboxes"] = bboxes
        return new_bbox_label

    def tfm_item(self, tfm):
        '''
        tfm (callable):
            eg:
                def tfm(item):
                    item['label'] = xxx
                    item['box'] = xxx
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

        new_bbox_label = self.deepcopy()
        new_bbox_label["labels"] = labels
        return new_bbox_label

    def tfm_bbox(self, tfm):
        '''
        tfm (callable):
            eg:
                def tfm(box):
                    ...
                    return bboxes
        '''
        bboxes = []
        for box in self.bboxes():
            box = tfm(box)
            bboxes.append(box)

        new_bbox_label = self.deepcopy()
        new_bbox_label["bboxes"] = bboxes
        return new_bbox_label

    def crop(self, box, iof_threshold=0.3):
        bboxes = self.bboxes()
        if not bboxes:
            return BBoxLabel()

        labels = self.labels()
        tbboxes = [box]
        overlaps = bbox_overlaps(np.array([box[:4] for box in bboxes], np.double),
                                 np.array(tbboxes, np.double), 'iof')
        overlaps = overlaps[:, 0]
        new_labels = []
        new_bboxes = []
        crop_w = box[2] - box[0]
        crop_h = box[3] - box[1]
        if overlaps.max() > iof_threshold:
            for j, iof in enumerate(overlaps.tolist()):
                if iof > iof_threshold:
                    new_labels.append(labels[j])
                    (l,t,r,b) = bboxes[j][:4]
                    l = max(0, min(l - box[0], crop_w))
                    r = max(0, min(r - box[0], crop_w))
                    t = max(0, min(t - box[1], crop_h))
                    b = max(0, min(b - box[1], crop_h))
                    new_bboxes.append([l,t,r,b]+bboxes[j][4:])
        return BBoxLabel({'labels': new_labels, 'bboxes': new_bboxes})

    def to_pascal_voc(self, img_path, out_dir):
        from PIL import Image
        from .pascal_voc_io import PascalVocWriter
        labels = self.labels()
        bboxes = self.bboxes()
        extern_infos = self.get("extern_infos", [dict()] * len(labels))
        file_name = osp.basename(img_path)
        os.makedirs(out_dir, exist_ok=True)

        try:
            with Image.open(img_path) as im:
                image_shape = (im.width, im.height)
            writer = PascalVocWriter(osp.basename(osp.dirname(img_path)),
                                     file_name, image_shape)
            writer.verified = False

            for label, box, extern_info in zip(labels, bboxes, extern_infos):
                writer.addBndBox(box[0], box[1], box[2], box[3], label, False, extern_info)
            out_path = osp.join(out_dir, file_name[:file_name.rindex('.')]+'.xml')
            writer.save(targetFile=out_path)
        except Exception as e:
            print(e)

    def to_disk(self, img_path, out_dir):
        self.to_pascal_voc(img_path, out_dir)

    @classmethod
    def from_pascal_voc(cls, path):
        from .pascal_voc_io import PascalVocReader
        reader = PascalVocReader(path)
        shapes = reader.getShapes()
        extern_infos = reader.getExternInfos()
        bboxes = list()
        labels = list()
        for lb_info in shapes:
            label = lb_info["name"]
            points = lb_info["points"]
            lt = points[0]
            rb = points[2]
            bboxes.append([lt[0], lt[1], rb[0], rb[1]])
            labels.append(label)
        return BBoxLabel({'labels': labels, 'bboxes': bboxes, 'extern_infos': extern_infos})

    @classmethod
    def from_disk(cls, path):
        return cls.from_pascal_voc(path)

    def offset(self, x_off, y_off):
        def _add_offset(box):
            l,t,r,b = box[:4]
            l += x_off
            r += x_off
            t += y_off
            b += y_off
            return [l,t,r,b]+box[4:]
        return self.tfm_bbox(_add_offset)

    def scale(self, x_scale, y_scale):
        def _scale(box):
            l,t,r,b = box[:4]
            l *= x_scale
            r *= x_scale
            t *= y_scale
            b *= y_scale
            return [l,t,r,b]+box[4:]
        return self.tfm_bbox(_scale)

    def limit_min_size(self, size=32):
        '''
        size (int or tuple): (w, h)
        '''
        if isinstance(size, tuple):
            min_w, min_h = size
        else:
            min_w, min_h = size, size

        def _limit_min(box):
            l, t, r, b = box[:4]
            x = (l + r) / 2
            y = (t + b) / 2
            w = max(min_w, (r - l))
            h = max(min_h, (b - t))
            l = x - w / 2
            t = y - h / 2
            r = x + w / 2
            b = y + h / 2
            return [l, t, r, b] + box[4:]

        return self.tfm_bbox(_limit_min)

    def filter(self, key):
        '''
        key (callable):
            eg:
                def key(label, box):
                    ...
                    return True or False (True for keep)
        '''
        keep = []
        for i, (label, box) in enumerate(zip(self.labels(), self.bboxes())):
            ret = key(label, box)
            if ret:
                keep.append(i)

        new_label = dict()
        for key, item in self.items():
            new_label[key] = [item[i] for i in keep]

        return BBoxLabel(new_label)

    def filter_item(self, key):
        '''
        key (callable):
            eg:
                def key(item):
                    label = item['label']
                    box = item['box']
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

        return BBoxLabel(new_label)

    def filter_by_label(self, key):
        '''
        key (callable):
            eg:
                def key(label):
                    ...
                    return True or False (True for keep)
        '''
        def _label_filter(label, box):
            return key(label)

        return self.filter(_label_filter)

    def filter_by_bbox(self, key):
        '''
        key (callable):
            eg:
                def key(box):
                    ...
                    return True or False (True for keep)
        '''
        def _bbox_filter(label, box):
            return key(box)

        return self.filter(_bbox_filter)


class BBoxLabelList(list):
    '''
    bbox_label_list (list): [{'labels': ['A', 'B'],
                  'bboxes': [[10, 20, 100, 200], [20, 40, 50, 80]]}, ... ]
    '''
    def __init__(self, bbox_label_list):
        super().__init__([BBoxLabel(bbox_label) for bbox_label in bbox_label_list])

    def labels(self):
        return [l for bbox_label in self for l in bbox_label['labels']]

    def bboxes(self):
        return [box for bbox_label in self for box in bbox_label['bboxes']]

    def labelset(self):
        return sorted(list(set(self.labels())))

    def nms(self, iou_threshold=0.5):
        '''
        iou_threshold (float or dict)
            eg1: 0.5
            eg2: {'label1': 0.6, 'label2': 0.3, default:0.5}, other key such as
            label3, label4 will use the value of the default key; beyond that if
            missing the default key, all other lables will use default value 1.0
            (meaning no nms)
        '''
        return BBoxLabelList([bbox_label.nms(iou_threshold=iou_threshold) for bbox_label in self])

    def tfm(self, tfm):
        '''
        tfm (callable):
            eg:
                def tfm(label, box):
                    ...
                    return label, box
        '''
        return BBoxLabelList([bbox_label.tfm(tfm=tfm) for bbox_label in self])

    def tfm_item(self, tfm):
        '''
        tfm (callable):
            eg:
                def tfm(item):
                    item['label'] = xxx
                    item['box'] = xxx
                    ...
                    return item
        '''
        return BBoxLabelList([bbox_label.tfm_item(tfm=tfm) for bbox_label in self])

    def tfm_label(self, tfm):
        '''
        tfm (callable):
            eg:
                def tfm(label):
                    ...
                    return label
        '''
        return BBoxLabelList([bbox_label.tfm_label(tfm=tfm) for bbox_label in self])

    def tfm_bbox(self, tfm):
        '''
        tfm (callable):
            eg:
                def tfm(box):
                    ...
                    return bboxes
        '''
        return BBoxLabelList([bbox_label.tfm_bbox(tfm=tfm) for bbox_label in self])

    def crop(self, box):
        return BBoxLabelList([bbox_label.crop(box=box) for bbox_label in self])

    def offset(self, x_off, y_off):
        return BBoxLabelList([bbox_label.offset(x_off, y_off) for bbox_label in self])

    def scale(self, x_scale, y_scale):
        return BBoxLabelList([bbox_label.scale(x_scale, y_scale) for bbox_label in self])

    def limit_min_size(self, size=32):
        '''
        size (int or tuple): (w, h)
        '''
        return BBoxLabelList([bbox_label.limit_min_size(size=size) for bbox_label in self])

    def filter(self, key):
        '''
        key (callable):
            eg:
                def key(label, box):
                    ...
                    return True or False (True for keep)
        '''
        return BBoxLabelList([bbox_label.filter(key=key) for bbox_label in self])

    def filter_item(self, key):
        '''
        key (callable):
            eg:
                def key(item):
                    label = item['label']
                    box = item['box']
                    ...
                    return True or False (True for keep)
        '''
        return BBoxLabelList([bbox_label.filter_item(key=key) for bbox_label in self])

    def filter_by_label(self, key):
        '''
        key (callable):
            eg:
                def key(label):
                    ...
                    return True or False (True for keep)
        '''
        return BBoxLabelList([bbox_label.filter_by_label(key=key) for bbox_label in self])

    def filter_by_bbox(self, key):
        '''
        key (callable):
            eg:
                def key(box):
                    ...
                    return True or False (True for keep)
        '''
        return BBoxLabelList([bbox_label.filter_by_bbox(key=key) for bbox_label in self])

    def show_bbox_dist(self, labelset=None, x='size', y='ratio',
                       xbins=20, ybins=20, need_show=True):
        ''' show distribution of bboxes
        labelset (list): only show class label in labelset
        x (str): one of ('ratio', 'size', 'area', 'w', 'h', 'x', 'y')
        y (str): one of ('ratio', 'size', 'area', 'w', 'h', 'x', 'y')
        '''
        bbox_label_list = self
        if labelset:
            bbox_label_list = self.filter_by_label(lambda label: label in labelset)

        bboxes = np.array([box[:4] for box in bbox_label_list.bboxes() if box])
        if len(bboxes) <= 0:
            print('no bboxes!')
            return

        all_w = bboxes[:, 2] - bboxes[:, 0]
        all_h = bboxes[:, 3] - bboxes[:, 1]
        all_x = (bboxes[:, 2] + bboxes[:, 0]) / 2
        all_y = (bboxes[:, 3] + bboxes[:, 1]) / 2

        _data_map = {'w': all_w, 'h': all_h, 'area': all_h * all_w,
                     'size': np.sqrt(all_h * all_w), 'ratio': all_h / all_w,
                     'x': all_x, 'y': all_y}

        data = {x: _data_map[x], y: _data_map[y]}
        return plot_bokeh_histogram('area=h*w, ratio=h/w, size=sqrt(h*w)',
                                    data, xlabel=x, ylabel=y,
                                    xbins=xbins, ybins=ybins, need_show=need_show)

    def show_dist(self, need_show=True):
        '''show distribution
        '''
        labelset = self.labelset()
        labelset.append('backgroud')

        all_labels = self.labels()
        all_labels += ['backgroud' for bbox_label in self if not bbox_label['labels']]
        count_data = {'labelset': labelset,
                      'total': [all_labels.count(l) for l in labelset]}

        return plot_stack_bar('label distribution', count_data, 'labelset', ['total'],
                              need_show=need_show)

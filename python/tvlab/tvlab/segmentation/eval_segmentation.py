'''
Copyright (C) 2023 TuringVision

Visualization evaluation result of instance segmentation model.
'''

import numpy as np
from bisect import bisect_right
from .polygon_label import PolygonLabelList, PolygonLabel
from .polygon_overlaps import polygon_overlaps
from ..category.eval_category import (to_evaluate_result, EvalCategory,
                                      to_threshold_dict, merge_eval_result,
                                      plot_categorical_curcve)
from ..ui import ImagePolygonCleaner
from ..utils import *

__all__ = ['EvalSegmentation', 'compare_polygons_overlaps']


def _check_polygons_only(y_pred):
    for y in y_pred:
        for l in y['labels']:
            if l != 'object':
                return False
    return True


def _convert_y_format(y):
    '''
    convert eg1 to eg2
    '''
    return [{'polygons': img_y, 'labels': ['object'] * len(img_y)} for img_y in y]


def _get_conf_list(y_pred):
    conf_set = {float(format(polygon[-1], '0.2f'))
                for pred in y_pred for polygon in pred['polygons']}
    if not conf_set:
        conf_set = {0.0}
    conf_set.add(max(min(conf_set) - 0.01, 0.0))
    conf_set.add(min(max(conf_set) + 0.01, 1.0))
    conf_list = list(conf_set)
    conf_list.sort()
    return conf_list


def _get_cls_result(y, c, need_idxs=False):
    y_polygons = list()
    y_idxs = []
    for img_y in y:
        new_polygons = list()
        idxs = list()
        for i, (polygon, label) in enumerate(zip(img_y['polygons'], img_y['labels'])):
            if label in [c, 'object']:
                new_polygons.append(polygon)
                idxs.append(i)
        y_polygons.append(new_polygons)
        y_idxs.append(idxs)
    if need_idxs:
        return y_polygons, y_idxs
    return y_polygons


def compare_polygons_overlaps(ppolygons, tpolygons, iou_threshold=0.5):
    '''
    compare two polygons
    in:
        ppolygons: (k, X) predict polygons
        tpolygons: (n, X) target polygons
        iou_threshold: iou threshold
    out:
        predict_idxs: missed predict index
        target_idxs: missed target index
    '''
    predict_idxs = list(range(len(ppolygons)))
    target_idxs = list(range(len(tpolygons)))

    if not predict_idxs or not target_idxs:
        return predict_idxs, target_idxs

    overlaps = polygon_overlaps(tpolygons, ppolygons)

    for n in range(len(tpolygons)):
        if overlaps[n].max() >= iou_threshold:
            target_idxs.remove(n)
            maxi = overlaps[n].argmax()
            predict_idxs.remove(maxi)
            overlaps[:, maxi] = 0

    return predict_idxs, target_idxs


def _segmentation_eval(y_pred, y_true, iou_threshold=0.5, classes=None):
    '''
    # Arguments:
        y_pred: (list)
            [ {'polygons': [[10, 20, 100, 200, 1.0], ...],
               'labels': ['A', 'B', ...]
              }, # for a.jpg
              {'polygons': [], 'labels': []}, # for b.jpg no polygons
             ...
            ]

        y_true: (list)
            eg2:
            [ {'polygons': [[10, 20, 100, 200], ...],
               'labels': ['A', 'B', ...]
              }, # for a.jpg
              {'polygons': [], 'labels': []}, # for b.jpg no polygons
             ...
            ]

        iou_threshold (float): IoU threshold
        classes: (list) class list
    '''

    assert len(y_pred) == len(y_true)

    conf_list = _get_conf_list(y_pred)
    hit_cnts = np.zeros((len(classes), len(conf_list)))
    pick_cnts = np.zeros((len(classes), len(conf_list)))
    target_cnt = [0 for _ in classes]

    for ci, c in enumerate(classes):
        y_ppolygons = _get_cls_result(y_pred, c)
        y_tpolygons = _get_cls_result(y_true, c)
        for ppolygons, tpolygons in zip(y_ppolygons, y_tpolygons):
            target_cnt[ci] += len(tpolygons)
            if ppolygons:
                pscores = [p[-1] for p in ppolygons]
                ppolygons = [p[:-1] for p in ppolygons]
                predict_idxs, _ = compare_polygons_overlaps(ppolygons=ppolygons,
                                                            tpolygons=tpolygons,
                                                            iou_threshold=iou_threshold)
                for i, score in enumerate(pscores):
                    conf_index = bisect_right(conf_list, score)
                    pick_cnts[ci, :conf_index] += 1
                    if i in predict_idxs:
                        predict_idxs.remove(i)
                    else:
                        hit_cnts[ci, :conf_index] += 1

    result_list = []
    for i, conf in enumerate(conf_list):
        total_pred_cnt = 0
        for img_y in y_pred:
            for polygons in img_y['polygons']:
                if polygons[-1] >= conf:
                    total_pred_cnt += 1

        total_hit_cnt = sum(hit_cnts[:, i])
        total_pick_cnt = sum(pick_cnts[:, i])
        if total_pick_cnt > total_pred_cnt:
            # for predict with class label
            other_pick_cnt = total_pred_cnt - total_hit_cnt
            c_cnt = len(classes)
            for ci in range(c_cnt):
                other_cnt = other_pick_cnt//c_cnt
                if ci == (c_cnt - 1):
                    other_cnt = other_pick_cnt%c_cnt + other_cnt
                pick_cnts[ci, i] = hit_cnts[ci, i] + other_cnt

        result = to_evaluate_result(hit_cnts[:, i], pick_cnts[:, i], target_cnt, classes, conf)
        result_list.append(result)
    return result_list


def _get_segmentation_error_list(y_pred, y_true, iou_threshold=0.5,
                                 classes=None, overshot=True, miss=True):
    '''
    # Arguments:
        y_pred: (list)
            [ {'polygons': [[10, 20, 100, 200, 1.0], ...],
               'labels': ['A', 'B', ...]
              }, # for a.jpg
              {'polygons': [], 'labels': []}, # for b.jpg no polygons
             ...
            ]

        y_true: (list)
            eg2:
            [ {'polygons': [[10, 20, 100, 200], ...],
               'labels': ['A', 'B', ...]
              }, # for a.jpg
              {'polygons': [], 'labels': []}, # for b.jpg no polygons
             ...
            ]

        iou_threshold (float): IoU threshold
        overshot (bool): include overshot
        miss (bool): include miss
    Returns:
        err_idx (list, n): index of error image
        err_pred_idx (list, (n, x)), index of error predict idx
        err_targ_idx (list, (n, y)), index of error target idx
    '''
    err_idx = list()
    err_pred_idx = list()
    err_targ_idx = list()

    for ci, c in enumerate(classes):
        y_ppolygons, y_pidxs = _get_cls_result(y_pred, c, True)
        y_tpolygons, y_tidxs = _get_cls_result(y_true, c, True)
        for i, (ppolygons, tpolygons) in enumerate(zip(y_ppolygons, y_tpolygons)):
            ppolygons = [p[:-1] for p in ppolygons]
            predict_idxs, target_idxs = compare_polygons_overlaps(ppolygons=ppolygons,
                                                                  tpolygons=tpolygons,
                                                                  iou_threshold=iou_threshold)
            if (overshot and predict_idxs) or (miss and target_idxs):
                if i not in err_idx:
                    err_idx.append(i)
                    err_pred_idx.append([y_pidxs[i][j] for j in predict_idxs])
                    err_targ_idx.append([y_tidxs[i][j] for j in target_idxs])
                else:
                    idx = err_idx.index(i)
                    err_pred_idx[idx] += [y_pidxs[i][j] for j in predict_idxs]
                    err_targ_idx[idx] += [y_tidxs[i][j] for j in target_idxs]
    return err_idx, err_pred_idx, err_targ_idx


class EvalSegmentation(EvalCategory):
    ''' evaluate object detection for each class

    # Arguments
        y_pred: (list)
            eg1:
            [
            #   l,  t,   r,   b, conf
             [[x1, y1, x2, y2, ..., xn, yn, 1.0],  ...], # for a.jpg
             [[x1, y1, x2, y2, ..., xn, yn, 0.8],  ...], # for b.jpg
             [], # for c.jpg no polygons
             ...
            ]
            eg2:
            [ {'polygons': [[x1, y1, x2, y2, ..., xn, yn, 1.0], ...],
               'labels': ['A', 'B', ...]
              }, # for a.jpg
              {'polygons': [], 'labels': []}, # for b.jpg no polygons
             ...
            ]

        y_true: (list)
            eg1:
            [
            #   l,  t,   r,   b, conf
             [[x1, y1, x2, y2, ..., xn, yn],  ...], # for a.jpg
             [[x1, y1, x2, y2, ..., xn, yn],  ...], # for b.jpg
             [], # for c.jpg no polygons
             ...
            ]
            eg2:
            [ {'polygons': [[x1, y1, x2, y2, ..., xn, yn], ...],
               'labels': ['A', 'B', ...]
              }, # for a.jpg
              {'polygons': [], 'labels': []}, # for b.jpg no polygons
             ...
            ]
        iou_threshold (float): IoU threshold
        classes: (list) class list
    '''

    def __init__(self, y_pred, y_true, iou_threshold=0.5, classes=None,
                 polygons_only=False, conf_threshold=0.0,
                 precision_threshold=None, recall_threshold=None):
        if not isinstance(y_true[0], dict):
            # convert eg1 format to eg2 format
            y_true = _convert_y_format(y_true)

        if not isinstance(y_pred[0], dict):
            # convert eg1 format to eg2 format
            y_pred = _convert_y_format(y_pred)

        if polygons_only:
            for y in y_pred:
                y['labels'] = ['object' for _ in y['labels']]

        if not classes:
            label_set = set()
            for info in y_true:
                if info['labels']:
                    for l in info['labels']:
                        label_set.add(l)
            for info in y_pred:
                if info['labels']:
                    for l in info['labels']:
                        label_set.add(l)
            if label_set:
                if 'object' in label_set and len(label_set) > 1:
                    label_set.remove('object')
                classes = sorted(list(label_set))
            else:
                classes = ['object']

        self.polygons_only = _check_polygons_only(y_pred)

        self.y_pred = PolygonLabelList(y_pred)
        self.y_true = PolygonLabelList(y_true)
        self.y_true = self.y_true.filter_by_polygon(lambda p: p if len(p) % 2 == 0 else p[:-1])
        self.iou_threshold = iou_threshold
        self.classes = classes

        self._result = None
        self._result_list = None
        self.get_result_list()

        conf_threshold = to_threshold_dict(conf_threshold, classes)
        precision_threshold = to_threshold_dict(precision_threshold, classes)
        recall_threshold = to_threshold_dict(recall_threshold, classes)
        self._result = merge_eval_result(classes, self._result_list,
                                         conf_threshold,
                                         precision_threshold,
                                         recall_threshold)
        for label in self._result['classes']:
            conf_threshold[label] = self._result[label]['confidence']
        self._conf_threshold = conf_threshold


    def get_result_list(self):
        if not self._result_list:
            self._result_list = _segmentation_eval(y_pred=self.y_pred, y_true=self.y_true,
                                                   iou_threshold=self.iou_threshold,
                                                   classes=self.classes)
        return self._result_list

    def to_pkl(self, pkl_path):
        result = {'y_pred': self.y_pred,
                  'y_true': self.y_true,
                  'classes': self.classes,
                 }
        obj_to_pkl(result, pkl_path)

    @classmethod
    def from_pkl(cls, pkl_path, **kwargs):
        result = obj_from_pkl(pkl_path)
        return cls(y_pred=result['y_pred'],
                   y_true=result['y_true'],
                   classes=result['classes'],
                   **kwargs)

    def plot_bokeh_table(self, title=None, need_show=True):
        if not title:
            title = 'iou threshold {} '.format(self.iou_threshold)
            if self.polygons_only:
                title += ', polygons only!'
        return super().plot_bokeh_table(title, need_show)

    def update_threshold(self, **kwargs):
        raise NotImplementedError

    def plot_confusion_matrix(self, conf_threshold=None, need_show=True):
        raise NotImplementedError

    def plot_bokeh_scatter(self, target=None, predict=None, yaxis='target', need_show=True):
        raise NotImplementedError

    def plot_x_iou(self, x='recall', need_show=True):
        ious = np.arange(0.05, 0.8, 0.05)
        iou_result_list = list()
        for iou in ious:
            result_list = _segmentation_eval(y_pred=self.y_pred,
                                             y_true=self.y_true,
                                             iou_threshold=iou,
                                             classes=self.classes)
            result = merge_eval_result(self.classes, result_list,
                                       None, None, None)
            if 'Total' in result:
                result['Total']['iou'] = iou
            if 'TotalNoOther' in result:
                result['TotalNoOther']['iou'] = iou

            for c in self.classes:
                result[c]['iou'] = iou
            iou_result_list.append(result)
        return plot_categorical_curcve(iou_result_list, 'iou', x, need_show=need_show)

    def plot_recall_iou(self, need_show=True):
        return self.plot_x_iou(x='recall', need_show=need_show)

    def plot_precision_iou(self, need_show=True):
        return self.plot_x_iou(x='precision', need_show=need_show)

    def get_error_images(self, iou_threshold=None, overshot=True, miss=True):
        """ get error images
        """
        if not iou_threshold:
            iou_threshold = self.iou_threshold
        error_result = _get_segmentation_error_list(y_pred=self.y_pred, y_true=self.y_true,
                                                    iou_threshold=iou_threshold,
                                                    classes=self.classes,
                                                    overshot=overshot, miss=miss)
        return error_result

    def _get_error_images_with_group(self, ipll, iou_threshold=None,
                                     overshot=True, miss=True,
                                     groud_num=1, with_text=True,
                                     drop_match=False):
        '''
        # Arguments
            ipll: ImagePolygonLabelList
            iou_threshold:
            groud_num: number of image of one groud
            overshot: need show overshot images?
            miss: need show miss images?
        '''
        ipll = ipll.copy()
        error_result = self.get_error_images(iou_threshold, overshot, miss)
        err_idx, err_pred_idx, err_targ_idx = error_result
        find_idxs = list()
        for i in range(len(ipll)//groud_num):
            for j in range(groud_num):
                n = i*groud_num + j
                if n in err_idx:
                    find_idxs += [i*groud_num + j for j in range(groud_num)]
                    break

        def _add_diff_result(i, labels):
            import copy
            labels = copy.deepcopy(labels)
            yt_err_idx = list()
            yp = self.y_pred[i]
            if i in err_idx:
                idx = err_idx.index(i)
                yt_err_idx = err_targ_idx[idx]

            colors = ['gold' if i in yt_err_idx else 'springgreen'
                      for i in range(len(labels['labels']))]

            labels['colors'] = colors
            labels['polygons'] += yp['polygons']
            labels['labels'] += ['{}:{:.2f}'.format(l, box[-1])
                                 if with_text else ''
                                 for l, box in zip(yp['labels'], yp['polygons'])]
            labels['colors'] += ['red'] * len(yp['polygons'])
            for k in labels.keys():
                if k not in ['colors', 'labels', 'polygons']:
                    labels[k] += [None] * len(yp['labels'])
            if drop_match:
                keep = [i for i, c in enumerate(colors) if c != 'springgreen']
                new_label = dict()
                for key, item in labels.items():
                    new_label[key] = [item[i] for i in keep]
                labels = PolygonLabel(new_label)
            return labels

        for i in find_idxs:
            ipll.y[i] = _add_diff_result(i, ipll.y[i])

        return ipll, find_idxs

    def export_error_images(self, ipll, out_dir, iou_threshold=None,
                            overshot=True, miss=True, format=None):
        """ export error images to out dir
        # Arguments
            ipll: ImagePolygonLabelList
            out_dir: directory for save output image
            iou_threshold:
            overshot: need show overshot images?
            miss: need show miss images?
            format: 'jpg', 'png', 'bmp' ...
        """
        ipll = ipll.copy()
        ipll.set_img_mode('RGB')
        import shutil
        from PIL import Image
        os.makedirs(out_dir)

        ipll, find_idxs = self._get_error_images_with_group(ipll, iou_threshold=iou_threshold,
                                                            overshot=overshot, miss=miss,
                                                            groud_num=1, with_text=True)


        for i in find_idxs:
            img, y = ipll[i]
            img = draw_polygons_on_img_pro(img, y['polygons'], y['labels'], colors=y['colors'])
            dst_path = osp.join(out_dir, osp.basename(ipll.x[i]))
            if format:
                dst_path = osp.splitext(dst_path)[0] + '.' + format
            Image.fromarray(img).save(dst_path)

    def export_error_crop(self, ipll, out_dir,
                          iou_threshold=None,
                          overshot=True, miss=True,
                          border=64, keep_img_dir=0,
                          **kwargs):
        """ export cropped error image and labels
        # Arguments
            ibll: ImageBBoxLabelList
            out_dir: directory for save output image
            iou_threshold:
            overshot: need show overshot images?
            miss: need show miss images?
            border (int): width/height expand border size for defect crop.
        """
        _ipll, find_idxs = self._get_error_images_with_group(ipll, iou_threshold=iou_threshold,
                                                            overshot=overshot, miss=miss,
                                                            groud_num=1, with_text=True, drop_match=True)
        _ipll = _ipll.__class__([_ipll.x[i] for i in find_idxs], [_ipll.y[i] for i in find_idxs])
        _err_crop_ipll = _ipll.crop(out_dir, border=border, keep_img_dir=keep_img_dir, **kwargs)
        for yi in _err_crop_ipll.y:
            yi['colors'] = ['gold' if ':' not in l else 'red' for l in yi['labels']]
        return _err_crop_ipll

    def show_error_images_with_group(self, ipll, iou_threshold=None,
                                     overshot=True, miss=True,
                                     groud_num=1,
                                     with_text=True,
                                     notebook_url=None,
                                     **kwargs):
        '''
        # Arguments
            ipll: ImagePolygonLabelList
            iou_threshold:
            groud_num: number of image of one groud
            overshot: need show overshot images?
            miss: need show miss images?
        '''
        labelset = kwargs.get('labelset', None)
        if not labelset:
            if not ipll.labelset():
                kwargs['labelset'] = self.classes
            else:
                kwargs['labelset'] = ipll.labelset()

        ipll, find_idxs = self._get_error_images_with_group(ipll, iou_threshold=iou_threshold,
                                                            overshot=overshot, miss=miss,
                                                            groud_num=groud_num, with_text=with_text)
        return ImagePolygonCleaner(ipll, find_idxs=find_idxs,
                                   notebook_url=notebook_url, **kwargs)

    def show_error_images(self, ipll, iou_threshold=None,
                          overshot=True, miss=True,
                          with_text=True,
                          notebook_url=None,
                          **kwargs):
        '''
        # Arguments
            ipll: ImagePolygonLabelList
            iou_threshold:
            overshot: need show overshot images?
            miss: need show miss images?
        '''
        return self.show_error_images_with_group(ipll, iou_threshold=iou_threshold,
                                                 overshot=overshot, miss=miss,
                                                 groud_num=1, with_text=with_text,
                                                 notebook_url=notebook_url, **kwargs)

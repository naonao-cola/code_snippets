'''
Copyright (C) 2023 TuringVision

Visualization evaluation result of category model.
'''

import numpy as np
import concurrent.futures
from bisect import bisect_right
from collections import OrderedDict
from ..utils import *
from ..ui import (get_one_color, plot_table, plot_lines,
                  plot_bokeh_matrix, plot_bokeh_scatter,
                  ImageCleaner)

__all__ = ['EvalCategory']


def to_threshold_dict(threshold, classes):
    threshold_dict = None
    if isinstance(threshold, float):
        threshold_dict = {c: threshold for c in classes}
    elif isinstance(threshold, dict):
        default = threshold.get('default', 1.0)
        threshold_dict = {c: default for c in classes}
        for key, value in threshold.items():
            if key in threshold_dict:
                threshold_dict[key] = value
    return threshold_dict


def _get_top_k(true_name, pred, top_k):
    yp_sort = [{'class': k, 'conf': conf}
               for k, conf in pred.items()]
    yp_sort.sort(key=lambda x: x['conf'], reverse=True)
    yp_top_k = yp_sort[:top_k]
    yp_top_k_name = [p['class'] for p in yp_top_k]
    yp_top_k_conf = [p['conf'] for p in yp_top_k]
    yp_conf = sum(yp_top_k_conf)
    if true_name in yp_top_k_name:
        yp_name = true_name
    else:
        yp_name = yp_top_k_name[0]
    return yp_name, yp_conf


def _update_total_result(result):
    hit = pick = target_sum = confs = percent = 0
    precision_sum = 0
    class_num = 0
    for value in result.values():
        if not isinstance(value, dict) or 'hit' not in value:
            continue
        hit += value['hit']
        pick += value['pick']
        target_sum += value['total']
        confs += value['confidence'] * value['total']
        percent += value['percent']
        if 'ap' in value:
            precision_sum += value['ap']
            class_num += 1
    mean_ap = safe_div(precision_sum, class_num)
    precision = safe_div(hit, pick)
    recall = safe_div(hit, target_sum)
    f1score = safe_div(2 * precision * recall, (precision + recall))
    total_label = 'Total'
    if 'Other' not in result:
        total_label = 'TotalNoOther'
    result[total_label] = {'precision': precision,
                           'recall': recall,
                           'ap': mean_ap,
                           'f1': f1score,
                           'confidence': safe_div(confs, target_sum),
                           'hit': hit, 'pick': pick,
                           'total': target_sum,
                           'percent': percent}

    if 'Other' in result:
        hit -= result['Other']['hit']
        pick -= result['Other']['pick']
        other_target = result['Other']['total']
        target_sum -= other_target
        confs -= result['Other']['confidence'] * other_target
        precision = safe_div(hit, pick)
        mean_ap = safe_div((precision_sum - result['Other']['precision']), (class_num - 1))
        recall = safe_div(hit, target_sum)
        f1score = safe_div(2 * precision * recall, (precision + recall))
        result['TotalNoOther'] = {'precision': precision, 'recall': recall,
                                  'ap': mean_ap, 'f1': f1score,
                                  'confidence': safe_div(confs, target_sum), 'hit': hit,
                                  'pick': pick, 'total': target_sum,
                                  'percent': percent - result['Other']['percent']}


def to_evaluate_result(hit_cnt, pick_cnt, target_cnt, classes, threshold):
    result = dict()

    target_sum = sum(target_cnt)
    for i, label in enumerate(classes):
        hit = hit_cnt[i]
        pick = pick_cnt[i]
        target = target_cnt[i]
        precision = safe_div(hit, pick)
        recall = safe_div(hit, target)
        f1score = safe_div(2 * precision * recall, (precision + recall))
        conf = threshold
        if isinstance(threshold, dict):
            conf = threshold[label]
        result[label] = {'precision': precision, 'recall': recall,
                         'f1': f1score, 'confidence': conf,
                         'hit': hit, 'pick': pick,
                         'total': target, 'percent': safe_div(target, target_sum)}

    _update_total_result(result)

    result['classes'] = classes.copy()

    return result


def _print_one_evaluate_result(label, single_result):
    print(('{:18} {:>#9.1f}% {:>#7.1f}% {:>#7.1f}% {:>#7.1f}%'
           + ' {:>#8.2f} {:>8} {:>8} {:>8} {:>#7.1f}%').format(
        label, 100 * single_result['precision'],
        100 * single_result['recall'], 100 * single_result['ap'],
        100 * single_result['f1'], single_result['confidence'],
        int(single_result['hit']), int(single_result['pick']),
        int(single_result['total']), 100 * single_result['percent']))


def _categorical_eval(y_true_str,
                      y_pred_dict,
                      classes,
                      threshold,
                      top_k=1):
    """ evaluate categorical result for each class with str lable

    # Arguments
        y_true_str: (list)
            eg: ['dog', 'cat', ...]
        y_pred_dict: (list)
            eg: [{'dog':0.8, 'cat':0.1}, {'cat':0.5, 'mouse':0.2}, ...]
        classes: (list)
            eg: ['dog', 'cat', ..]
        threshold: (dict)
            eg: {'dog': 0.8, 'cat':0.5}
            eg: 0.85
        top_k: (int)

    # Returns
        result: (dict)
        eg: {'Total': {'precision':0.8, 'recall':0.5, 'hit':800, 'pick':1000, 'total':2000},
        'dog': {'precision': ... 'total':..}, {'cat': {'precision': ... 'total':..}}}
    """

    assert len(y_true_str) == len(y_pred_dict)

    hit_cnt = [0 for _ in classes]
    pick_cnt = [0 for _ in classes]
    target_cnt = [0 for _ in classes]

    for i, yt_name in enumerate(y_true_str):
        if yt_name not in classes:
            yt_name = 'Other'
        target_cnt[classes.index(yt_name)] += 1

        pred = y_pred_dict[i]
        yp_name, yp_conf = _get_top_k(yt_name, pred, top_k)
        if yp_name not in classes:
            yp_name = 'Other'

        conf_threshold = threshold
        if isinstance(threshold, dict):
            conf_threshold = threshold[yp_name]
        if yp_conf < conf_threshold:
            continue

        pick_cnt[classes.index(yp_name)] += 1

        if yt_name == yp_name:
            hit_cnt[classes.index(yt_name)] += 1

    return to_evaluate_result(hit_cnt, pick_cnt, target_cnt, classes, threshold)


def _categorical_eval_func(args):
    return _categorical_eval(*args)


def _get_conf_list(y_pred_dict):
    conf_set = {float(format(max(pred.values()), '0.2f'))
                for pred in y_pred_dict}
    if not conf_set:
        conf_set = {0.0}
    conf_set.add(max(min(conf_set) - 0.01, 0.0))
    conf_set.add(min(max(conf_set) + 0.01, 1.0))
    conf_list = list(conf_set)
    conf_list.sort()
    return conf_list


def _categorical_eval_pro(y_true_str,
                          y_pred_dict,
                          classes,
                          top_k=1):
    # 1. get each class confidence list
    conf_list = _get_conf_list(y_pred_dict)

    result_list = []

    args_list = [[y_true_str, y_pred_dict, classes, conf, top_k] for conf in conf_list]

    # 2. evaluate result with each confidence
    with concurrent.futures.ProcessPoolExecutor() as executor:
        result_list = executor.map(_categorical_eval_func, args_list)

    return result_list


def _categorical_eval_pro_v2(y_true_str,
                             y_pred_dict,
                             classes,
                             top_k=1):
    # 1. get each class confidence list
    conf_list = _get_conf_list(y_pred_dict)

    hit_cnts = np.zeros((len(classes), len(conf_list)))
    pick_cnts = np.zeros((len(classes), len(conf_list)))
    target_cnt = [0 for _ in classes]

    for i, yt_name in enumerate(y_true_str):
        if yt_name not in classes:
            yt_name = 'Other'
        target_cnt[classes.index(yt_name)] += 1

        pred = y_pred_dict[i]
        yp_name, yp_conf = _get_top_k(yt_name, pred, top_k)
        if not yp_name:
            continue

        conf_index = bisect_right(conf_list, yp_conf)

        if yp_name not in classes:
            yp_name = 'Other'
        pick_cnts[classes.index(yp_name), :conf_index] += 1

        if yt_name == yp_name:
            hit_cnts[classes.index(yt_name), :conf_index] += 1

    result_list = []
    for i, conf in enumerate(conf_list):
        result = to_evaluate_result(hit_cnts[:, i], pick_cnts[:, i], target_cnt, classes, conf)
        result_list.append(result)
    return result_list


def merge_eval_result(classes, result_list,
                      conf_threshold,
                      precision_threshold,
                      recall_threshold):
    merge_result = {}
    need_classes = set(classes)
    precision_sum = {key: 0 for key in classes}
    precision_cnt = {key: 0 for key in classes}

    if not precision_threshold and recall_threshold:
        result_list = result_list.copy()
        result_list.reverse()
    for result in result_list:
        for key in classes:
            if result[key]['precision'] > 0 or result[key]['recall'] > 0:
                precision_sum[key] += result[key]['precision']
                precision_cnt[key] += 1
        for key in need_classes.copy():
            if conf_threshold and result[key]['confidence'] < conf_threshold[key]:
                continue
            if precision_threshold and result[key]['precision'] < precision_threshold[key]:
                continue
            if recall_threshold and result[key]['recall'] < recall_threshold[key]:
                continue
            need_classes.remove(key)
            merge_result[key] = result[key]

    average_precision = {key: safe_div(precision_sum[key], precision_cnt[key]) for key in classes}
    for key in classes:
        if key not in merge_result:
            merge_result[key] = result_list[-1][key]
        merge_result[key]['ap'] = average_precision[key]

    _update_total_result(merge_result)

    merge_result['classes'] = classes

    return merge_result


def _categorical_result_dump(result):
    print('{:18} {:>10} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}'.format(
        'Label', 'Precision', 'Recall', 'AP', 'F1', 'Conf', 'Hit', 'Pick', 'Total', 'Percent'))
    show_labels = result['classes'].copy()
    show_labels.sort()
    if 'Total' in result:
        show_labels.append('Total')
    if 'TotalNoOther' in result:
        show_labels.append('TotalNoOther')
    for label in show_labels:
        _print_one_evaluate_result(label, result[label])


def _plot_categorical_result_table(result, title=None, need_show=True):
    disp_fmt = {'precision': '0.0%',
                'recall': '0.0%',
                'ap': '0.0%',
                'f1': '0.00',
                'confidence': '0.00',
                'percent': '0.0%'}
    show_labels = result['classes'].copy()
    show_labels.sort()
    if 'Total' in result:
        show_labels.append('Total')
    if 'TotalNoOther' in result:
        show_labels.append('TotalNoOther')
    disp_data = OrderedDict()
    disp_cols = ('precision', 'recall', 'ap', 'f1', 'confidence',
                 'hit', 'pick', 'total', 'percent')
    for label in show_labels:
        disp_data.setdefault('Label', [])
        disp_data['Label'].append(label)
        for col in disp_cols:
            disp_data.setdefault(col, [])
            disp_data[col].append(result[label][col])

    return plot_table(columns_data=disp_data, columns_fmt=disp_fmt,
                      title=title, need_show=need_show)


def plot_categorical_curcve(result_list, xkey, ykey, need_show=True):
    xdata = {}
    ydata = {}
    xmin, xmax = None, None
    ymin, ymax = None, None

    for result in result_list:
        show_labels = result['classes'].copy()
        show_labels.sort()
        if 'Total' in result:
            show_labels.append('Total')
        if 'TotalNoOther' in result:
            show_labels.append('TotalNoOther')

        for label in show_labels:
            xdata.setdefault(label, [])
            ydata.setdefault(label, [])

            x = result[label][xkey]
            y = result[label][ykey]

            if y == 0:
                continue

            xdata[label].append(x)
            if xmin is None or x < xmin:
                xmin = x
            if xmax is None or x > xmax:
                xmax = x

            ydata[label].append(y)
            if ymin is None or y < ymin:
                ymin = y
            if ymax is None or y > ymax:
                ymax = y
    xmax = 1.0 if xmax is None else xmax
    xmin = 0.0 if xmin is None else xmin

    xdiff = max(xmax - xmin, 0.5)
    xpad = xdiff * 0.025
    xmax += xpad
    xmin -= xpad

    ymax = 1.0 if ymax is None else ymax
    ymin = 0.0 if ymin is None else ymin
    ydiff = max(ymax - ymin, 0.5)
    ypad = ydiff * 0.025
    ymax += ypad
    ymin -= ypad

    x_range = [xmin, xmax]
    if xkey == 'confidence':
        x_range = [xmax, xmin]

    return plot_lines(title=ykey + '/' + xkey,
                      xdata=xdata,
                      ydata=ydata,
                      x_range=x_range,
                      y_range=[ymin, ymax],
                      xlabel=xkey,
                      ylabel=ykey,
                      need_show=need_show)


def _plot_categorical_confusion_matrix(y_true_str, y_pred_dict, classes, threshold=0.0,
                                       need_show=True):
    num_classes = len(classes)
    conf_matrix = np.zeros((num_classes + 1, num_classes + 2), dtype=int)

    for pred, yt_name in zip(y_pred_dict, y_true_str):
        yp_name, yp_conf = _get_top_k(yt_name, pred, top_k=1)
        if yp_name not in classes:
            yp_name = 'Other'
        if yt_name not in classes:
            yt_name = 'Other'

        conf_threshold = threshold
        if isinstance(threshold, dict):
            conf_threshold = threshold[yp_name]
        if yp_conf < conf_threshold:
            conf_matrix[classes.index(yt_name)][-1] += 1
        else:
            conf_matrix[classes.index(yt_name)][classes.index(yp_name)] += 1

    error_values = conf_matrix[0:-1, 0:-2] * (np.ones((num_classes, num_classes))
                                              - np.eye(num_classes))
    for i in range(num_classes):
        conf_matrix[i][-2] = np.sum(error_values[i, :])
        conf_matrix[-1][i] = np.sum(error_values[:, i])

    x_cnts = conf_matrix[:num_classes, :num_classes].sum(axis=1).reshape(-1, 1)
    x_cnts = x_cnts + conf_matrix[:num_classes, -1].reshape(-1, 1) + 0.0001
    y_cnts = conf_matrix[:num_classes, :num_classes].sum(axis=0) + 0.0001

    x_percent = np.zeros_like(conf_matrix, dtype=np.float32)
    x_percent[:num_classes, :] = conf_matrix[:num_classes, :] / x_cnts

    y_percent = np.zeros_like(conf_matrix, dtype=np.float32)
    y_percent[:, :num_classes] = conf_matrix[:, :num_classes] / y_cnts

    x_percent = (100 * x_percent).flatten().tolist()
    y_percent = (100 * y_percent).flatten().tolist()
    tips = ['R:{:.1f}, P:{:.1f}'.format(x, y) for x, y in zip(x_percent, y_percent)]

    error_color = "#f7b6d2"
    error_subtotal_color = "#c49c94"
    unknown_color = "#777777"
    correct_color = "#9edae5"

    texts = conf_matrix.flatten().tolist()

    x_labels = [classes[x] for x in range(num_classes)]
    x_labels.append('Miss Total')
    x_labels.append('unknown')

    y_labels = [classes[x] for x in range(num_classes)]
    y_labels.append('False Positive')

    colors = np.full(conf_matrix.shape, error_color)
    for i in range(num_classes):
        colors[i, i] = correct_color
    colors[:, -2] = error_subtotal_color
    colors[:, -1] = unknown_color
    colors[-1, :] = error_subtotal_color
    colors = colors.flatten().tolist()

    alphas = np.zeros(conf_matrix.shape)
    correct_values = conf_matrix[:num_classes, :num_classes] * np.eye(num_classes)
    correct_alphas = correct_values / (np.max(correct_values) + 0.001)

    error_alphas = error_values / (np.max(error_values) + 0.001)

    alphas[:num_classes, :num_classes] = correct_alphas + error_alphas

    alphas[:-1, -2] = conf_matrix[:-1, -2] / (np.max(conf_matrix[:-1, -2]) + 0.001)
    alphas[:-1, -1] = conf_matrix[:-1, -1] / (np.max(conf_matrix[:-1, -1]) + 0.001)
    alphas[-1, :-2] = conf_matrix[-1, :-2] / (np.max(conf_matrix[-1, :-2]) + 0.001)
    alphas = alphas.flatten().tolist()

    return plot_bokeh_matrix(title='Confusion Matrix',
                             x_labels=x_labels,
                             y_labels=y_labels,
                             colors=colors,
                             alphas=alphas,
                             texts=texts,
                             tips=tips,
                             need_show=need_show)


def _plot_categorical_confusion_scatter(y_true_str, y_pred_dict, classes,
                                        yaxis='target', target=None, predict=None, need_show=True):
    conf_scatter = []

    def _add_one_point(ypred, conf, ytrue, color_class):
        right = ypred == ytrue
        color = get_one_color(classes.index(color_class))
        alpha = 0.8 * (1 - conf) if right else 0.8 * conf
        alpha = max(0.2, alpha)
        size = 8 * (1 - conf) if right else 8 * conf
        size = max(2, size)
        conf_scatter.append({'conf': conf,
                             'predict': ypred,
                             'target': ytrue,
                             'color': color,
                             'alpha': alpha,
                             'size': size})

    for pred, ytrue in zip(y_pred_dict, y_true_str):
        yp_sort = [{'class': k, 'conf': conf} for k, conf in pred.items()]
        yp_sort.sort(key=lambda x: x['conf'], reverse=True)
        if target is not None:
            yaxis = 'predict'
            if target == ytrue:
                for yp in yp_sort:
                    ypred = yp['class']
                    conf = yp['conf']
                    _add_one_point(ypred, conf, ytrue, ypred)
        elif predict is not None:
            yaxis = 'target'
            for yp in yp_sort:
                ypred = yp['class']
                conf = yp['conf']
                if predict == ypred:
                    _add_one_point(ypred, conf, ytrue, ytrue)
        else:
            top1 = yp_sort[0]
            ypred = top1['class']
            conf = top1['conf']
            _add_one_point(ypred, conf, ytrue, ypred if yaxis == 'target' else ytrue)
    return plot_bokeh_scatter('Prediction Confidence Scatter',
                              x_range=(1.0, 0.0),
                              y_range=classes,
                              x='conf', y=yaxis,
                              x_label='Confidence',
                              y_label=yaxis,
                              data=conf_scatter,
                              need_show=need_show)


def _get_categorical_error_list(y_true_str, y_pred_dict, classes,
                                target_cls=None, predict_cls=None, threshold=0.0):

    error_list = []
    conf_list = []
    error_desc_list = []
    for i, (pred, yt_name) in enumerate(zip(y_pred_dict, y_true_str)):
        if yt_name not in classes:
            yt_name = 'Other'
        if target_cls is not None and target_cls != yt_name:
            continue

        yp_sort = [{'class': k, 'conf': conf}
                   for k, conf in pred.items()]
        yp_sort.sort(key=lambda x: x['conf'], reverse=True)
        pred_top1 = yp_sort[0]
        yp_name, yp_conf = pred_top1['class'], pred_top1['conf']
        if yp_name not in classes:
            yp_name = 'Other'

        if predict_cls is not None and predict_cls != yp_name:
            continue

        conf_threshold = threshold
        if isinstance(threshold, dict):
            conf_threshold = threshold[yp_name]
        if yp_name != yt_name and yp_conf >= conf_threshold:
            error_list.append(i)
            conf_list.append(yp_conf)
            error_desc = ','.join([p['class'] + ':' + format(p['conf'], '.3f') for p in yp_sort[:3]])
            error_desc_list.append(error_desc)
    sort_index = np.argsort(conf_list)
    error_list = [error_list[i] for i in reversed(sort_index)]
    error_desc_list = [error_desc_list[i] for i in reversed(sort_index)]
    return error_list, error_desc_list


class EvalCategory:
    """ evaluate categorical result for each class

    # Arguments
        y_true: (list)
            eg: ['dog', 'cat', ...]
            eg1: [[0, 0, 1], [1, 0, 0] ..]
            eg2: [2, 0, ..]
        y_pred: (list)
            eg: [{'dog':0.8, 'cat':0.1}, {'cat':0.5, 'mouse':0.2}, ...]
            eg1: [[0.0, 0.2, 0.8], [0.8, 0.1, 0.1] ...]
        conf_threshold: (dict)
            eg: {'dog': 0.8, 'cat':0.5}
            eg1: 0.85
        precision_threshold: (dict)
            eg: {'dog': 0.8, 'cat':0.5}
            eg1: 0.85
        recall_threshold: (dict)
            eg: {'dog': 0.8, 'cat':0.5}
            eg1: 0.85
        top_k: (int)
        classes: (list) class list for predict
            eg: ['dog', 'cat', ..]
        true_classes (list) class list for target
            eg: ['dog', 'cat', ..]
        force_other (bool)
    """

    def __init__(self, y_pred, y_true,
                 conf_threshold=0.0,
                 precision_threshold=None,
                 recall_threshold=None,
                 top_k=1, **kwargs):
        y_pred_dict = y_pred
        if 'classes' in kwargs:
            classes = kwargs['classes'].copy()
            if not isinstance(y_pred[0], dict):
                # onehot: [[0.2, 0.5, 0.1] ...]
                y_pred = np.array(y_pred)
                y_pred_dict = [{classes[i]: yi for i, yi in enumerate(y)} for y in y_pred]
        else:
            assert isinstance(y_pred[0], dict)
            classes = list({yi for y in y_pred for yi in y.keys()})

        true_classes = kwargs.get('true_classes', classes).copy()

        y_true_str = y_true
        if isinstance(y_true, (tuple, list)) and isinstance(y_true[0], str):
            if 'true_classes' not in kwargs:
                true_classes = list({y for y in y_true})
        else:
            y_true = np.array(y_true)
            if len(y_true.shape) == 2:
                # onehot: [[0, 1, 0], ...]
                y_true_str = [true_classes[y.argmax()] for y in y_true]
            else:
                # index: [1, 3, ...]
                y_true_str = [true_classes[y] for y in y_true]

        force_other = kwargs.get('force_other', True)

        if force_other and 'Other' not in classes:
            classes.append('Other')

        good_classes = [c for c in classes if c in true_classes]
        if force_other and 'Other' not in good_classes:
            good_classes.append('Other')
        elif not force_other and 'Other' in good_classes:
            good_classes.remove('Other')

        conf_threshold = to_threshold_dict(conf_threshold, classes)

        self._result = None
        self._result_list = None

        self._y_true_str = y_true_str
        self._y_pred_dict = y_pred_dict
        self._good_classes = good_classes
        self._conf_threshold = conf_threshold
        self._top_k = top_k

        self.get_result_list()
        precision_threshold = to_threshold_dict(precision_threshold, classes)
        recall_threshold = to_threshold_dict(recall_threshold, classes)
        self._result = merge_eval_result(good_classes, self._result_list,
                                         conf_threshold,
                                         precision_threshold,
                                         recall_threshold)
        for label in self._result['classes']:
            self._conf_threshold[label] = self._result[label]['confidence']

    def to_pkl(self, pkl_path):
        result = {'y_true_str': self._y_true_str,
                  'y_pred_dict': self._y_pred_dict,
                  'classes': self._good_classes
                  }
        obj_to_pkl(result, pkl_path)

    @classmethod
    def from_pkl(cls, pkl_path, **kwargs):
        result = obj_from_pkl(pkl_path)
        return cls(y_true=result['y_true_str'],
                   y_pred=result['y_pred_dict'],
                   classes=result['classes'],
                   **kwargs)

    def update_threshold(self, **kwargs):
        return EvalCategory(y_true=self._y_true_str,
                            y_pred=self._y_pred_dict,
                            classes=self._good_classes,
                            **kwargs)

    def get_result(self):
        return self._result

    def get_result_list(self):
        if not self._result_list:
            self._result_list = _categorical_eval_pro_v2(self._y_true_str,
                                                         self._y_pred_dict,
                                                         self._good_classes,
                                                         self._top_k)
        return self._result_list

    def dump(self):
        _categorical_result_dump(self.get_result())

    def plot_bokeh_table(self, title=None, need_show=True):
        return _plot_categorical_result_table(self.get_result(), title, need_show)

    def plot_precision_conf(self, need_show=True):
        result_list = self.get_result_list()
        return plot_categorical_curcve(result_list, 'confidence', 'precision', need_show)

    def plot_recall_conf(self, need_show=True):
        result_list = self.get_result_list()
        return plot_categorical_curcve(result_list, 'confidence', 'recall', need_show)

    def plot_f1_conf(self, need_show=True):
        result_list = self.get_result_list()
        return plot_categorical_curcve(result_list, 'confidence', 'f1', need_show)

    def plot_precision_recall(self, need_show=True):
        result_list = self.get_result_list()
        return plot_categorical_curcve(result_list, 'recall', 'precision', need_show)

    def plot_confusion_matrix(self, conf_threshold=None, need_show=True):
        if conf_threshold is None:
            conf_threshold = self._conf_threshold
        else:
            conf_threshold = to_threshold_dict(conf_threshold, self._good_classes)

        return _plot_categorical_confusion_matrix(self._y_true_str, self._y_pred_dict,
                                                  self._good_classes, conf_threshold,
                                                  need_show)

    def plot_bokeh_scatter(self, target=None, predict=None, yaxis='target', need_show=True):
        return _plot_categorical_confusion_scatter(self._y_true_str, self._y_pred_dict,
                                                   self._good_classes, yaxis,
                                                   target, predict, need_show)

    def get_error_images(self, target_cls=None, predict_cls=None, conf_threshold=None):
        """ get error images
        # Arguments
            target_cls: str
                eg: 'dog'

            predict_cls: str
                eg: 'cat'
        """

        if conf_threshold is None:
            conf_threshold = self._conf_threshold
        else:
            conf_threshold = to_threshold_dict(conf_threshold, self._good_classes)

        error_list, error_desc_list = _get_categorical_error_list(self._y_true_str, self._y_pred_dict,
                                                                  self._good_classes, target_cls,
                                                                  predict_cls, conf_threshold)
        return error_list, error_desc_list

    def export_error_images(self, ill, out_dir, target_cls=None, predict_cls=None, conf_threshold=None):
        """ export error images to out dir
        # Arguments
            target_cls: str
                eg: 'dog'

            predict_cls: str
                eg: 'cat'
        """
        import shutil
        os.makedirs(out_dir)
        error_list, error_desc_list = self.get_error_images(target_cls, predict_cls, conf_threshold)
        for idx, desc in zip(error_list, error_desc_list):
            src_path = ill.x[idx]
            dst_name = ill.y[idx] + '_|_' + desc + '_|_' + osp.basename(src_path)
            dst_path = osp.join(out_dir, dst_name)
            shutil.copy(src_path, dst_path)

    def show_error_images(self, ill, model_vis=None, target_cls=None,
                          predict_cls=None, conf_threshold=None, ncols=4,
                          nrows=2, labelset=None, **kwargs):
        '''
        # Arguments
            ill: ImageLabelList
            model_vis: CategoryModelVis for show heatmap
            target_cls: str
                eg: 'dog'

            predict_cls: str
                eg: 'cat'
        '''
        error_list, error_desc_list = self.get_error_images(
            target_cls, predict_cls, conf_threshold)
        if len(error_list) == 0:
            print('error_list len:', len(error_list))
            return None
        if model_vis:
            return model_vis.show(ill, idxs=error_list, desc_list=error_desc_list, **kwargs)
        return ImageCleaner(ill, find_idxs=error_list, desc_list=error_desc_list,
                            ncols=ncols, nrows=nrows, labelset=labelset, **kwargs)

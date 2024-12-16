'''
Copyright (C) 2023 TuringVision

Classification task experiment comparison tool
'''
import os.path as osp
import os
import yaml
import csv
from collections import OrderedDict
from .eval_category import EvalCategory
from ..ui import plot_table, plot_lines

__all__ = ['CategoryModelInfo', 'CategoryExperiment']

class CategoryModelInfo:
    def __init__(self, model_dir, evac_name='evaluate.pkl'):
        ''' Get category model info from model directory.
        '''
        self._model_dir = model_dir
        self._evac_name = evac_name
        self._name = osp.basename(model_dir)
        try:
            self._info = self._load_info(model_dir)
            self._evac = self._load_evac(model_dir)
            self._history = self._load_history(model_dir)
        except (NotADirectoryError, FileNotFoundError):
            self._info = None
            self._evac = None
            self._history = None

    def _load_info(self, model_dir):
        model_info_path = osp.join(model_dir, 'model_info.yml')
        if not osp.isfile(model_info_path):
            model_info_path = osp.join(model_dir, 'vision/model_info.yml')
        with open(model_info_path, 'rt', encoding='utf-8') as fp:
            return yaml.load(fp, Loader=yaml.FullLoader)

    def _load_evac(self, model_dir):
        return EvalCategory.from_pkl(osp.join(model_dir, self._evac_name))

    def _load_history(self, model_dir):
        type_table = {'epoch': int,
                      'train_loss': float,
                      'valid_loss': float,
                      'accuracy': float}
        history_path = osp.join(model_dir, 'history.csv')
        if not osp.isfile(history_path):
            history_path = osp.join(model_dir, 'vision/history.csv')
        with open(history_path, 'rt', encoding='utf-8') as fp:
            reader = csv.reader(fp)
            rows = list(reader)
        history = {}
        keys = list()
        for row in rows:
            if row[0] == 'epoch':
                if not keys:
                    keys = row
                    for item in row:
                        history.setdefault(item, [])
            else:
                for i, value in enumerate(row):
                    item = keys[i]
                    dtype = type_table.get(item, str)
                    history[item].append(dtype(value))
        return history

    def name(self):
        ''' get model name
        '''
        return self._name

    def info(self):
        ''' get model training info
        '''
        return self._info

    def evac(self):
        ''' get model evaluate result
        '''
        return self._evac

    def history(self):
        ''' get model training history
        '''
        return self._history


class CategoryExperiment:
    def __init__(self, exp_dir, evac_name='evaluate.pkl'):
        ''' Classification task experiment comparison tool

        exp_dir: An experimental directory containing many model directories
        '''
        self._exp_dir = exp_dir
        self._evac_name = evac_name
        self._model_info_list = self.get_model_info_list(exp_dir)
        self._model_name_list = [info.name() for info in self._model_info_list]

    def get_model_info_list(self, exp_dir):
        ''' get info for all the models
        '''
        from fastprogress.fastprogress import progress_bar
        model_info_list = list()
        for model_dir in progress_bar(os.listdir(exp_dir)):
            model_path = osp.join(exp_dir, model_dir)
            model_info = CategoryModelInfo(model_path, self._evac_name)
            if model_info.info() and model_info.evac():
                model_info_list.append(model_info)
        return model_info_list

    def show(self, need_show=True):
        ''' show info for all the models
        '''
        disp_data = OrderedDict()
        for info in self:
            disp_data.setdefault('Name', [])
            disp_data['Name'].append(info.name())
            disp_data.setdefault('Date', [])
            model_info = info.info()
            disp_data['Date'].append(model_info['date'])
            disp_data.setdefault('BaseModel', [])
            train_schedule = model_info['train_schedule']
            if isinstance(train_schedule, (tuple, list)):
                train_schedule = train_schedule[0]
            disp_data['BaseModel'].append(train_schedule['basemodel'])
            disp_data.setdefault('BS', [])
            disp_data['BS'].append(train_schedule['bs'])
            disp_data.setdefault('MixupRatio', [])
            disp_data['MixupRatio'].append(train_schedule.get('mixup_ratio', 0.0))
            disp_data.setdefault('Workers', [])
            disp_data['Workers'].append(train_schedule['num_workers'])
            evac_ressult = info.evac().get_result()['Total']
            disp_data.setdefault('Precision', [])
            disp_data['Precision'].append(evac_ressult['precision'])
            disp_data.setdefault('Recall', [])
            disp_data['Recall'].append(evac_ressult['recall'])
            disp_data.setdefault('Conf', [])
            disp_data['Conf'].append(evac_ressult['confidence'])
            disp_data.setdefault('Hit', [])
            disp_data['Hit'].append(evac_ressult['hit'])
            disp_data.setdefault('Pick', [])
            disp_data['Pick'].append(evac_ressult['pick'])
            disp_data.setdefault('Total', [])
            disp_data['Total'].append(evac_ressult['total'])
        disp_fmt = {'Precision': '0.0%',
                    'Recall': '0.0%',
                    'Conf': '0.00'}
        return plot_table(columns_data=disp_data, columns_fmt=disp_fmt, need_show=need_show)

    def show_evac(self, item='precision', need_show=True):
        ''' show evaluate result for all the models

        In:
            item: (str) class name, eg: 'Cat', 'Dog', ...
            or score name eg: 'precision', 'recall', ...
        '''
        disp_fmt = {'precision': '0.0%',
                    'recall': '0.0%',
                    'f1': '0.00',
                    'confidence': '0.00',
                    'percent': '0.0%'}
        evac_disp_fmt = {}
        disp_data = OrderedDict()

        for info in self:
            disp_data.setdefault('Name', [])
            disp_data['Name'].append(info.name())
            evac_result = info.evac().get_result()
            if item in evac_result.keys():
                evac_disp_fmt = disp_fmt
                code_result = evac_result[item]
                for key in code_result.keys():
                    disp_data.setdefault(key, [])
                    disp_data[key].append(code_result[key])
            else:
                for code in evac_result.keys():
                    if code == 'classes':
                        continue
                    if code not in evac_disp_fmt:
                        evac_disp_fmt[code] = disp_fmt[item]
                    disp_data.setdefault(code, [])
                    disp_data[code].append(evac_result[code][item])
        return plot_table(columns_data=disp_data, columns_fmt=evac_disp_fmt,
                          need_show=need_show)

    def show_history(self, item='accuracy', idxs=None, smooth_wl=11, smooth_po=5, need_show=True):
        ''' show training history info fro all the models

        In:
            item: (str) history item name.
            idxs: list of model index
        '''
        from scipy.signal import savgol_filter

        xdata = {}
        ydata = {}
        smooth_po = min(smooth_wl-1, smooth_po)
        if idxs is None:
            idxs = list(range(len(self)))
        for index in idxs:
            info = self[index]
            history_info = info.history()[item]
            xdata[info.name()] = list(range(len(history_info)))
            ydata[info.name()] = savgol_filter(history_info, smooth_wl, smooth_po)
        return plot_lines(title=item, xdata=xdata, ydata=ydata,
                          xlabel='epoch', ylabel=item,
                          need_show=need_show)

    def evac(self, index):
        return self[index].evac()

    def plot_bokeh_table(self, index, **kwargs):
        ''' show bokeh table
        '''
        evac = self.evac(index)
        evac.plot_bokeh_table(**kwargs)

    def plot_confusion_matrix(self, index, **kwargs):
        ''' show confusion matrix
        '''
        evac = self.evac(index)
        evac.plot_confusion_matrix(**kwargs)

    def __len__(self):
        return len(self._model_info_list)

    def __getitem__(self, index):
        if isinstance(index, str):
            index = self._model_name_list.index(index)
        return self._model_info_list[index]

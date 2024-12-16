'''
Copyright (C) 2023 TuringVision

Image category interface for model training and inference
'''

import os.path as osp
import PIL, yaml, time, os
import numpy as np
from zipfile import ZipFile, ZIP_DEFLATED
from uuid import getnode as get_mac
from .eval_category import EvalCategory
from .image_data import ImageLabelList
from ..ui import bokeh_figs_to_html
from ..utils import package_capp, unpackage_capp


__all__ = ['BasicCategoryTrain', 'BasicCategoryInference',
           'FastCategoryTrain', 'FastCategoryInference',
           'get_learner_preds']

_MODEL_INFO_FILE = 'model_info.yml'


def get_learner_preds(learner, activ=None, callback_pbar = None):
    '''get fastai learner valid data predict result
    In :
        activ: activ function
        callback_pbar: parent callback bar, fastprogress.master_bar
    Out:
        res (list): list of predict result
            single category task: [[0.0, 0.2, 0.8], [0.8, 0.1, 0.1] ...]
            multi category task: [[[0.0, 0.2, 0.8], ...], # for task a
                                  [[0.2, 0.3, 0.9], ...] # for task b
                                 ]
        y_true (list): groud truth of valid data
            single category task: ['A', 'B', 'C' ...]
            multi category task: [['A', 'B', 'C', ...], ['M', 'N', ..]]
    '''
    import torch
    import torch.nn.functional as F
    from fastai.vision import validate, CallbackHandler
    from functools import partial

    activ = activ if activ else partial(F.softmax, dim=-1)
    cb_handler = CallbackHandler(learner.callbacks)
    out = validate(learner.model, learner.data.valid_dl, cb_handler=cb_handler,
                   pbar=callback_pbar, average=False, n_batch=None)
    y_pred = [pred for pred, _ in out]
    if isinstance(y_pred[0], (tuple, list)):
        y_pred = [o for o in zip(*y_pred)]
        res = [torch.cat(o).cpu() for o in y_pred]
        res = [activ(r) for r in res]
        y_true = None
        if isinstance(learner.data.valid_ds.y[0], (tuple, list)):
            y_true = [o for o in zip(*learner.data.valid_ds.y)]
    else:
        res = torch.cat(y_pred).cpu()
        res = activ(res)
        y_true = [o.data for o in learner.data.valid_ds.y]
    return (res, y_true)


class FastCategoryInference:
    ''' fastai category model inference
    '''
    def __init__(self, model_path, work_dir=None, learner=None):
        '''
        model_path: capp model path
        work_dir:
        learner: fastai learner
        '''
        self._model_dir = unpackage_capp(model_path)
        self._work_dir = self._model_dir if work_dir is None else work_dir
        self._model_info = None
        self._tfms = []
        self._learner = learner
        self._post_init()

    def _post_init(self):
        ''' for update tfms
        '''
        pass

    def _load_learner(self, data):
        from fastai.vision import cnn_learner, models, CrossEntropyFlat

        if self._learner is None:
            # 1. load model info
            model_info = self.model_info()

            # 2. create learner (need basemodel)
            basemodel = model_info['train_schedule']['basemodel']
            bn_final = model_info['train_schedule'].get('bn_final', False)

            learner = cnn_learner(data, eval('models.'+basemodel), train_bn=False, bn_final=bn_final)
            learner = learner.load(open(model_info['model_path'], 'rb'))
            loss_func = CrossEntropyFlat()
            learner.loss_func = loss_func
            self._learner = learner
        else:
            self._learner.data = data
        return self._learner

    def load_learner(self, ill, tfms=None, bs=None, num_workers=None):
        ''' load fastai learner
        ill: ImageLabelList
        tfms: transform function list, see ImageLabelList.add_tfm
        bs: batch_size, default from model_info
        num_workers: default from model_info
        '''
        model_info = self.model_info()

        _, valid = ill.split(show=False)

        if bs is None:
            bs = model_info['train_schedule']['bs']
            bs = min(len(valid), bs)
        if num_workers is None:
            num_workers = model_info['train_schedule']['num_workers']
        tfms = self._tfms if tfms is None else tfms
        # prepare databunch (need bs, num_workers, preproc_func)
        data = ill.databunch(train_tfms=tfms, valid_tfms=tfms,
                             path=self._work_dir,
                             bs=bs, num_workers=num_workers,
                             show_dist=False)
        data.classes = self.get_class_list()
        data.c = len(data.classes)
        if isinstance(data.classes[0], (tuple, list)):
            data.c = [len(cls) for cls in data.classes]

        return self._load_learner(data)

    def model_info(self):
        ''' load model info
        '''
        if self._model_info is None:
            with open(osp.join(self._model_dir, _MODEL_INFO_FILE), 'rt', encoding='utf-8') as fp:
                self._model_info = yaml.load(fp, Loader=yaml.FullLoader)
                if isinstance(self._model_info['train_schedule'], (tuple, list)):
                    self._model_info['train_schedule'] = self._model_info['train_schedule'][0]
                self._model_info['model_path'] = osp.join(self._model_dir, 'model.pth')
        return self._model_info

    def get_class_list(self):
        '''get model class list
        '''
        model_info = self.model_info()
        return model_info['classes']

    def predict(self, ill, tfms=None, bs=None, num_workers=None, activ=None, callback_pbar = None):
        ''' predict ImageLabelList valid data
        In:
            ill: ImageLabelList
            tfms: transform function list, see ImageLabelList.add_tfm
            bs: batch_size, default from model_info
            num_workers: default from model_info
            activ: activ function
            callback_pbar: parent callback bar, fastprogress.master_bar
        Out:
            preds (list): list of predict result
            single category task: [[0.0, 0.2, 0.8], [0.8, 0.1, 0.1] ...]
            single category task sigmoid output: [0.9, 0.2, ...]
            multi category task: [[[0.0, 0.2, 0.8], ...], # for task a
                                  [[0.2, 0.3, 0.9], ...] # for task b
                                 ]
            multi category task sigmoid output: [[0.9, 0.2, ...], # for task a
                                                 [0.8, 0.5, ...], # for task b
                                                 ]
        '''
        learner = self.load_learner(ill, tfms, bs, num_workers)
        use_sigmoid = self.model_info()['train_schedule'].get('sigmoid', False)
        if use_sigmoid:
            import torch
            activ = torch.sigmoid
        preds, _ = get_learner_preds(learner, activ=activ, callback_pbar = callback_pbar)
        return preds

    def get_actns(self, ill, layer_ls=[-1, -2], tfms=None, bs=None, num_workers=None):
        ''' get ImageLabelList valid data features
        In:
            ill: ImageLabelList
            layer_ls: layer index of model
            tfms: transform function list, see ImageLabelList.add_tfm
            bs: batch_size, default from model_info
            num_workers: default from model_info
        Out:
            features (Tensor): NxM, N is valid data number, M is feature size
        '''
        import torch
        from fastprogress.fastprogress import progress_bar
        from fastai.callbacks.hooks import hook_output
        learner = self.load_learner(ill, tfms, bs, num_workers)
        model = learner.model
        hook_module = learner.model
        for l in layer_ls:
            hook_module = hook_module[l]

        with hook_output(hook_module) as hook:
            model.eval()
            dl = learner.data.valid_dl
            actns = []
            with torch.no_grad():
                for (xb, _) in progress_bar(dl):
                    model(xb)
                    actns.append((hook.stored).cpu())
            return torch.cat(actns).view(len(dl.x), -1)

    def evaluate(self, ill, tfms=None, bs=None, num_workers=None):
        ''' get ImageLabelList valid data evaluate result
        In:
            ill: ImageLabelList
            tfms: transform function list, see ImageLabelList.add_tfm
            bs: batch_size, default from model_info
            num_workers: default from model_info
        Out:
            evac:
                single category task: EvalCategory
                multi category task: list of EvalCategory
        '''
        if self.model_info()['train_schedule'].get('sigmoid', False):
            raise NotImplementedError('current sigmoid output not support evaluete!')
        _, valid = ill.split(show=False)
        preds = self.predict(ill, tfms=tfms, bs=bs, num_workers=num_workers)
        if isinstance(preds, (tuple, list)):
            # for multi-task category
            y_target = [o for o in zip(*valid.y)]
            evac = [EvalCategory(y_pred=y_p, y_true=y_t, classes=classes)
                    for y_p, y_t, classes in zip(preds, y_target, self.get_class_list())]
        else:
            evac = EvalCategory(y_pred=preds.numpy(), y_true=list(valid.y),
                                classes=self.get_class_list())
        return evac


def get_inference_status_cb(cb, n_batches, desc):
    ''' get inference status for adc system
    '''
    from fastai.callback import Callback

    class StatusUpdateCallback(Callback):
        def __init__(self, cb, n_batches, desc):
            self._desc = desc
            self._cb = cb
            self._n_batches = n_batches

        def on_batch_end(self, **kwargs):
            citer = kwargs['iteration']
            percent = int(100*(citer + 1) / self._n_batches)
            status = {
                'desc': self._desc,
                'percent': percent,
            }
            if self._cb is not None:
                self._cb(status)

    return StatusUpdateCallback(cb, n_batches, desc)


def get_training_status_cb(cb, n_batches, step_desc, fix_layer):
    ''' get training status for adc system
    '''
    from fastai.callback import Callback

    class StatusUpdateCallback(Callback):
        def __init__(self, cb, n_batches, step_desc, fix_layer):
            self._cb = cb
            self._n_batches = n_batches
            self._step_desc = step_desc
            self._fix_layer = fix_layer
            self._last_acc = 0.0
            self._last_valacc = 0.0
            self._last_valloss = 0.0

        def on_step_end(self, **kwargs):
            percent = int(100*(kwargs['iteration'] + 1) / (self._n_batches*kwargs['n_epochs']))
            status = {
                'desc': 'training'+step_desc+'epoch'+str(kwargs['epoch']),
                'percent': percent
            }
            if self._cb is not None:
                self._cb(status)

        def on_epoch_end(self, **kwargs):
            if 'last_metrics' in kwargs:
                metrics = kwargs['last_metrics']
                self._last_valloss = float(metrics[0])
                self._last_valacc = 0.0
                if len(metrics) >= 2:
                    self._last_valacc = float(metrics[1])
                self._last_acc = self._last_valacc
            epoch = kwargs.get('epoch')
            epochs = kwargs.get('n_epochs')
            percent = int(100*(epoch + 1) / epochs)
            status = {
                'desc': 'training'+step_desc+'epoch'+str(kwargs['epoch']),
                'percent': percent,
                'epoch': epoch,
                'epochs': epochs,
                'pass_desc': self._step_desc,
                'acc': self._last_acc,
                'val_acc': self._last_valacc,
                'loss': float(kwargs['smooth_loss']),
                'val_loss': self._last_valloss,
                'fix_layer': self._fix_layer
            }
            if self._cb is not None:
                need_stop = self._cb(status)
                if need_stop:
                    print('########## REQUEST STOP TRAINING ##########')
                    return {"stop_training": True}
    return StatusUpdateCallback(cb, n_batches, step_desc, fix_layer)


class FastCategoryTrain:
    ''' fastai category model training
    '''
    SUPPORT_SCHEDULE_KEYS = ['basemodel', 'bs', 'num_workers', 'mixup_ratio',
                             'optimizer', 'monitor', 'wd', 'class_weight',
                             'bn_final', 'steps', 'batch_n_cls']
    SUPPORT_STEPS_KEYS = ['epochs', 'lr', 'freeze_layer']
    def __init__(self, work_dir):
        '''
        work_dir:
        '''
        self._work_dir = work_dir
        self._data = None
        self._learner = None
        self._ill = None
        self._train_schedule = None

    def _check_train_schedule(self, train_schedule):
        for key in train_schedule.keys():
            if key not in self.SUPPORT_SCHEDULE_KEYS:
                print('unsupport:', key)
                print('SUPPORT_SCHEDULE_KEYS:', self.SUPPORT_SCHEDULE_KEYS)
                raise KeyError(key)

        for step in train_schedule['steps']:
            for key in step.keys():
                if key not in self.SUPPORT_STEPS_KEYS:
                    print('unsupport:', key)
                    print('SUPPORT_STEPS_KEYS:', self.SUPPORT_STEPS_KEYS)
                    raise KeyError(key)

    def train_from_learner(self, ill, learner, train_schedule, callback=None,
                           resume_from=None, learner_cbs=None):
        '''
        ill: ImageLabelList
        learner: learner from fastai
        train_schedule:
            {'mixup_ratio': 0.2,
             'monitor': 'valid_loss', # one of ['valid_loss', 'accuracy']
             'steps': [{'epochs': 20, 'lr': 0.01, 'freeze_layer': 2},
                        ...]
            }
        '''
        from fastai.callbacks import SaveModelCallback, CSVLogger
        self._check_train_schedule(train_schedule)

        self._ill = ill
        self._learner = learner
        self._data = learner.data

        if resume_from is not None:
            learner = learner.load(resume_from)
        if 'mixup_ratio' in train_schedule:
            learner = learner.mixup(train_schedule['mixup_ratio'])

        last_schedule = self._train_schedule
        if last_schedule is None or resume_from is None:
            self._train_schedule = train_schedule

        monitor = train_schedule.get('monitor', 'valid_loss')
        save_model = SaveModelCallback(learner, monitor=monitor)
        csv_logger = CSVLogger(learner, filename='history', append=True)

        for i, step in enumerate(train_schedule['steps']):
            lr = step.get('lr', 0.001)
            epochs = step.get('epochs', 10)
            step_desc = 'step_' + str(i)
            fix_layer = 'group'
            if 'freeze_layer' in step:
                fix_layer = 'group' + str(step['freeze_layer'])
                learner.freeze_to(step['freeze_layer'])
            else:
                learner.freeze()
            status_cb = get_training_status_cb(callback, len(learner.data.train_dl),
                                               step_desc, fix_layer)
            cb_list = [save_model, csv_logger, status_cb]
            if learner_cbs:
                cb_list.extend([cb(learner) for cb in learner_cbs])
            learner.fit(epochs, lr, callbacks=cb_list)

        if resume_from is not None and last_schedule is not None:
            if isinstance(self._train_schedule, (tuple, list)):
                self._train_schedule.append(train_schedule)
            else:
                self._train_schedule = [self._train_schedule, train_schedule]

    def train(self, ill, train_schedule, train_tfms, valid_tfms,
              callback=None, resume_from=None, learner_cbs=None):
        '''
        ill: ImageLabelList
        train_tfms: list of transform function for train dataset
        valid_tfms: list of transform function for valid dataset
        train_schedule:
            {'basemodel': 'densenet161','bs': 16, 'num_workers':1, 'mixup_ratio': 0.2,
             'optimizer': 'adam', # one of ['adam', 'rms', 'sgd']
             'monitor': 'valid_loss', # one of ['valid_loss', 'accuracy']
             'wd': 1e-2, 'bn_final': False,
             'class_weight': {'Particle': 10, 'Damage': 20},
             'batch_n_cls': 8,
             'steps': [{
                    'epochs': 20, 'lr': 0.01,
                    'freeze_layer': 2
                }, ...]
            }
        '''
        import torch
        from torch import optim
        from fastai.vision import (cnn_learner, CrossEntropyFlat, models, accuracy)
        from functools import partial

        self._check_train_schedule(train_schedule)

        bs = train_schedule['bs']
        batch_n_cls = train_schedule.get('batch_n_cls', None)
        batch_sampler = None
        if batch_n_cls:
            from .batch_sampler import PairBatchSampler

            n_img = max(1, bs//batch_n_cls)
            n_cls = bs // n_img
            batch_sampler = partial(PairBatchSampler, n_img=n_img, n_cls=n_cls)

        data = ill.databunch(train_tfms=train_tfms,
                             valid_tfms=valid_tfms,
                             path=self._work_dir,
                             bs=bs,
                             num_workers=train_schedule['num_workers'],
                             batch_sampler=batch_sampler,
                             show_dist=False)

        opt = train_schedule.get('optimizer', 'adam')
        mom = 0.9
        alpha = 0.99
        eps = 1e-6

        if opt == 'rms':
            opt_func = partial(optim.RMSprop, alpha=alpha, eps=eps)
        elif opt == 'sgd':
            opt_func = partial(optim.SGD, momentum=mom)
        else:
            opt_func = partial(optim.Adam, betas=(mom, alpha), eps=eps)

        wd = train_schedule.get('wd', 1e-2)
        bn_final = train_schedule.get('bn_final', False)

        learner = cnn_learner(data, eval('models.'+train_schedule['basemodel']),
                              opt_func=opt_func, metrics=[accuracy],
                              train_bn=True, bn_final=bn_final, wd=wd)

        weight = torch.ones(len(ill.labelset()), dtype=torch.float32).to('cuda')
        if 'class_weight' in train_schedule:
            for key, value in train_schedule['class_weight'].items():
                weight[data.classes.index(key)] = value
        loss_func = CrossEntropyFlat(weight=weight)
        learner.loss_func = loss_func

        self.train_from_learner(ill, learner, train_schedule, callback, resume_from, learner_cbs)

    def evaluate(self, result_path, callback=None):
        ''' generate valid dataset evaluate index.html
        result_path: dir for save evaluate result.
        '''
        if self._train_schedule.get('sigmoid', False):
            raise NotImplementedError('current sigmoid output not support evaluete!')
        status_cb = get_inference_status_cb(callback, len(self._data.valid_dl), 'evaluate')
        learner = self._learner.load('bestmodel')
        learner.callbacks = [status_cb]
        preds, target = get_learner_preds(learner)
        learner.callbacks = []
        train, valid = self._ill.split()
        if isinstance(preds, (tuple, list)):
            # for multi-task category
            main_label_idx = self._ill.main_label_idx
            evac_list = [EvalCategory(y_pred=y_p, y_true=y_t, classes=classes)
                         for y_p, y_t, classes in zip(preds, target, learner.data.classes)]
            fig_list = list()
            for i, evac in enumerate(evac_list):
                self._ill.set_main_label_idx(i)
                train.set_main_label_idx(i)
                valid.set_main_label_idx(i)
                other_i = (i+1) % len(evac_list)

                def get_other_y(x, y):
                    return y[other_i]
                fig_list += [
                    self._ill.show_dist(get_other_y, need_show=False),
                    self._ill.show_split(train, valid, need_show=False),
                    evac.plot_precision_conf(need_show=False),
                    evac.plot_recall_conf(need_show=False),
                    evac.plot_f1_conf(need_show=False),
                    evac.plot_precision_recall(need_show=False),
                    evac.plot_confusion_matrix(need_show=False),
                    evac.plot_bokeh_scatter(yaxis='target', need_show=False),
                    evac.plot_bokeh_scatter(yaxis='predict', need_show=False),
                    evac.plot_bokeh_table(need_show=False)
                ]
                evac.to_pkl(osp.join(result_path, 'evaluate_task_{}.pkl'.format(i)))
            bokeh_figs_to_html(fig_list, html_path=osp.join(result_path, 'index.html'),
                               title='Evaluate Catgory')
            self._ill.set_main_label_idx(main_label_idx)
            evac = evac_list[main_label_idx]
            total = evac.get_result()['Total']
            return total['precision'], total['recall'], total['f1']

        evac = EvalCategory(y_pred=preds.numpy(), y_true=target, classes=self._data.classes)
        fig_list = [
            self._ill.show_dist(need_show=False),
            self._ill.show_split(train, valid, need_show=False),
            evac.plot_precision_conf(need_show=False),
            evac.plot_recall_conf(need_show=False),
            evac.plot_f1_conf(need_show=False),
            evac.plot_precision_recall(need_show=False),
            evac.plot_confusion_matrix(need_show=False),
            evac.plot_bokeh_scatter(yaxis='target', need_show=False),
            evac.plot_bokeh_scatter(yaxis='predict', need_show=False),
            evac.plot_bokeh_table(need_show=False)
        ]
        bokeh_figs_to_html(fig_list, html_path=osp.join(result_path, 'index.html'),
                           title='Evaluate Catgory')
        evac.to_pkl(osp.join(result_path, 'evaluate.pkl'))
        total = evac.get_result()['Total']
        return total['precision'], total['recall'], total['f1']

    def package_model(self, model_path, import_cmd=None, **kwargs):
        ''' package capp model
        model_path: capp model path
        import_cmd: import cmd for model inference (used by adc gpu server)
        '''
        ext_info = {'classes': self._data.classes,
                    'train_schedule': self._train_schedule}
        ext_info.update(kwargs)

        pkg_files = {'history.csv': osp.join(self._data.path, 'history.csv'),
                     'model.pth': osp.join(self._data.path, 'models/bestmodel.pth')}
        package_capp(model_path, import_cmd,
                     ext_info=ext_info, pkg_files=pkg_files,
                     export_model_info=True)


def _resize_to_256(x, label):
    x = np.array(PIL.Image.fromarray(x).resize((256, 256)))
    return x, label


class BasicCategoryInference(FastCategoryInference):
    def _post_init(self):
        self._tfms = [_resize_to_256]

    def run(self, image_list):
        ill = ImageLabelList(image_list, None)
        _, valid = ill.split(valid_pct=1.0, show=False)

        results = self.predict(ill)

        class_list = self.get_class_list()
        result_dict = {}
        for i, img_path in enumerate(valid.x):
            code_list = [{'code': class_list[index], 'conf': conf}
                         for index, conf in enumerate(results[i])]
            code_list.sort(key=lambda x: x['conf'], reverse=True)
            top3_code_list = code_list[0:3]
            top3_codes = ';'.join(x['code'] for x in top3_code_list)
            top3_confs = ';'.join(format(x['conf'], '0.4f') for x in top3_code_list)
            boxes = ['0.0, 0.0, 1.0, 1.0']
            result_dict[img_path] = {'code': top3_codes, 'status': 'OK',
                                     'conf': top3_confs, 'boxes': boxes}
        return result_dict


def update_status(cb, desc, percent):
    if cb is not None:
        cb({'desc': desc, 'percent': percent})

class BasicCategoryTrain:
    def __init__(self, work_dir, need_stop_cb=None):
        self._work_dir = work_dir
        self._need_stop_cb = need_stop_cb
        self._ill = None
        self._model = None

    def train(self, ill, train_schedule, callback=None, resume_from=None):
        self._ill = ill
        self._model = FastCategoryTrain(self._work_dir)

        tfms = [_resize_to_256]
        def check_stop_cb(status):
            if callback:
                callback(status)
            if self._need_stop_cb:
                return self._need_stop_cb()
            return False
        self._model.train(self._ill, train_schedule, tfms, tfms,
                          check_stop_cb, resume_from)

    def run(self, data_path, training_info, callback=None):
        # 1. prepare dataset
        ill = ImageLabelList.from_label_info(data_path, training_info)

        percent_cb = lambda x: update_status(callback, 'check_img', x)
        ill = ill.filter_invalid_img(check_img_data=True, percent_cb=percent_cb)
        ill = ill.shuffle()
        _, _ = ill.split(valid_pct=0.2, show=False)

        # 2. start training
        if 'train_schedule' in training_info:
            train_schedule = training_info['train_schedule']
        else:
            train_schedule = {
                'basemodel': 'resnet50', 'bs': 16, 'num_workers': 1,
                'steps': [
                    {'epochs': 2, 'lr': 0.001},
                    {'epochs': 2, 'lr': 0.0001},
                ]
            }
        self.train(ill, train_schedule, callback=callback)

    def package_model(self, model_path, callback=None):
        import_cmd = 'from tvlab import BasicCategoryInference'
        update_status(callback, 'package_model', 0)
        self._model.package_model(model_path, import_cmd)
        update_status(callback, 'package_model', 100)

    def evaluate(self, result_path, callback=None):
        return self._model.evaluate(result_path, callback)

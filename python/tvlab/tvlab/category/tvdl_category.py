'''
Copyright (C) 2023 TuringVision
'''

import numpy as np
import os.path as osp

import yaml

from .eval_category import EvalCategory
from ..ui import bokeh_figs_to_html

from ..utils import package_capp, unpackage_capp

__all__ = ['TvdlTrain', 'TvdlInference', 'TvdlCategoryTrain', 'TvdlCategoryInference']


def get_inference_status_cb(epochs, status_cb, monitor):
    import pytorch_lightning as pl

    class LitProgressBar(pl.Callback):

        def __init__(self, epochs, status_cb=None, monitor="val_acc"):
            super().__init__()  # don't forget this :)
            self.epochs = epochs
            self.epoch_count = 0
            self.enable = True
            self.percent = 0
            self.last_train_acc = 0.
            self.last_train_loss = 0.
            self.status_cb = status_cb
            self._train_batch_idx = 0
            self._trainer = None
            self.monitor = monitor

        @property
        def trainer(self):
            return self._trainer

        @property
        def train_batch_idx(self) -> int:
            """
            The current batch index being processed during training.
            Use this to update your progress bar.
            """
            return self._train_batch_idx

        @property
        def total_train_batches(self) -> int:
            """
            The total number of training batches during training, which may change from epoch to epoch.
            Use this to set the total number of iterations in the progress bar. Can return ``inf`` if the
            training dataloader is of infinite size.
            """
            return self.trainer.num_training_batches

        def on_init_end(self, trainer):
            self._trainer = trainer

        def on_train_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
            self._train_batch_idx = trainer.batch_idx

        def on_train_epoch_start(self, trainer, pl_module):
            self._train_batch_idx = 0

        def on_validation_epoch_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
            super().on_validation_epoch_end(trainer, pl_module)
            if "loss" in trainer.callback_metrics:
                status = {
                    'desc': f'trainingstep_0epoch{self.epoch_count}',
                    'percent': int(self.percent),
                    'epoch': self.epoch_count,
                    'epochs': self.epochs,
                    'pass_desc': 'step_0',
                    monitor: trainer.callback_metrics[monitor].item(),
                    'loss': trainer.callback_metrics["loss"].item(),
                    'fix_layer': 'group'
                }
                if "val_loss" in trainer.callback_metrics:
                    status.update({"val_loss": trainer.callback_metrics["val_loss"].item()})
                else:
                    status.update({"val_loss": trainer.callback_metrics["loss"].item()})

                if self.status_cb is not None:
                    should_stop = self.status_cb(status)
                    if should_stop:
                        self._run_early_stopping(self._trainer)
                self.epoch_count += 1

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
            super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx,
                                       dataloader_idx)  # don't forget this :)

            self._train_batch_idx += 1

            dis = 1 / self.epochs
            self.percent = (self.epoch_count / self.epochs + (
                    self.train_batch_idx * dis / self.total_train_batches)) * 100

            status = {
                'desc': f'trainingstep_0epoch{self.epoch_count}',
                'percent': int(self.percent)
            }
            if self.status_cb is not None:
                should_stop = self.status_cb(status)
                if should_stop:
                    self._run_early_stopping(self._trainer)

        def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx,
                                    ) -> None:
            super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx,
                                            dataloader_idx)  # don't forget this :)

            status = {
                'desc': f'validationstep_0epoch{self.epoch_count}',
                'percent': int(self.percent)
            }
            if self.status_cb is not None:
                should_stop = self.status_cb(status)
                if should_stop:
                    self._run_early_stopping(self._trainer)

        def _run_early_stopping(self, trainer):
            if trainer.fast_dev_run:  # disable early_stopping with fast_dev_run
                return
            # stop every ddp process if any world process decides to stop
            try:
                # for old version pytorch-lighting < 1.2
                should_stop = trainer.training_type_plugin.reduce_early_stopping_decision(True)
            except Exception:
                should_stop = trainer.training_type_plugin.reduce_boolean_decision(True)
            trainer.should_stop = trainer.should_stop or should_stop

    return LitProgressBar(epochs, status_cb, monitor)


class TvdlTrain:
    ''' Tvdl model training
    '''
    SUPPORT_SCHEDULE_KEYS = ['backbone', 'lr', 'bs', 'num_workers', 'epochs', 'monitor',
                             'gpus', 'check_per_epoch', 'img_c']

    def __init__(self, work_dir):
        '''
        work_dir: work directory for training
        '''
        self.work_dir = work_dir
        self.trainer = None
        self.model = None
        self.classes = None

    def set_model_trainer(self, model, trainer):
        '''Set the model/trainer build outside
        model: tvdl model
        trainer: pytorch lighting trainer
        '''
        self.model = model
        self.trainer = trainer

    def build_model(self):
        ''' build tvdl model. should be implemented in subclass.
        '''
        return None

    def build_trainer(self, cbs=None):
        '''
        cbs: (tupple) callbacks
        '''
        import pytorch_lightning as pl

        epochs = self._get_param('epochs', 10)
        gpus = self._get_param('gpus', [0])
        check_per_epoch = self._get_param('check_per_epoch', 1)
        trainer = pl.Trainer(default_root_dir=self.work_dir,
                             max_epochs=epochs,
                             gpus=gpus,
                             check_val_every_n_epoch=check_per_epoch,
                             callbacks=cbs)
        return trainer

    def train(self, ill, train_schedule, train_tfms=[], valid_tfms=[],
              cbs=[]):
        '''
        ill: ImageLabelList
        train_tfms: list of transform function for train dataset
        valid_tfms: list of transform function for valid dataset
        train_schedule:
            {'backbone': 'resnet50',
             'lr': 0.01,
             'bs': 16,
             'num_workers':1,
             'epochs': 10,
             'gpus': [0],
             'check_per_epoch': 2,
             'monitor': 'val_acc',
            }
        '''
        from pytorch_lightning.callbacks import ModelCheckpoint

        self.train_schedule = train_schedule
        self._check_train_schedule()
        self.classes = ill.labelset()
        model = self.build_model()

        monitor = self._get_param('monitor', 'val_acc')
        mode = 'min' if 'loss' in monitor else 'max'
        self._checkpoint_callback = ModelCheckpoint(
            monitor=monitor,
            dirpath=self.work_dir,
            filename='bestmodel',
            mode=mode,
            verbose=True,
            save_weights_only=True,
        )

        trainer = self.build_trainer([self._checkpoint_callback,
                                      *[get_inference_status_cb(self.train_schedule["epochs"], cb, monitor) for cb in
                                        cbs]])
        self.set_model_trainer(model, trainer)

        bs = self._get_param('bs', 16)
        num_workers = self._get_param('num_workers', 8)
        train_dl, valid_dl = ill.dataloader(train_tfms, valid_tfms,
                                            bs=bs,
                                            num_workers=num_workers,
                                            pin_memory=True)
        self._ill = ill
        self._train_dl = train_dl
        self._valid_dl = valid_dl
        self.fit(train_dl, valid_dl)

    def _check_train_schedule(self):
        for key in self.train_schedule.keys():
            if key not in self.SUPPORT_SCHEDULE_KEYS:
                print('unsupport:', key)
                print('SUPPORT_SCHEDULE_KEYS:', self.SUPPORT_SCHEDULE_KEYS)
                raise KeyError(key)

    def _get_param(self, key, def_val):
        if key in self.train_schedule.keys():
            return self.train_schedule[key]
        else:
            return def_val

    def fit(self, train_dl, valid_dl):
        ''' start training
        train_dl: train dataloader
        valid_dl: valid dataloader
        '''
        self.input_shape = list(valid_dl.dataset[0][0].shape)
        self.trainer.fit(self.model, train_dl, valid_dl)

    def package_model(self, model_path, import_cmd=None, classes=None, model_fmt='onnx', **kwargs):
        ''' package capp model
        model_path: capp model path
        import_cmd: import cmd for model inference (used by adc gpu server)
        classes: label set
        model_fmt: package model format, one of ['onnx','ckpt','all']
        '''
        ckpt_path = self._checkpoint_callback.best_model_path

        classes = classes if classes else self.classes
        ext_info = {'classes': classes,
                    'model_cls': str(self.model.__class__)[8:-2]}  # e.g. tvdl.detection.yolo.YOLO
        ext_info.update(self.model.hparams)

        pkg_files = {}
        if model_fmt == 'all' or model_fmt == 'onnx':
            onnx_path = osp.join(self.work_dir, 'model.onnx')
            input_shape = self.input_shape.copy()
            input_shape.insert(0, 1)
            self.model.export_onnx(onnx_path, input_shape, **kwargs)
            pkg_files['model.onnx'] = onnx_path
        if model_fmt == 'all' or model_fmt == 'ckpt':
            pkg_files['model.ckpt'] = ckpt_path

        package_capp(model_path, import_cmd,
                     ext_info=ext_info, pkg_files=pkg_files,
                     export_model_info=True)

    def evaluate(self, result_path, callback=None):
        return 1, 1, 1  # precision, recall, f1


class TvdlCategoryTrain(TvdlTrain):
    ''' Tvdl catetory model training
    '''
    SUPPORT_SCHEDULE_KEYS = ['backbone', 'lr', 'bs', 'num_workers', 'epochs', 'gpus', 'check_per_epoch', 'monitor',
                             'train_bn', 'img_c', 'multi_label', 'mixup_ratio', 'freeze_to_n']

    def build_model(self):
        from tvdl.classification import TransferLearningModel

        backbone = self._get_param('backbone', 'resnet18')
        lr = self._get_param('lr', 0.001)
        train_bn = self._get_param('train_bn', True)
        img_c = self._get_param('img_c', 3)
        multi_label = self._get_param('multi_label', False)
        mixup_ratio = self._get_param('mixup_ratio', 0.0)
        freeze_to_n = self._get_param('freeze_to_n', -1)

        model = TransferLearningModel(num_classes=len(self.classes),
                                      backbone=backbone,
                                      lr=lr,
                                      train_bn=train_bn,
                                      img_c=img_c,
                                      multi_label=multi_label,
                                      mixup_ratio=mixup_ratio,
                                      freeze_to_n=freeze_to_n)
        return model

    def evaluate(self, result_path, callback=None):
        ''' generate valid dataset evaluate index.html
                result_path: dir for save evaluate result.
                '''

        from tvdl.classification import TransferLearningModel
        import torch
        from fastprogress.fastprogress import progress_bar

        self.model.load_from_checkpoint(self._checkpoint_callback.best_model_path)
        self.model.eval()

        with torch.no_grad():
            y_pred = []
            for idx, (bx, bx_info) in enumerate(progress_bar(self._valid_dl)):

                outputs = self.model.forward(bx)
                pp = TransferLearningModel.post_process(outputs, self._get_param('multi_label', False))
                for yp in pp:
                    y_pred.append(yp.cpu().numpy())
                if callback is not None:
                    status = {
                        'desc': "evaluate",
                        'percent': int(idx * 100 / len(self._valid_dl)),
                    }
                    callback(status)
            if callback is not None:
                status = {
                    'desc': "evaluate",
                    'percent': 100,
                }
                callback(status)

        train, valid = self._ill.split()
        evac = EvalCategory(y_pred=y_pred, y_true=valid.y, classes=self.classes)
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


class TvdlInference:
    ''' Tvdl model inference
    '''

    def __init__(self, model_path, work_dir=None, use_onnx=True, use_fp16=False, devices=['cuda']):
        '''
        model_path: capp model path
        work_dir: work dir
        use_onnx: whether use onnx for inference
        use_fp16: whether use fp16 for inference
        devices:
            use_onnx True: devices can be any combination of ['cuda', 'tensorrt', 'openvino', 'cpu']
            use_onnx False: device can be one of ['cuda', 'cpu']
        '''
        self.model_dir = unpackage_capp(model_path)
        self.work_dir = self.model_dir if work_dir is None else work_dir
        self.model_info = None
        self.devices = devices
        self.use_onnx = use_onnx
        self.use_fp16 = use_fp16
        self.model = None

    def get_model_info(self):
        ''' load model info
        '''
        if self.model_info is None:
            with open(osp.join(self.model_dir, 'model_info.yml'), 'rt', encoding='utf-8') as fp:
                self.model_info = yaml.load(fp, Loader=yaml.FullLoader)
                self.model_info['ckpt_path'] = osp.join(self.model_dir, 'model.ckpt')
                self.model_info['onnx_path'] = osp.join(self.model_dir, 'model.onnx')
        return self.model_info

    def get_class_list(self):
        '''get model class list
        '''
        model_info = self.get_model_info()
        return model_info['classes']

    def load_model(self):
        ''' load the model and warup for inference
        '''
        model_info = self.get_model_info()
        model_cls = model_info['model_cls']
        pkg = '.'.join(model_cls.split('.')[:2])
        tvdl_cls = model_cls.split('.')[-1]
        self.import_model_cmd = f'from {pkg} import {tvdl_cls}'
        self.model = None
        if self.use_onnx:
            from tvdl.common import OrtInference
            self.model = OrtInference(model_info['onnx_path'],
                                      devices=self.devices)
        else:
            exec(self.import_model_cmd)
            self.model = eval(tvdl_cls).load_from_checkpoint(model_info['ckpt_path'])
            self.model.freeze()
            self.model.cuda() if self.devices[0] == 'cuda' else self.model.cpu()
            if self.use_fp16:
                self.model = self.model.half()


class TvdlCategoryInference(TvdlInference):
    ''' Tvdl category model inference
    '''

    def predict(self, ill, tfms=None, bs=16, num_workers=8):
        '''
        ill (ImageBoxLabelList)
        tfms (list) tfm list
        bs (int) batch size
        num_works (int) works's num
        box_tfm (Callable) box transform function
        output:
            sigmod result for multi_label category
            softmax result for single_label category
        '''
        from tvdl.classification import TransferLearningModel
        from fastprogress.fastprogress import progress_bar

        if not self.model:
            self.load_model()

        _, valid_dl = ill.dataloader(tfms, tfms, bs=bs, num_workers=num_workers)
        y_pred = []
        for bx, bx_info in progress_bar(valid_dl):
            if not self.use_onnx:
                bx = bx.to(self.model.device)
                if self.use_fp16:
                    bx = bx.half()
            outputs = self.model.forward(bx)
            outputs = outputs[0] if self.use_onnx else outputs
            pp = TransferLearningModel.post_process(outputs, self.model_info['multi_label'])
            for yp in pp:
                y_pred.append(yp.cpu().numpy())
        return y_pred

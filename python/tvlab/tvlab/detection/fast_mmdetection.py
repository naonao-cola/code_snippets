'''
Copyright (C) 2023 TuringVision

Image detection interface for model training and inference
'''
import os.path as osp
import yaml, time, os
from zipfile import ZipFile, ZIP_DEFLATED
from uuid import getnode as get_mac
from .eval_detection import EvalDetection
from ..ui import bokeh_figs_to_html
from ..utils import package_capp, unpackage_capp

_MODEL_INFO_FILE = 'model_info.yml'

__all__ = ['FastMMDetectionInference', 'FastMMDetectionTrain']


def get_mmdet_model_pred(model, dataset, class_list, box_tfm=None):
    '''
    get detectron2 model predict result
    In:
        model: detectron2 model
        loader: data_loader
        class_list (list): ['A', 'B', 'C' ..]
        box_tfm: transform function for predict result.
            Default will rescale the output instances to the original image size if box_tfm is None.

            If there is clipping preprocessing, auto rescale will cause abnormal results.
            So you can convert the predicted bbox by adding a `box_tfm` function.


            eg: Image resolution is: 1024x1024

                preprocessing is:
                    def xxx_crop(x):
                        x = cv2.resize(512, 512) # resize 1024 to 512
                        x = x[64:384, 32:468]    # crop
                        return x

                so box_tfm is :
                    def box_tfm(box, ori_shape):
                        # box is [l, t, r, b]
                        # ori_shape is [h, w] = [1024, 1024]

                        box[0] += 32 # left add crop offset
                        box[2] += 32 # right add crop offset

                        box[1] += 64 # top add crop offset
                        box[3] += 64 # bottom add crop offset

                        box = [l*2 for l in box] # resize 512 to 1024
                        return box
    out:
        y_pred (list): [{'bboxes': [...], 'labels': [...]}]
    '''
    from fastprogress.fastprogress import progress_bar

    from mmcv.parallel import collate, scatter
    import torch
    model = model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        y_pred = list()
        for inputs in progress_bar(dataset):
            data = {key: [value] for key, value in inputs.items()}
            data = scatter(collate([data], samples_per_gpu=1), [device])[0]
            result = model(return_loss=False, rescale=box_tfm is None, **data)

            bboxes, labels = list(), list()
            for i, output in enumerate(result[0]):
                bboxes += output.tolist()
                labels += [class_list[i]] * len(output)
            if box_tfm:
                ori_shape = inputs['ori_shape']
                bboxes = [box_tfm(y, ori_shape) for box in bboxes]
            y = {'bboxes': bboxes, 'labels': labels}
            y_pred.append(y)
    return y_pred


class FastMMDetectionInference:
    ''' Detectron2 detection model inference
    '''
    def __init__(self, model_path, work_dir=None):
        '''
        model_path: capp model path
        work_dir:
        '''
        from mmcv import Config
        self._model_dir = unpackage_capp(model_path)
        self._work_dir = self._model_dir if work_dir is None else work_dir

        with open(osp.join(self._model_dir, 'cfg.yaml'), 'rt', encoding='utf-8') as fp:
            cfg_dict = yaml.load(fp, Loader=yaml.FullLoader)
            self.cfg = Config(cfg_dict)
        self.model = None
        self._model_info = None
        self._tfms = []
        self._post_init()

    def _post_init(self):
        ''' for update tfms
        '''
        pass

    def model_info(self):
        ''' load model info
        '''
        if self._model_info is None:
            with open(osp.join(self._model_dir, _MODEL_INFO_FILE), 'rt', encoding='utf-8') as fp:
                self._model_info = yaml.load(fp, Loader=yaml.FullLoader)
                if isinstance(self._model_info['train_schedule'], (tuple, list)):
                    self._model_info['train_schedule'] = self._model_info['train_schedule'][0]
        return self._model_info

    def get_class_list(self):
        '''get model class list
        '''
        model_info = self.model_info()
        return model_info['classes']

    def load_model(self):
        ''' load model
        '''
        from mmdet.apis import init_detector
        if not self.model:
            self.model = init_detector(self.cfg, osp.join(self._model_dir, 'model.pth'))
        return self.model

    def predict(self, ibll, tfms=None, bs=None, num_workers=None, box_tfm=None):
        ''' get ImageBBoxLabelList valid data predict result
        In:
            ibll: ImageBBoxLabelList
            tfms: transform function list, see ImageBBoxLabelList.add_tfm
            bs: batch_size, default from model_info
            num_workers: default from model_info
            box_tfm: transform function for predict result.
                Default will rescale the output instances to the original image size if box_tfm is None.

                If there is clipping preprocessing, auto rescale will cause abnormal results.
                So you can convert the predicted bbox by adding a `box_tfm` function.


                eg: Image resolution is: 1024x1024

                    preprocessing is:
                        def xxx_crop(x):
                            x = cv2.resize(512, 512) # resize 1024 to 512
                            x = x[64:384, 32:468]    # crop
                            return x

                    so box_tfm is :
                        def box_tfm(box, ori_shape):
                            # box is [l, t, r, b]
                            # ori_shape is [h, w] = [1024, 1024]

                            box[0] += 32 # left add crop offset
                            box[2] += 32 # right add crop offset

                            box[1] += 64 # top add crop offset
                            box[3] += 64 # bottom add crop offset

                            box = [l*2 for l in box] # resize 512 to 1024
                            return box
        Out:
            y_pred (list): [{'bboxes': [...], 'labels': [...]}]
        '''
        model_info = self.model_info()
        _, valid = ibll.split(show=False)
        if bs is None:
            bs = model_info['train_schedule']['bs']
            bs = min(len(valid), bs)

        if num_workers is None:
            num_workers = model_info['train_schedule']['num_workers']

        _, dataset = ibll.mmdet_data(self.cfg, tfms, tfms,
                                     path=self._work_dir,
                                     bs=bs, num_workers=num_workers,
                                     show_dist=False)

        class_list = self.get_class_list()
        model = self.load_model()
        return get_mmdet_model_pred(model, dataset, class_list, box_tfm=box_tfm)

    def evaluate(self, ibll, tfms=None, bs=None, num_workers=None,
                 iou_threshold=0.5, bboxes_only=False, box_tfm=None):
        ''' get ImageBBoxLabelList valid data evaluate result
        In:
            ibll: ImageBBoxLabelList
            tfms: transform function list, see ImageBBoxLabelList.add_tfm
            bs: batch_size, default from model_info
            num_workers: default from model_info
            iou_threshold: iou threshold
            bboxes_only: only use model predict bboxes, ignore model predict class
            box_tfm: transform function for predict result.
                Default will rescale the output instances to the original image size if box_tfm is None.

                If there is clipping preprocessing, auto rescale will cause abnormal results.
                So you can convert the predicted bbox by adding a `box_tfm` function.


                eg: Image resolution is: 1024x1024

                    preprocessing is:
                        def xxx_crop(x):
                            x = cv2.resize(512, 512) # resize 1024 to 512
                            x = x[64:384, 32:468]    # crop
                            return x

                    so box_tfm is :
                        def box_tfm(box, ori_shape):
                            # box is [l, t, r, b]
                            # ori_shape is [h, w] = [1024, 1024]

                            box[0] += 32 # left add crop offset
                            box[2] += 32 # right add crop offset

                            box[1] += 64 # top add crop offset
                            box[3] += 64 # bottom add crop offset

                            box = [l*2 for l in box] # resize 512 to 1024
                            return box
        Out:
            evad: EvalDetection
        '''
        _, valid = ibll.split(show=False)
        y_pred = self.predict(ibll, tfms=tfms, bs=bs,
                              num_workers=num_workers,
                              box_tfm=box_tfm)
        return EvalDetection(y_pred=y_pred, y_true=valid.y, iou_threshold=iou_threshold,
                             classes=self.get_class_list(), bboxes_only=bboxes_only)


def mm_config_change(cfg, find_key, value):
    from mmcv.utils.config import Config
    if isinstance(cfg, (dict, Config)):
        for key, v in cfg.items():
            if key == find_key:
                cfg[key] = value
            else:
                mm_config_change(cfg[key], find_key, value)
    elif isinstance(cfg, (list, tuple)):
        for v in cfg:
            mm_config_change(v, find_key, value)


def change_syncbn_to_bn(cfg):
    from mmcv.utils.config import Config
    if isinstance(cfg, (dict, Config)):
        for key, v in cfg.items():
            if key == 'norm_cfg' and cfg[key]['type'] == 'SyncBN':
                cfg[key]['type'] = 'BN'
            else:
                change_syncbn_to_bn(cfg[key])
    elif isinstance(cfg, (list, tuple)):
        for v in cfg:
            change_syncbn_to_bn(v)


class FastMMDetectionTrain:
    ''' mmdetection detection model training
    '''
    SUPPORT_SCHEDULE_KEYS = ['bs', 'num_workers']

    def __init__(self, work_dir, cfg_file):
        from mmcv import Config
        self.cfg = Config.fromfile(cfg_file)
        self.cfg.gpu_ids = [0]
        self.cfg.log_config.interval = 20
        self.cfg.checkpoint_config.interval = -1
        change_syncbn_to_bn(self.cfg)
        self._work_dir = work_dir
        self.classes = None
        self.ibll = None
        self._train_schedule = None
        self.model = None
        self.datasets = None

    def _check_train_schedule(self, train_schedule):
        for key in train_schedule.keys():
            if key not in self.SUPPORT_SCHEDULE_KEYS:
                print('unsupport:', key)
                print('SUPPORT_SCHEDULE_KEYS:', self.SUPPORT_SCHEDULE_KEYS)
                raise KeyError(key)

    def train(self, ibll, train_schedule,
              train_tfms=None, valid_tfms=None,
              callback=None, resume=False):
        '''
        ibll: ImageBBoxLabelList
        train_schedule: {'bs': 1, 'num_workers': 1}
        train_tfms: list of transform function for train dataset
        valid_tfms: list of transform function for valid dataset
        resume(bool): resume from work_dir
        '''
        from mmdet.models import build_detector
        from mmdet.utils import get_root_logger
        from mmdet.apis import train_detector
        import logging
        from .mmdet_logger_hook import MMdetLoggerHook

        self._check_train_schedule(train_schedule)
        self.ibll = ibll
        self.classes = ibll.labelset()
        mm_config_change(self.cfg, 'num_classes', len(self.classes))
        mm_config_change(self.cfg, 'min_bbox_size', 16)
        # prepare datasets
        self.datasets = ibll.mmdet_data(self.cfg, train_tfms, valid_tfms,
                                        path=self._work_dir,
                                        bs=train_schedule['bs'],
                                        num_workers=train_schedule['num_workers'],
                                        show_dist=False)

        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        self.cfg.log_level = logging.WARNING
        log_file = osp.join(self._work_dir, '{}.log'.format(timestamp))
        get_root_logger(log_file=log_file, log_level=self.cfg.log_level)

        # build detector
        self.model = build_detector(self.cfg.model,
                                    train_cfg=self.cfg.get("train_cfg", None),
                                    test_cfg=self.cfg.get("test_cfg", None))
        self.model.CLASSES = ibll.labelset()

        self.cfg.log_config.hooks = [{'type': MMdetLoggerHook, 'by_epoch': False}]

        # start train
        self._train_schedule = train_schedule

        train_detector(
                    self.model,
                    [self.datasets[0]],
                    self.cfg,
                    distributed=False,
                    validate=False,
                    timestamp=timestamp)

    def evaluate(self, result_path, iou_threshold=0.5,
                 bboxes_only=False, callback=None, box_tfm=None):
        ''' generate valid dataset evaluate index.html
        result_path: dir for save evaluate result.
        iou_threshold: iou_threshold
        bboxes_only: only use model predict bboxes, ignore model predict class
        box_tfm: transform function for predict result.
            Default will rescale the output instances to the original image size if box_tfm is None.

            If there is clipping preprocessing, auto rescale will cause abnormal results.
            So you can convert the predicted bbox by adding a `box_tfm` function.


            eg: Image resolution is: 1024x1024

                preprocessing is:
                    def xxx_crop(x):
                        x = cv2.resize(512, 512) # resize 1024 to 512
                        x = x[64:384, 32:468]    # crop
                        return x

                so box_tfm is :
                    def box_tfm(box, ori_shape):
                        # box is [l, t, r, b]
                        # ori_shape is [h, w] = [1024, 1024]

                        box[0] += 32 # left add crop offset
                        box[2] += 32 # right add crop offset

                        box[1] += 64 # top add crop offset
                        box[3] += 64 # bottom add crop offset

                        box = [l*2 for l in box] # resize 512 to 1024
                        return box
        '''
        train, valid = self.ibll.split(show=False)
        y_pred = get_mmdet_model_pred(self.model, self.datasets[1],
                                      self.classes, box_tfm=box_tfm)

        evad = EvalDetection(y_pred=y_pred, y_true=valid.y,
                             iou_threshold=iou_threshold,
                             classes=self.classes, bboxes_only=bboxes_only)
        fig_list = [
            self.ibll.show_dist(need_show=False),
            self.ibll.show_split(train, valid, need_show=False),
            evad.plot_precision_conf(need_show=False),
            evad.plot_recall_conf(need_show=False),
            evad.plot_f1_conf(need_show=False),
            evad.plot_precision_recall(need_show=False),
            evad.plot_precision_iou(need_show=False),
            evad.plot_recall_iou(need_show=False),
            evad.plot_bokeh_table(need_show=False)
        ]
        bokeh_figs_to_html(fig_list, html_path=osp.join(result_path, 'index.html'),
                           title='Evaluate Catgory')
        evad.to_pkl(osp.join(result_path, 'evaluate.pkl'))

        result = evad.get_result()
        if 'Total' in result:
            total = result['Total']
        elif 'TotalNoOther' in result:
            total = result['TotalNoOther']

        return total['precision'], total['recall'], total['f1']

    def package_model(self, model_path, import_cmd=None, **kwargs):
        ''' package capp model
        model_path: capp model path
        import_cmd: import cmd for model inference (used by adc gpu server)
        '''
        from mmcv.runner import save_checkpoint

        with open(osp.join(self._work_dir, 'cfg.yaml'), 'wt', encoding='utf-8') as fp:
            new_cfg_dict = self.cfg._cfg_dict.to_dict()
            new_cfg_dict['log_config']['hooks'] = []
            yaml.dump(new_cfg_dict, fp)

        classes = self.classes
        meta = {'CLASSES': classes}
        save_checkpoint(self.model, osp.join(self._work_dir, "model_final.pth"), meta=meta)

        ext_info = {'classes': self.classes,
                    'train_schedule': self._train_schedule}
        ext_info.update(kwargs)

        pkg_files = {'cfg.yaml': osp.join(self._work_dir, 'cfg.yaml'),
                     'model.pth': osp.join(self._work_dir, 'model_final.pth')}
        package_capp(model_path, import_cmd,
                     ext_info=ext_info, pkg_files=pkg_files)

'''
Copyright (C) 2023 TuringVision

Image detection interface for model training and inference
'''
import os.path as osp
import yaml, time, os
import numpy as np
from zipfile import ZipFile, ZIP_DEFLATED
from uuid import getnode as get_mac
from .eval_segmentation import EvalSegmentation
from ..ui import bokeh_figs_to_html
from ..utils import mask_to_polygons
from ..utils import package_capp, unpackage_capp

_MODEL_INFO_FILE = 'model_info.yml'

__all__ = ['FastMMSegmentationInference', 'FastMMSegmentationTrain']


def _get_evas(y_pred, y_true, iou_threshold=0.5, class_list=None, polygons_only=False):
    if polygons_only:
        y_pred = y_pred.tfm_label(lambda l: 'object')

    evas = EvalSegmentation(y_pred=y_pred, y_true=y_true,
                            iou_threshold=iou_threshold,
                            classes=class_list)
    return evas


def get_mmseg_model_pred(model, dataset, class_list, polygon_tfm=None):
    '''
    get mmsegmentation model predict result
    In:
        model: mmsegmentation model
        dataset: dataset
        class_list (list): ['A', 'B', 'C' ..]
        box_tfm: transform function for predict bboxes result.
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
        polygon_tfm: transform function for predict polygons result.
    out:
        y_pred: [{'bboxes': [...], labels': [...]}]
        or
        y_pred: [{'polygons': [...], labels': [...]}]
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
            result = model(return_loss=False, rescale=False, **data)

            polygons, labels = list(), list()
            seg = result[0]
            for i, cls in enumerate(class_list):
                mask = seg == i + 1
                ps = mask_to_polygons(mask, area_threshold=25)
                if ps:
                    labels += [cls] * len(ps)
                    polygons += [p+[1.0] for p in ps]

            if polygon_tfm:
                ori_shape = inputs['img_metas'].data['ori_shape']
                polygons = [polygon_tfm(polygon[:-1], ori_shape)+polygon[-1:]
                            for polygon in polygons]
            else:
                scale_factor = inputs['img_metas'].data['scale_factor']
                w_scale = scale_factor[0]
                h_scale = scale_factor[1]
                def _rescale_polygon(p):
                    p = np.array(p).reshape(-1, 2)
                    p[:, 0] /= w_scale
                    p[:, 1] /= h_scale
                    return p.flatten().tolist()
                polygons = [_rescale_polygon(polygon[:-1])+polygon[-1:]
                            for polygon in polygons]

            y = {'polygons': polygons, 'labels': labels}
            y_pred.append(y)
    return y_pred


class FastMMSegmentationInference:
    ''' mmsegmentation model inference
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
        from mmseg.apis import init_segmentor
        if not self.model:
            self.model = init_segmentor(self.cfg, osp.join(self._model_dir, 'model.pth'))
        return self.model

    def predict(self, ipll, tfms=None, bs=None, num_workers=None, polygon_tfm=None):
        ''' get ImagePolygonLabelList valid data predict result
        In:
            ipll: ImagePolygonLabelList
            tfms: transform function list, see ImagePolygonLabelList.add_tfm
            bs: batch_size, default from model_info
            num_workers: default from model_info
            polygon_tfm: transform function for predict result.
                Default will rescale the output instances to the original image size if
                polygon_tfm is None.

                If there is clipping preprocessing, auto rescale will cause abnormal results.
                So you can convert the predicted bpolygon by adding a `polygon_tfm` function.

                eg: Image resolution is: 1024x1024

                    preprocessing is:
                        def xxx_crop(x):
                            x = cv2.resize(512, 512) # resize 1024 to 512
                            x = x[64:384, 32:468]    # crop
                            return x

                    so polygon_tfm is :
                        def polygon_tfm(polygon, ori_shape):
                            # polygon is [x1, y1, x2, y2, x3, y3, ...]
                            # ori_shape is [h, w] = [1024, 1024]
                            polygon = np.array(polygon).reshape(-1, 2)

                            polygon[:, 0] += 64 # x add crop offset
                            polygon[:, 1] += 32 # y add crop offset

                            polygon *= 2 # resize 512 to 1024

                            return polygon.flatten().tolist()
        '''
        model_info = self.model_info()
        _, valid = ipll.split(show=False)
        if bs is None:
            bs = model_info['train_schedule']['bs']
            bs = min(len(valid), bs)

        if num_workers is None:
            num_workers = model_info['train_schedule']['num_workers']

        _, dataset = ipll.mmseg_data(self.cfg, tfms, tfms,
                                     path=self._work_dir,
                                     bs=bs, num_workers=num_workers,
                                     show_dist=False)

        class_list = self.get_class_list()
        model = self.load_model()
        return get_mmseg_model_pred(model, dataset, class_list, polygon_tfm=polygon_tfm)

    def evaluate(self, ipll, tfms=None, bs=None, num_workers=None,
                 iou_threshold=0.5, polygons_only=False, polygon_tfm=None):
        ''' get ImagePolygonLabelList valid data evaluate result
        In:
            ipll: ImagePolygonLabelList
            tfms: transform function list, see ImagePolygonLabelList.add_tfm
            bs: batch_size, default from model_info
            num_workers: default from model_info
            iou_threshold: iou threshold
            polygon_only: only use model predict polygons, ignore model predict class
            polygon_tfm: transform function for predict result.
                Default will rescale the output instances to the original image size if
                polygon_tfm is None.

                If there is clipping preprocessing, auto rescale will cause abnormal results.
                So you can convert the predicted bpolygon by adding a `polygon_tfm` function.

                eg: Image resolution is: 1024x1024

                    preprocessing is:
                        def xxx_crop(x):
                            x = cv2.resize(512, 512) # resize 1024 to 512
                            x = x[64:384, 32:468]    # crop
                            return x

                    so polygon_tfm is :
                        def polygon_tfm(polygon, ori_shape):
                            # polygon is [x1, y1, x2, y2, x3, y3, ...]
                            # ori_shape is [h, w] = [1024, 1024]
                            polygon = np.array(polygon).reshape(-1, 2)

                            polygon[:, 0] += 64 # x add crop offset
                            polygon[:, 1] += 32 # y add crop offset

                            polygon *= 2 # resize 512 to 1024

                            return polygon.flatten().tolist()
            Out:
                evas: EvalSegmentation
            '''
        _, valid = ipll.split(show=False)
        y_pred = self.predict(ipll, tfms=tfms, bs=bs,
                              num_workers=num_workers,
                              polygon_tfm=polygon_tfm)
        return _get_evas(y_pred, valid.y, iou_threshold,
                         self.get_class_list(), polygons_only)


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


class FastMMSegmentationTrain:
    ''' mmsegmentation model training
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
        self.ipll = None
        self._train_schedule = None
        self.model = None
        self.datasets = None

    def _check_train_schedule(self, train_schedule):
        for key in train_schedule.keys():
            if key not in self.SUPPORT_SCHEDULE_KEYS:
                print('unsupport:', key)
                print('SUPPORT_SCHEDULE_KEYS:', self.SUPPORT_SCHEDULE_KEYS)
                raise KeyError(key)

    def train(self, ipll, train_schedule,
              train_tfms=None, valid_tfms=None,
              callback=None, resume=False):
        '''
        ipll: ImagePolygonLabelList
        train_schedule: {'bs': 1, 'num_workers': 1}
        train_tfms: list of transform function for train dataset
        valid_tfms: list of transform function for valid dataset
        resume(bool): resume from work_dir
        '''
        from mmseg.models import build_segmentor
        from mmseg.utils import get_root_logger
        from mmseg.apis import train_segmentor
        import logging
        from .mmseg_logger_hook import MMsegLoggerHook
        from ..detection.fast_mmdetection import mm_config_change

        self._check_train_schedule(train_schedule)
        self.ipll = ipll
        self.classes = ipll.labelset()
        mm_config_change(self.cfg, 'num_classes', len(self.classes) + 1)
        # prepare datasets
        self.datasets = ipll.mmseg_data(self.cfg, train_tfms, valid_tfms,
                                        path=self._work_dir,
                                        bs=train_schedule['bs'],
                                        num_workers=train_schedule['num_workers'],
                                        show_dist=False)

        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        self.cfg.log_level = logging.ERROR
        log_file = osp.join(self._work_dir, '{}.log'.format(timestamp))
        get_root_logger(log_file=log_file, log_level=self.cfg.log_level)

        # build segmentor
        self.model = build_segmentor(self.cfg.model,
                                     train_cfg=self.cfg.get("train_cfg", None),
                                     test_cfg=self.cfg.get("test_cfg", None))
        self.model.CLASSES = ipll.labelset()

        self.cfg.log_config.hooks = [{'type': MMsegLoggerHook, 'by_epoch': False}]

        # start train

        self._train_schedule = train_schedule

        train_segmentor(self.model,
                        [self.datasets[0]],
                        self.cfg,
                        distributed=False,
                        validate=False,
                        timestamp=timestamp)

    def evaluate(self, result_path, iou_threshold=0.5,
                 polygons_only=False, callback=None, polygon_tfm=None):
        ''' generate valid dataset evaluate index.html
        result_path: dir for save evaluate result.
        iou_threshold: iou_threshold
        polygon_only: only use model predict polygons, ignore model predict class
        polygon_tfm: transform function for predict result.
            Default will rescale the output instances to the original image size if polygon_tfm is None.

            If there is clipping preprocessing, auto rescale will cause abnormal results.
            So you can convert the predicted bpolygon by adding a `polygon_tfm` function.

            eg: Image resolution is: 1024x1024

                preprocessing is:
                    def xxx_crop(x):
                        x = cv2.resize(512, 512) # resize 1024 to 512
                        x = x[64:384, 32:468]    # crop
                        return x

                so polygon_tfm is :
                    def polygon_tfm(polygon, ori_shape):
                        # polygon is [x1, y1, x2, y2, x3, y3, ...]
                        # ori_shape is [h, w] = [1024, 1024]
                        polygon = np.array(polygon).reshape(-1, 2)

                        polygon[:, 0] += 64 # x add crop offset
                        polygon[:, 1] += 32 # y add crop offset

                        polygon *= 2 # resize 512 to 1024

                        return polygon.flatten().tolist()
        '''
        ipll = self.ipll
        train, valid = self.ipll.split(show=False)
        y_pred = get_mmseg_model_pred(self.model, self.datasets[1],
                                      self.classes, polygon_tfm=polygon_tfm)

        y_pred = ipll.y.__class__(y_pred)
        evas = _get_evas(y_pred, valid.y, iou_threshold,
                         self.classes, polygons_only)
        fig_list = [
            ipll.show_dist(need_show=False),
            ipll.show_split(train, valid, need_show=False),
            evas.plot_precision_conf(need_show=False),
            evas.plot_recall_conf(need_show=False),
            evas.plot_f1_conf(need_show=False),
            evas.plot_precision_recall(need_show=False),
            evas.plot_precision_iou(need_show=False),
            evas.plot_recall_iou(need_show=False),
            evas.plot_bokeh_table(need_show=False)
        ]
        bokeh_figs_to_html(fig_list, html_path=osp.join(result_path, 'index.html'),
                           title='Evaluate Catgory')
        evas.to_pkl(osp.join(result_path, 'evaluate.pkl'))

        result = evas.get_result()
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
        palette = np.random.randint(0, 255, size=(len(classes), 3))
        meta = {'CLASSES': classes, 'PALETTE': palette}
        save_checkpoint(self.model, osp.join(self._work_dir, "model_final.pth"), meta=meta)

        ext_info = {'classes': self.classes,
                    'train_schedule': self._train_schedule}
        ext_info.update(kwargs)

        pkg_files = {'cfg.yaml': osp.join(self._work_dir, 'cfg.yaml'),
                     'model.pth': osp.join(self._work_dir, 'model_final.pth')}
        package_capp(model_path, import_cmd,
                     ext_info=ext_info, pkg_files=pkg_files)

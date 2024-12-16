'''
Copyright (C) 2023 TuringVision

Image detection interface for model training and inference
'''
import os.path as osp
import PIL, yaml, time, os
import numpy as np
import cv2
from zipfile import ZipFile, ZIP_DEFLATED
from uuid import getnode as get_mac
from .eval_detection import EvalDetection
from ..ui import bokeh_figs_to_html
from ..utils import mask_to_polygon
from .bbox_label import BBoxLabelList
from ..utils import package_capp, unpackage_capp

__all__ = ['FastDetectionInference', 'FastDetectionTrain',
           'get_detectron2_model_pred']

_MODEL_INFO_FILE = 'model_info.yml'


def get_detectron2_model_pred(model, loader, class_list, box_tfm=None, polygon_tfm=None):
    '''
    get detectron2 model predict result
    In:
        model: detectron2 model
        loader: data_loader
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
    import torch
    model = model.eval()

    with torch.no_grad():
        y_pred = list()
        for inputs in progress_bar(loader):
            outputs = model(inputs)
            for input, output in zip(inputs, outputs):
                has_mask = False
                bboxes = list()
                polygons = list()
                labels = list()
                if 'instances' in output:
                    instances = output['instances'].to("cpu")
                    image_size = instances.image_size
                    output = instances.get_fields()
                    if 'pred_masks' in output:
                        has_mask = True
                    for box, s in zip(output['pred_boxes'].tensor.tolist(),
                                      output['scores'].tolist()):
                        bboxes.append(box+[s])
                    if has_mask:
                        for mask, s in zip(output['pred_masks'].cpu().numpy(),
                                           output['scores'].tolist()):
                            polygon = mask_to_polygon(mask)
                            if polygon:
                                polygons.append(polygon + [s])
                    for cls in output['pred_classes'].tolist():
                        labels.append(class_list[cls])
                else:
                    proposals = output['proposals'].to("cpu")
                    image_size = proposals.image_size
                    output = proposals.get_fields()
                    logit = output['objectness_logits']
                    scores = torch.sigmoid(logit)
                    for box, s in zip(output['proposal_boxes'].tensor.tolist(),
                                      scores.tolist()):
                        bboxes.append(box+[s])
                        labels.append('object')

                ori_h = image_size[0]
                ori_w = image_size[1]
                scale_h = input['image'].shape[1]
                scale_w = input['image'].shape[2]
                if box_tfm:
                    def _restore_box(box):
                        l,t,r,b = box
                        l = l * scale_w / ori_w
                        r = r * scale_w / ori_w
                        t = t * scale_h / ori_h
                        b = b * scale_h / ori_h
                        return [l,t,r,b]

                    bboxes = [box_tfm(_restore_box(box[:4]), image_size)+box[4:] for box in bboxes]
                y = {'bboxes': bboxes, 'labels': labels}
                if has_mask:
                    if polygon_tfm:
                        def _restore_polygon(polygon):
                            polygon = np.array(polygon).reshape(-1, 2)
                            polygon[:, 0] *= scale_w / ori_w
                            polygon[:, 1] *= scale_h / ori_h
                            return polygon.flatten().tolist()
                        polygons = [polygon_tfm(_restore_polygon(polygon[:-1]), image_size)+polygon[-1:]
                                    for polygon in polygons]
                    y = {'polygons': polygons, 'labels': labels}
                y_pred.append(y)
    return y_pred


def _get_evad(y_pred, y_true, iou_threshold=0.5, class_list=None, bboxes_only=False):
    if bboxes_only:
        y_pred = y_pred.tfm_label(lambda l: 'object')

    evad = EvalDetection(y_pred=y_pred, y_true=y_true,
                         iou_threshold=iou_threshold,
                         classes=class_list)
    return evad


class FastDetectionInference:
    ''' Detectron2 detection model inference
    '''
    def __init__(self, model_path, work_dir=None):
        '''
        model_path: capp model path
        work_dir:
        '''
        from detectron2.config import get_cfg

        self._model_dir = unpackage_capp(model_path)
        self._work_dir = self._model_dir if work_dir is None else work_dir

        with open(osp.join(self._model_dir, 'cfg.yaml'), 'rt', encoding='utf-8') as fp:
            self.cfg = yaml.load(fp, Loader=yaml.UnsafeLoader)

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
        from detectron2.modeling import build_model
        from detectron2.checkpoint import DetectionCheckpointer

        if not self.model:
            self.model = build_model(self.cfg)
            self.model.eval()
            checkpointer = DetectionCheckpointer(self.model)
            checkpointer.load(osp.join(self._model_dir, 'model.pth'))
        return self.model

    def predict(self, ibll, tfms=None, bs=1, num_workers=0, box_tfm=None):
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

        _, loader = ibll.detectron2_data(self.cfg, tfms, tfms,
                                         path=self._work_dir,
                                         bs=bs, num_workers=num_workers,
                                         show_dist=False)

        class_list = self.get_class_list()
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_list)
        model = self.load_model()
        y_pred = get_detectron2_model_pred(model, loader, class_list, box_tfm=box_tfm)
        return ibll.y.__class__(y_pred)

    def evaluate(self, ibll, tfms=None, bs=1, num_workers=0,
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
        return _get_evad(y_pred, valid.y, iou_threshold,
                         self.get_class_list(), bboxes_only)

def get_training_hook(cb, trainer):
    from detectron2.engine import HookBase
    class TrainningHook(HookBase):
        def __init__(self, cb, trainer):
            self._trainer = trainer
            self._cb = cb

        def after_step(self):
            if self._cb is not None:
                status = {'desc': 'training_step', 'iter':self._trainer.iter, 'max_iter':self._trainer.max_iter}
                self._cb(status)
    return TrainningHook(cb, trainer)


class FastDetectionTrain:
    ''' Detectron2 detection model training
    '''
    SUPPORT_SCHEDULE_KEYS = ['bs', 'num_workers','epochs', 'lr']

    def __init__(self, work_dir, cfg_file, add_config_func=None):
        '''
        work_dir:
        cfg_file: Detectron2 config file path
        add_config_func: change default cfg before merge from cfg_file
            eg:
                def add_confg(cfg):
                    cfg.xxx.xx = xxx

                or

                def add_config(cfg):
                    from xxx import get_cfg
                    cfg = get_cfg()
                    return cfg
        '''
        from detectron2.config import get_cfg

        self._work_dir = work_dir
        self.cfg = get_cfg()
        if add_config_func:
            ret = add_config_func(self.cfg)
            if ret is not None:
                self.cfg = ret
        self.cfg.merge_from_file(cfg_file)
        self.classes = None
        self.loader = None
        self.trainer = None
        self.ibll = None
        self._train_schedule = None

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
        train_schedule: {'bs': 1, 'num_workers': 1, 'epochs':10, 'lr':0.02}
        train_tfms: list of transform function for train dataset
        valid_tfms: list of transform function for valid dataset
        resume(bool): resume from work_dir
        '''
        from .detectron2_trainer import Detectron2Trainer

        # do train schedule check
        self._check_train_schedule(train_schedule)

        self.ibll = ibll
        self.classes = ibll.labelset()
        if 'epochs' in train_schedule:
            self.cfg.SOLVER.MAX_ITER = int(len(ibll._train_idx) * train_schedule['epochs'] / train_schedule['bs'])
            self.cfg.SOLVER.STEPS = (int(self.cfg.SOLVER.MAX_ITER*0.7), )
        if 'lr' in train_schedule:
            self.cfg.BASE_LR = train_schedule['lr']
        loader = ibll.detectron2_data(self.cfg, train_tfms, valid_tfms,
                                      path=self._work_dir,
                                      bs=train_schedule['bs'],
                                      num_workers=train_schedule['num_workers'],
                                      show_dist=False)
        self.loader = loader

        trainer = Detectron2Trainer(self.cfg, *loader, callback)
        self.trainer = trainer

        self.trainer.register_hooks([get_training_hook(callback, self.trainer)])

        trainer.resume_or_load(resume=resume)

        self._train_schedule = train_schedule

        try:
            trainer.train()
        except KeyboardInterrupt:
            print('Trainning stopped by user request!')

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

        y_pred = get_detectron2_model_pred(self.trainer.model,
                                           self.loader[1],
                                           self.classes,
                                           box_tfm=box_tfm)
        y_pred = self.ibll.y.__class__(y_pred)
        evad = _get_evad(y_pred, valid.y, iou_threshold,
                         self.classes, bboxes_only)
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
        with open(osp.join(self._work_dir, 'cfg.yaml'), 'wt', encoding='utf-8') as fp:
            yaml.dump(self.cfg, fp)

        additional_state = {"iteration": self.trainer.iter}
        self.trainer.checkpointer.save("model_final", **additional_state)

        ext_info = {'classes': self.classes,
                    'train_schedule': self._train_schedule}
        ext_info.update(kwargs)

        pkg_files = {'cfg.yaml': osp.join(self._work_dir, 'cfg.yaml'),
                     'model.pth': osp.join(self._work_dir, 'model_final.pth')}
        package_capp(model_path, import_cmd,
                     ext_info=ext_info, pkg_files=pkg_files)

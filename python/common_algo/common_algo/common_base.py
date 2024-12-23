'''
Copyright (C) 2020 ThunderSoft

'''

from abc import ABC, abstractmethod
import os.path as osp
import numpy as np
from imgaug import augmenters as iaa
from tvlab import *

__all__ = ['CommonTrainBase']

TFMS_KEYS = ['blur', 'crop', 'noise', 'sharpen', 'color', 'rotate']


class CommonTrainBase(ABC):
    ''' basic class for common model trainning
    '''
    def __init__(self):
        self._ill = None
        self._model = None
        self._train_tfms = None
        self._valid_tfms = None
        self._img_aug_seq = None
        self._status_cb = None
        self._need_stop_cb = None
        self._inference_cmd = None
        self._ill_cls = None
        self._training_info = None

    def run(self, work_dir, training_info,
            need_stop_cb=None, status_cb=None):
        '''
        work_dir: work directory for save temp training files
        training_info:
        {
            'import_training': 'from common_algo import CommonCategoryTrain',
            train_schedule: {
                "tfms": ['blur','crop'], # valid key:['blur', 'crop', 'noise', 'sharpen', 'color', 'rotate']
                "model_level": "small", 
                "total_steps": 500,  
            },
            classList: ["classA", "classB", "classC"],
            "labelSet": [{
                "imageName": "a.jpg",
                "imagePath": "projects/dataset/a.jpg",
                "shapes": [{
                        "label": "classA",
                        "points": [878.33, 115.11, 1053.33, 209.55],
                        "confidence": 1,
                        "shapeType": "rectangle"
                    },{
                        ...
                    }]
            },{
                ...
            }]
        }
        need_stop_cb: callback function to check whether need to stop the training
        status_cb: callback function to output the training status
        '''
        self._training_info = training_info
        self._need_stop_cb = need_stop_cb
        self._status_cb = status_cb
        # 1. prepare dataset
        self._init_data()
        # 2. image aug
        self._init_tfms()
        # 3. init model
        self._init_model(work_dir)
        # 4. start training
        train_schedule = self._init_train_schedule()
        self._model.train(self._ill, train_schedule, self._train_tfms, self._valid_tfms,
                          self._process_and_route_cb)

    def package_model(self, model_path, callback=None):
        '''
        model_path: model path to save the packaged zip file
        '''
        self.update_status(callback, 'package_model', 0)
        self._model.package_model(model_path, self._inference_cmd)
        self.update_status(callback, 'package_model', 100)

    def evaluate(self, result_path, callback=None):
        '''
        result_path: the path contains the evaluate files
        '''
        return self._model.evaluate(result_path, callback=callback)

    def update_status(self, cb, desc, percent):
        if cb is not None:
            cb({'desc': desc, 'percent': percent})

    def _init_data(self):
        '''construct training data(ImageXXLabelList) from turbox format dict,
           and split train/valid by valid_pct
        '''
        self._mask_to_polygon(self._training_info)
        self._ill = eval(self._ill_cls).from_turbox_data(self._training_info)
        valid_pct = self._training_info['train_schedule'].get('valid_pct', 0.2)
        _, _valid = self._ill.split(valid_pct, show=False)
        assert len(_valid) > 0, f'At least {np.ceil(1.0/valid_pct)} label data is needed for trainning.'

    def _process_and_route_cb(self, status):
        ''' route the status info to user and pre-process some status:
        1. re-construct new status {'desc':'training_progress', 'percent': xx} to inform the total training progress
        2. check to stop the training
        '''
        if self._status_cb:
            self._status_cb(status)
            self._report_training_progress(status)
        if self._need_stop_cb and self._need_stop_cb():
            if 'desc' in status and status['desc'] == 'training_step':
                return self._stop_training()
        return False

    def _report_training_progress(self, status):
        '''
        TurboX need display the training progress. each model should report the approximate progress
        '''
        if 'epochs' in status:
            self._status_cb({'desc':'training_progress', 'percent': status['percent']})

    @abstractmethod
    def _stop_training(self):
        ''' Stop training method, rely on the training framework.
        In fastai, return True will stop the training
        In detectron2 or PaddlePaddle, raise a KeyboardInterrupt is a working method
        '''
        return False

    @abstractmethod
    def _init_model(self, work_dir):
        '''Instance the actual training class
        work_dir: same as run()
        '''
        pass

    @abstractmethod
    def _init_tfms(self):
        '''Initialize the transforms for train and valid.
        Custom image augument tfms and model require tfms should be concat here.
        Custom tfms like: ['blur', 'crop', 'noise', 'sharpen', 'color', 'rotate']
        Model require tfms e.g. resize_256
        '''
        pass

    def _get_img_aug_seq(self):
        ''' get the custom image augument Sequential from training_info
        '''
        schedule_info = self._training_info['train_schedule']
        if 'tfms' in schedule_info and len(schedule_info['tfms']) > 0:
            tfms = [
                iaa.GaussianBlur(sigma=(0, 0.5)),
                iaa.Crop(percent=(0, 0.2), keep_size=True), #iaa.Affine(shear=(-8, 8)),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.2*255), per_channel=0.5),
                iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0)),
                iaa.MultiplyHueAndSaturation((0.5, 1.5), per_channel=True),
                iaa.Affine(rotate=(-45, 45)),
            ]
            tfms = [tfms[TFMS_KEYS.index(k)] for k in schedule_info['tfms']]
            return iaa.Sequential(tfms)
        return None

    def _init_train_schedule(self):
        return {'bs':2, 'num_workers':2, 'epochs': 10, 'lr':0.001}

    def _mask_to_polygon(self, dict_info):
        '''
            generate the mask contours coordinates
        :param dict_info: input dict information
        :return:
        '''
        label_set = dict_info['labelSet']
        for i in range(len(label_set)):
            img_dir = label_set[i]['imagePath'][:-4]
            if not os.path.isdir(img_dir):
                continue
            shapes = label_set[i]['shapes']
            for img_name in os.listdir(img_dir):
                if img_name.split('.')[-1].lower() not in ['png', 'jpg', 'bmp', 'gif', 'jpeg', 'tiff']:
                    continue
                img_path = os.path.join(img_dir, img_name)
                gray = cv2.imread(img_path, 0)
                ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
                contours_context = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                if len(contours_context) == 2:
                    contours, hierarchy = contours_context
                else:
                    _, contours, hierarchy = contours_context
                for i in range(len(contours)):
                    shape = {
                        "label": None,
                        "points": None,
                        "confidence": 1,
                        "shapeType": "polygon"
                    }
                    shape['label'] = img_name.split('.')[0]
                    shape['points'] = list(contours[i].ravel())
                    shapes.append(shape)


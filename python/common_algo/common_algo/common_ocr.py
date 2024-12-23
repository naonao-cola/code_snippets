'''
Copyright (C) 2020 ThunderSoft

'''

import os
import os.path as osp
from imgaug import augmenters as iaa
from .common_base import CommonTrainBase
from tvlab import *

__all__ = ['CommonOcrInference', 'CommonOcrTrain']


class CommonOcrInference(FastOcrEnd2EndInference):
    '''Inference class for common Ocr
    '''
    def run(self, image_list, extra_info=None):
        '''
        image_list: image path list
        extra_info: extra parameters,
            default: {
                'tfms': None,
                'conf_threshold': 0.5 # confidence threshold to filter the result
            }
        return Turbox format result:
        {
            classList: ["classA", "classB", "classC"],
            "labelSet": [{
                "imageName": "a.jpg",
                "imagePath": "projects/dataset/a.jpg",
                "shapes": [{
                        "label": "classA",
                        "points": [878.3333, 115.1111, 1053.3333, 209.55555],
                        "confidence": 0.7878,
                        "shapeType": "rectangle"
                    },{
                        ...
                    }]
            },{
                ...
            }]
        }
        '''
        ill = ImageOCRPolygonLabelList(image_list)
        _, valid = ill.split(valid_pct=1.0, show=False)

        extra_info = {} if extra_info is None else extra_info
        conf_threshold = extra_info.get('conf_threshold', 0.5)
        tfms = extra_info.get('tfms', None)

        results = self.predict(ill, tfms)

        result_dict = {}
        result_dict['labelSet'] = []

        for idx, result in enumerate(results):
            img_path = valid.x[idx]
            img_info = {'imageName': osp.basename(img_path),'imagePath': img_path,
                'status': 'OK', 'shapes': []
            }
            if len(result['polygons']) > 0:
                polygons_scores = result['polygons']
                labels = result['labels']
                for label, ps in zip(labels, polygons_scores):
                    if ps[-1] > conf_threshold:
                        shape = {
                            'label': label,
                            'points': [round(p, 1) for p in ps[:-1]],
                            'confidence': round(ps[-1], 4),
                            'shapeType': 'polygon'
                        }
                        img_info['shapes'].append(shape)
            result_dict['labelSet'].append(img_info)
        return result_dict


class CommonOcrTrain(CommonTrainBase):
    '''Train class for common ocr
    '''
    def __init__(self):
        super(CommonOcrTrain, self).__init__()
        self._inference_cmd = 'from common_algo import CommonOcrInference'
        self._ill_cls = 'ImageOCRPolygonLabelList'

    def _img_aug_func(self, img, gt, idx, img_path):
        img, gt = imgaug_img_ocr_polygon_tfm(img, gt, self._img_aug_seq)
        return img, gt

    def _stop_training(self):
        raise KeyboardInterrupt('Request stop!')

    def _init_tfms(self):
        self._img_aug_seq = self._get_img_aug_seq()
        self._train_tfms =  [self._img_aug_func] if self._img_aug_seq else None
        self._valid_tfms = None

    def _init_model(self, work_dir):
        schedule_info = self._training_info['train_schedule']
        configs = {
            'small':'ocr_det_mv3_db_mtwi_v1.1.yaml',
            'normal':'ocr_det_mv3_db_mtwi_v1.1.yaml',
            'large':'ocr_det_mv3_db_mtwi_v1.1.yaml'
        }
        cur_path = os.path.dirname(os.path.realpath(__file__))
        cfg_file = configs[schedule_info['model_level']]
        cfg_file = osp.join(osp.abspath(cur_path), 'configs', cfg_file)
        self._model = FastOcrEnd2EndTrain(work_dir, cfg_file)

    def _init_train_schedule(self):
        schedule_info = self._training_info['train_schedule']
        train_schedule = {}
        train_schedule['bs'] = min(len(self._ill._train_idx), 2)
        train_schedule['num_workers'] = min(len(self._ill._train_idx), 2)
        train_schedule['epochs'] = schedule_info['total_steps']
        train_schedule['lr'] = 0.02
        return train_schedule

    def evaluate(self, result_path, callback=None):
        return self._model.evaluate()


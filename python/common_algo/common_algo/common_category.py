'''
Copyright (C) 2020 ThunderSoft

'''

import os.path as osp
import PIL
import numpy as np
from tvlab import *
from imgaug import augmenters as iaa
from .common_base import CommonTrainBase

__all__ = ['CommonCategoryInference', 'CommonCategoryTrain']


class CommonCategoryInference(FastCategoryInference):
    '''Inference class for common category
    '''
    def _post_init(self):
        self._tfms = [_resize_to_256]

    def run(self, image_list, extra_info=None):
        '''
        image_list: image path list
        extra_info: extra parameters
            default: {'conf_threshold': 0.5}
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
        ill = ImageLabelList(image_list, None)
        _, valid = ill.split(valid_pct=1.0, show=False)

        results = self.predict(ill)

        class_list = self.get_class_list()
        result_dict = {}
        result_dict['labelSet'] = []
        extra_info = {} if extra_info is None else extra_info
        conf_threshold = extra_info.get('conf_threshold', 0.5)

        for i, img_path in enumerate(valid.x):
            img_info = {
                'imageName': osp.basename(img_path),
                'imagePath': img_path,
                'status': 'OK',
                'shapes': []
            }
            code_list = [{'code': class_list[index], 'conf': conf}
                         for index, conf in enumerate(results[i])]
            code_list.sort(key=lambda x: x['conf'], reverse=True)
            top3_code_list = code_list[0:3]
            for x in top3_code_list:
                if x['conf'] < conf_threshold:
                    continue
                shape = {
                    'label': x['code'],
                    'confidence': round(x['conf'].item(), 4),
                }
                img_info['shapes'].append(shape)
            result_dict['labelSet'].append(img_info)
        return result_dict

def _resize_to_256(x, label):
    x = np.array(PIL.Image.fromarray(x).resize((256, 256)))
    return x, label

class CommonCategoryTrain(CommonTrainBase):
    '''Train class for common category
    '''
    def __init__(self):
        super(CommonCategoryTrain, self).__init__()
        self._inference_cmd = 'from common_algo import CommonCategoryInference'
        self._ill_cls = 'ImageLabelList'

    def _img_aug_func(self, img, label, idx, img_path):
        seq_det = self._img_aug_seq.to_deterministic()
        img = seq_det.augment_images([img])[0]
        return img, label

    def _report_training_progress(self, status):
        if 'epochs' in status:
            step_epochs = [step['epochs'] for step in self._train_schedule['steps']]
            step = int(status['pass_desc'].split('_')[-1])
            pass_epochs = sum(step_epochs[:step]) + status['epoch'] + 1
            percent = int(pass_epochs * 100.0 / sum(step_epochs))
            self._status_cb({'desc':'training_progress', 'percent': percent})

    def _stop_training(self):
        return True

    def _init_tfms(self):
        self._img_aug_seq = self._get_img_aug_seq()
        self._train_tfms =  [self._img_aug_func, _resize_to_256] if self._img_aug_seq else [_resize_to_256]
        self._valid_tfms = [_resize_to_256]

    def _init_model(self, work_dir):
        self._model = FastCategoryTrain(work_dir)

    def _init_train_schedule(self):
        train_schedule = {
            'basemodel': 'resnet50', 'num_workers': 4,
            'bs': min(len(self._ill._train_idx), 4),
        }
        schedule_info = self._training_info['train_schedule']
        models = {'small':'resnet18', 'normal':'resnet50', 'large':'resnet101'}
        train_schedule['basemodel'] = models[schedule_info['model_level']]
        train_schedule['steps'] = [{'epochs': schedule_info['total_steps'], 'lr': 0.001}]

        self._train_schedule = train_schedule
        return train_schedule

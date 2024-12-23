'''
Copyright (C) 2020 ThunderSoft

'''

import os
import os.path as osp
import numpy as np
import torchvision, torch
from imgaug import augmenters as iaa
from .common_base import CommonTrainBase
from tvlab import *

__all__ = ['CommonDetectionInference', 'CommonDetectionTrain']


class CommonDetectionInference(FastDetectionInference):
    '''Inference class for common detection
    '''
    def _resize_1024(self, img, gt, idx, path):
        img, gt = imgaug_img_bbox_tfm(img, gt, iaa.Resize({'longer-side': 1024,
                                                        'shorter-side': 'keep-aspect-ratio'}))
        return img, gt

    def run(self, image_list, extra_info=None):
        '''
        image_list: image path list
        extra_info: extra parameters,
            default: {
                'tfms': None,
                'iou_threshold': 0.3, # iou threshold for NMS
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
        ill = ImageBBoxLabelList(image_list)
        _, valid = ill.split(valid_pct=1.0, show=False)

        extra_info = {} if extra_info is None else extra_info
        iou_threshold = extra_info.get('iou_threshold', 0.3)
        conf_threshold = extra_info.get('conf_threshold', 0.5)
        tfms = extra_info.get('tfms', None)
        tfms = [tfms, self._resize_1024] if tfms is not None else [self._resize_1024]

        results = self.predict(ill, tfms)

        result_dict = {}
        result_dict['labelSet'] = []
        for idx, result in enumerate(results):
            img_path = valid.x[idx]
            img_info = {'imageName': osp.basename(img_path),'imagePath': img_path,
                'status': 'OK', 'shapes': []
            }
            if len(result['bboxes']) > 0:
                bboxes_scores = result['bboxes']
                labels = result['labels']
                bboxes = torch.from_numpy(np.array(bboxes_scores)[:,:-1])
                scores = torch.from_numpy(np.array(bboxes_scores)[:,-1])
                keep = torchvision.ops.nms(bboxes, scores, iou_threshold)
                for i in keep:
                    if scores[i] > conf_threshold:
                        shape = {
                            'label': labels[i],
                            'points': [round(p,1) for p in bboxes_scores[i][:-1]],
                            'confidence': round(bboxes_scores[i][-1], 4),
                            'shapeType': 'rectangle'
                        }
                        img_info['shapes'].append(shape)
            result_dict['labelSet'].append(img_info)
        return result_dict

class CommonDetectionTrain(CommonTrainBase):
    '''Train class for common detection
    '''
    def __init__(self):
        super(CommonDetectionTrain, self).__init__()
        self._inference_cmd = 'from common_algo import CommonDetectionInference'
        self._ill_cls = 'ImageBBoxLabelList'

    def _img_aug_func(self, img, gt, idx, img_path):
        img, gt = imgaug_img_bbox_tfm(img, gt, self._img_aug_seq)
        return img, gt

    def _resize_1024(self, img, gt, idx, path):
        img, gt = imgaug_img_bbox_tfm(img, gt, iaa.Resize({'longer-side': 1024,
                                                        'shorter-side': 'keep-aspect-ratio'}))
        return img, gt

    def _stop_training(self):
        raise KeyboardInterrupt('Request stop!')

    def _init_tfms(self):
        self._img_aug_seq = self._get_img_aug_seq()
        self._train_tfms =  [self._img_aug_func, self._resize_1024] if self._img_aug_seq else [self._resize_1024]
        self._valid_tfms = [self._resize_1024]

    def _init_model(self, work_dir):
        schedule_info = self._training_info['train_schedule']
        configs = {
            'small':'detection_faster_rcnn_R_50_FPN_1x.yaml',
            'normal':'detection_faster_rcnn_R_50_FPN_1x.yaml',
            'large':'detection_faster_rcnn_R_101_FPN_3x.yaml'
        }
        cur_path = os.path.dirname(os.path.realpath(__file__))
        cfg_file = configs[schedule_info['model_level']]
        cfg_file = osp.join(osp.abspath(cur_path), 'configs', cfg_file)
        self._model = FastDetectionTrain(work_dir, cfg_file)

    def _init_train_schedule(self):
        schedule_info = self._training_info['train_schedule']
        train_schedule = {}
        train_schedule['bs'] = min(len(self._ill._train_idx), 2)
        train_schedule['num_workers'] = min(len(self._ill._train_idx), 2)
        train_schedule['epochs'] = schedule_info['total_steps']
        train_schedule['lr'] = 0.02
        return train_schedule

'''
Copyright (C) 2020 ThunderSoft

'''
import os
import os.path as osp
from imgaug import augmenters as iaa
from .common_detection import CommonDetectionTrain
from tvlab import *

__all__ = ['CommonSegmentationInference', 'CommonSegmentationTrain']


class CommonSegmentationInference(FastSegmentationInference):
    '''Inference class for common segmentation
    '''
    def _resize_1024(self, img, gt, idx, path):
        img, gt = imgaug_img_polygon_tfm(img, gt, iaa.Resize({'longer-side': 1024,
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
        ipll = ImagePolygonLabelList(image_list)
        _, valid = ipll.split(valid_pct=1.0, show=False)

        extra_info = {} if extra_info is None else extra_info
        iou_threshold = extra_info.get('iou_threshold', 0.3)
        conf_threshold = extra_info.get('conf_threshold', 0.5)
        tfms = extra_info.get('tfms', None)
        tfms = [tfms, self._resize_1024] if tfms is not None else [self._resize_1024]

        results = self.predict(ipll, tfms)

        results.nms(iou_threshold)
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


class CommonSegmentationTrain(CommonDetectionTrain):
    '''Train class for common segmentation
    '''
    def __init__(self):
        super(CommonDetectionTrain, self).__init__()
        self._inference_cmd = 'from common_algo import CommonSegmentationInference'
        self._ill_cls = 'ImagePolygonLabelList'

    def _img_aug_func(self, img, gt, idx, img_path):
        img, gt = imgaug_img_polygon_tfm(img, gt, self._img_aug_seq)
        return img, gt

    def _resize_1024(self, img, gt, idx, path):
        img, gt = imgaug_img_polygon_tfm(img, gt, iaa.Resize({'longer-side': 1024,
                                                        'shorter-side': 'keep-aspect-ratio'}))
        return img, gt

    def _init_model(self, work_dir):
        schedule_info = self._training_info['train_schedule']
        configs = {
            'small':'segmentation_mask_rcnn_R_50_FPN_1x.yaml',
            'normal':'segmentation_mask_rcnn_R_50_FPN_1x.yaml',
            'large':'segmentation_mask_rcnn_R_101_FPN_3x.yaml'
        }
        cur_path = os.path.dirname(os.path.realpath(__file__))
        cfg_file = configs[schedule_info['model_level']]
        cfg_file = osp.join(osp.abspath(cur_path), 'configs', cfg_file)
        self._model = FastSegmentationTrain(work_dir, cfg_file)



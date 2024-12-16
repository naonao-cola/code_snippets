'''
Copyright (C) 2023 TuringVision
'''
import sys
import math
import cv2
import numpy as np
import json
import random
import paddle
import paddleocr
from ppocr.data.det.db_process import DBProcessTrain, DBProcessTest
from ppocr.data.det.random_crop_data import RandomCropData
from ppocr.data.det.make_shrink_map import MakeShrinkMap
from ppocr.data.det.make_border_map import MakeBorderMap
from ppocr.postprocess.db_postprocess import DBPostProcess

__all__ = ['reader_db_data', "DBPostProcessTS"]


class DBProcessTrainTS(DBProcessTrain):
    """
    The pre-process of DB for train mode
    """

    def __init__(self, params):
        """
        :param params: dict of params
        """
        super(DBProcessTrainTS, self).__init__(params)

    def __call__(self, imgvalue, labels):
        data = self.make_data_dict(imgvalue, labels)
        data = RandomCropData(data, self.image_shape[1:])
        data = MakeShrinkMap(data)
        data = MakeBorderMap(data)
        data = self.NormalizeImage(data)
        data = self.FilterKeys(data)
        return data['image'], data['shrink_map'], data['shrink_mask'], data[
            'threshold_map'], data['threshold_mask']


class DBProcessTestTS(DBProcessTest):
    """
    DB pre-process for Test mode
    """

    def __init__(self, params):
        """
        :param params: dict of params
        """
        super(DBProcessTestTS, self).__init__(params)
    def add_ignore_tag(self, labels):
        if labels is None:
            return None

        for item in labels:
            item['ignore'] = item['transcription'] in ['*', '###']
        return labels

    def __call__(self, im, labels):
        if self.resize_type == 0:
            im, (ratio_h, ratio_w) = self.resize_image_type0(im)
        else:
            im, (ratio_h, ratio_w) = self.resize_image_type1(im)
        im = self.normalize(im)
        im = im[np.newaxis, :]
        labels = self.add_ignore_tag(labels)
        return [im, (ratio_h, ratio_w), labels]

def transform_label(labels):
    if len(labels["labels"]) == 0:
        return None

    new_lbs = []
    for lb, pt in zip(labels["labels"], labels["polygons"]):
        points = np.array(pt).astype(np.int0).reshape((-1, 2)).tolist()
        result = {"transcription": lb, "points": points}
        new_lbs.append(result)
    return new_lbs

class DBDataReader(object):
    def __init__(self, ocrll, params, process_cls=None):
        self.mode = params['mode']
        if self.mode == "train":
            self.num_workers = params['num_workers']
            self.batch_size = params['train_batch_size_per_card']
        else:
            self.num_workers = 1
            self.batch_size = 1

        self.process = process_cls(params)
        self.ocrll = ocrll

    def __call__(self, process_id):
        def sample_iter_reader():
            img_num = len(self.ocrll)
            img_id_list = list(range(img_num))
            if self.mode == "train":
                random.shuffle(img_id_list)
            if sys.platform == "win32" and self.num_workers != 1:
                print("multiprocess is not fully compatible with Windows."
                      "num_workers will be 1.")
                self.num_workers = 1
            for img_id in range(process_id, img_num, self.num_workers):
                image, labels = self.ocrll[img_id_list[img_id]]
                if self.ocrll.img_mode == 'RGB':
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                elif self.ocrll.img_mode == 'L':
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

                tlt_labels = transform_label(labels)
                outs = self.process(image, tlt_labels)

                if outs is None:
                    continue
                yield outs

        def batch_iter_reader():
            batch_outs = []
            for outs in sample_iter_reader():
                batch_outs.append(outs)
                if len(batch_outs) == self.batch_size:
                    yield batch_outs
                    batch_outs = []

        return batch_iter_reader

def reader_db_data(ocrll, mode=None, config=None):
    """Create a reader for trainning

    Args:
        settings: arguments

    Returns:
        train reader
    """
    assert mode in ["train", "eval", "test"],\
        "Nonsupport mode:{}".format(mode)
    global_params = {
        'img_set_dir': "",
        "train_batch_size_per_card": 4,
        "test_batch_size_per_card": 4,
        "image_shape": [3, 640, 640],
    }
    if config is not None:
        global_params.update(config)

    _process_cls = None
    if mode == "train":
        params = {
            'num_workers': 2,
        }
        _process_cls = DBProcessTrainTS
    elif mode == "eval":
        params = {
            'test_image_shape': [736, 1280],
        }
        _process_cls = DBProcessTestTS
    else:
        params = {
            'do_eval': True
        }
        _process_cls = DBProcessTestTS

    params['mode'] = mode
    params.update(global_params)
    function = DBDataReader(ocrll, params, _process_cls)
    if mode == "train":
        if sys.platform == "win32":
            return function(0)
        readers = []
        num_workers = params['num_workers']
        for process_id in range(num_workers):
            readers.append(function(process_id))
        return paddle.reader.multiprocess_reader(readers, False)
    else:
        return function(0)

class DBPostProcessTS(DBPostProcess):
    def __init__(self, params):
        super(DBPostProcessTS, self).__init__(params)

    def __call__(self, outs_dict, ratio_list):
        pred = outs_dict['maps']

        pred = pred[:, 0, :, :]
        segmentation = pred > self.thresh
        boxes_batch = []
        scores_batch = []
        for batch_index in range(pred.shape[0]):

            mask = cv2.dilate(
                np.array(segmentation[batch_index]).astype(np.uint8),
                self.dilation_kernel)
            tmp_boxes, tmp_scores = self.boxes_from_bitmap(pred[batch_index],
                                                           mask)

            boxes = []
            scores = []
            for k in range(len(tmp_boxes)):
                if tmp_scores[k] > self.box_thresh:
                    boxes.append(tmp_boxes[k])
                    scores.append(tmp_scores[k])
            if len(boxes) > 0:
                boxes = np.array(boxes)

                ratio_h, ratio_w = ratio_list[batch_index]
                boxes[:, :, 0] = boxes[:, :, 0] / ratio_w
                boxes[:, :, 1] = boxes[:, :, 1] / ratio_h

            boxes_batch.append(boxes)
            scores_batch.append(scores)
        return boxes_batch, scores_batch

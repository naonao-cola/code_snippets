'''
Copyright (C) 2023 TuringVision

Implementation of TSNN Model base classes
'''
import pytorch_lightning as pl
import numpy as np
import torch
import torch.nn.functional as F

__all__ = ['ModelBase', 'ClassificationModelBase', 'DetectionModelBase',
           'SegmentationModelBase', 'AnomalyDetectionModelBase', 'OcrRecModelBase']


class ModelBase(pl.LightningModule):
    '''  Basic class for tsnn models
    '''

    def __init__(self, **kwargs):
        super().__init__()
        if 'pretrained' in kwargs.keys():
            self.hparams.update({'pretrained': kwargs['pretrained']})
        else:
            self.hparams.update({'pretrained': True})

    @classmethod
    def load_from_checkpoint(cls, path, **kwargs):
        model = super().load_from_checkpoint(path, pretrained=False, **kwargs)
        return model

    def export_onnx(self, onnx_path, input_shape, **kwargs):
        '''
        onnx_path: (str) onnx model path
        input_shape: (tuple) model input shape
        '''
        input_sample = torch.randn(input_shape)
        kwargs['opset_version'] = kwargs.get('opset_version', 11)
        kwargs['export_params'] = kwargs.get('export_params', True)
        kwargs['input_names'] = kwargs.get('input_names', ['inputs'])
        kwargs['output_names'] = kwargs.get('output_names', ['outputs'])
        kwargs['dynamic_axes'] = kwargs.get('dynamic_axes', {'inputs': {0: 'batch_size'},
                                                             'outputs': {0: 'batch_size'}})
        self.cpu()
        self.to_onnx(onnx_path, input_sample, **kwargs)


class ClassificationModelBase(ModelBase):
    ''' classification model base
    '''

    @classmethod
    def post_process(cls, outputs, multi_label):
        '''
        outputs: (numpy.ndarray) model outputs
        '''
        if isinstance(outputs, np.ndarray):
            outputs = torch.from_numpy(outputs)

        if multi_label:
            y_pred = torch.sigmoid(outputs)
        else:
            y_pred = F.softmax(outputs, dim=-1)
        return y_pred


class DetectionModelBase(ModelBase):
    ''' detection model base
    '''

    @classmethod
    def post_process(cls, outputs, x_info, classes, box_tfm=None, bias=0):
        '''
        outputs: (numpy.ndarray) model outputs
        x_info: (dict) inputs' infomation
        classes: (tuple) label set
        box_tfm: (callable)
        '''
        ypred = []
        for o, info in zip(outputs, x_info):
            boxes, labels, scores = o
            yp_bboxes = []
            yp_labels = []
            for box, label, s in zip(boxes, labels, scores):
                if box_tfm:
                    box = box_tfm(box, info['ori_shape'])
                l, t, r, b = box
                yp_bboxes.append([l, t, r, b, s])
                yp_labels.append(classes[int(label) + bias])
            yp = {'labels': yp_labels, 'bboxes': yp_bboxes}
            ypred.append(yp)
        return ypred


class SegmentationModelBase(ModelBase):
    ''' segmentation model base
    '''

    @classmethod
    def post_process(cls, outputs, x_info, classes,
                     mask_threshold=0.2, area_threshold=25,
                     blob_filter_func=None, polygon_tfm=None):
        '''
        outputs: (numpy.ndarray) model outputs
        x_info: (dict) inputs' infomation
        classes: (tuple) label set
        polygon_tfm (callable)
        mask_threshold (float) mask score threshold
        area_threshold (int) blob area threshold
        blob_filter_func (callable): return False for unwanted blob
            eg:
                def filter_blob(blob):
                    if blob.roundness < 0.5:
                        return False
        output:
            [{'labels': ['a', 'b', ...], 'polygons': [[x1,y1,x2,y2,x3,y3,...,conf], ...]}, ...]
        '''
        import cv2
        from tvlab.cv import BlobTool
        ypred = []
        for masks, info in zip(outputs, x_info):
            yp_polygons = []
            yp_labels = []
            for i, mask in enumerate(masks):
                mask = mask.squeeze(0) if mask.shape[0] == 1 else mask
                label = classes[i]
                mask = (mask * 255).astype(np.uint8)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = mask / 255.0
                bin_mask = mask > mask_threshold
                blobs = BlobTool(bin_mask * 255, mask).blobs
                for blob in blobs:
                    conf = blob.max_intensity
                    if blob.area > area_threshold:
                        if blob_filter_func and not blob_filter_func(blob):
                            continue
                        polygon = np.array(blob.contour, dtype=np.float32).flatten().tolist()
                        if polygon_tfm:
                            polygon = polygon_tfm(polygon, info['ori_shape'])
                        yp_polygons.append(polygon + [conf])
                        yp_labels.append(label)
            yp = {'labels': yp_labels, 'polygons': yp_polygons}
            ypred.append(yp)
        return ypred


class AnomalyDetectionModelBase(ModelBase):
    ''' AnomalyDetection model base
    '''

    @classmethod
    def post_process(cls, amap, v_min=0, v_max=1.0, border_coef=1.618, border_ratio=0.05):
        '''
        amap (numpy.ndarray NxHxW): model outputs
        v_min (float): clip v_min
        v_max (float: clip v_max
        border_coef (float): Image boundary weakening factor, 1.618 means significant weakening,
                                 1.0 means no weakening
        border_ratio (0.05): border_size / image_size
        out: amap (numpy.ndarry NxHxW): 0 ~ 1.0
        '''
        bw = int(amap.shape[-1] * border_ratio)
        bh = int(amap.shape[-2] * border_ratio)
        amap[:, :bh] /= np.sqrt(border_coef)
        amap[:, :2 * bh] /= np.sqrt(border_coef)
        amap[:, -bh:] /= np.sqrt(border_coef)
        amap[:, -2 * bh:] /= np.sqrt(border_coef)
        amap[:, :bw] /= np.sqrt(border_coef)
        amap[:, :, :2 * bw] /= np.sqrt(border_coef)
        amap[:, :, -bw:] /= np.sqrt(border_coef)
        amap[:, :, -2 * bw:] /= np.sqrt(border_coef)

        amap = (amap - v_min) / (v_max - v_min)
        amap = np.clip(amap, 0.0, 1.0)
        return amap


class OcrRecModelBase(ModelBase):
    ''' classification model base
    '''
    @classmethod
    def post_process(cls, outputs, multi_label):
        '''
        outputs: (numpy.ndarray) model outputs
        '''
        return

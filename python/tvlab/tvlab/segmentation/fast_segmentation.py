'''
Copyright (C) 2023 TuringVision

Image instance segmentation interface for model training and inference
'''

import os.path as osp
from .eval_segmentation import EvalSegmentation
from ..ui import bokeh_figs_to_html
from ..detection.fast_detection import (FastDetectionTrain,
                                        FastDetectionInference,
                                        get_detectron2_model_pred)

__all__ = ['FastSegmentationInference', 'FastSegmentationTrain']


def _get_evas(y_pred, y_true, iou_threshold=0.5, class_list=None, polygons_only=False):
    if polygons_only:
        y_pred = y_pred.tfm_label(lambda l: 'object')

    evas = EvalSegmentation(y_pred=y_pred, y_true=y_true,
                            iou_threshold=iou_threshold,
                            classes=class_list)
    return evas


class FastSegmentationInference(FastDetectionInference):
    ''' Detectron2 segmentation model inference
    '''
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

        _, loader = ipll.detectron2_data(self.cfg, tfms, tfms,
                                         path=self._work_dir,
                                         bs=bs, num_workers=num_workers,
                                         show_dist=False)

        class_list = self.get_class_list()
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_list)
        model = self.load_model()
        y_pred = get_detectron2_model_pred(model, loader, class_list, polygon_tfm=polygon_tfm)
        return ipll.y.__class__(y_pred)

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


class FastSegmentationTrain(FastDetectionTrain):
    ''' Detectron2 segmentation model training
    '''
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
        super().train(ipll, train_schedule=train_schedule,
                      train_tfms=train_tfms,
                      valid_tfms=valid_tfms,
                      callback=callback,
                      resume=resume)

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
        ipll = self.ibll
        train, valid = ipll.split(show=False)

        y_pred = get_detectron2_model_pred(self.trainer.model,
                                           self.loader[1],
                                           self.classes,
                                           polygon_tfm=polygon_tfm)
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

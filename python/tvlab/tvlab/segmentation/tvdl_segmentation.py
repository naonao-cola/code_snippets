'''
Copyright (C) 2023 TuringVision
'''
from ..category import TvdlTrain, TvdlInference
from os import path as osp

__all__ = ['TvdlSegmentationTrain', 'TvdlSegmentationInference']


class TvdlSegmentationTrain(TvdlTrain):
    ''' Tvdl UNet model training
    '''
    SUPPORT_SCHEDULE_KEYS = ['lr', 'bs', 'num_workers', 'monitor',
                             'epochs', 'gpus', 'check_per_epoch', 'img_c',
                             'num_layers', 'features_start', 'bilinear']

    def build_model(self):
        from tvdl.segmentation import UNet

        img_c = self._get_param('img_c', 3)
        lr = self._get_param('lr', 0.001)
        num_layers = self._get_param('num_layers', 5)
        features_start = self._get_param('features_start', 64)
        bilinear = self._get_param('bilinear', True)

        model = UNet(self.classes,
                     img_c=img_c,
                     lr=lr,
                     num_layers=num_layers,
                     features_start=features_start,
                     bilinear=bilinear)
        return model

    def train(self, ill, train_schedule, train_tfms=[], valid_tfms=[], cbs=[]):
        if 'monitor' not in train_schedule.keys():
            train_schedule["monitor"] = "val_loss"
        super(TvdlSegmentationTrain, self).train(ill, train_schedule, train_tfms, valid_tfms, cbs)

    def evaluate(self, result_path, iou_threshold=0.5,
                 mask_threshold=0.2,
                 area_threshold=25,
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

        from tvlab import bokeh_figs_to_html
        import torch
        from fastprogress.fastprogress import progress_bar
        from .fast_segmentation import _get_evas

        train, valid = self._ill.split(show=False)

        self.model.load_from_checkpoint(self._checkpoint_callback.best_model_path)
        self.model.eval()

        y_pred = []
        with torch.no_grad():
            for idx, (images, targets) in enumerate(progress_bar(self._valid_dl)):
                y_pred = y_pred + self.model.predict(images, targets,
                                                     mask_threshold=mask_threshold,
                                                     area_threshold=area_threshold,
                                                     polygon_tfm=polygon_tfm)
                if callback is not None:
                    status = {
                        'desc': "evaluate",
                        'percent': int(idx * 100 / len(self._valid_dl)),
                    }
                    callback(status)
            if callback is not None:
                status = {
                    'desc': "evaluate",
                    'percent': 100,
                }
                callback(status)

        evas = _get_evas(y_pred, valid.y, iou_threshold,
                         self.classes, polygons_only)
        fig_list = [
            self._ill.show_dist(need_show=False),
            self._ill.show_split(train, valid, need_show=False),
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


class TvdlSegmentationInference(TvdlInference):
    ''' Tvdl UNet model inference
    '''

    def predict(self, ill, tfms=None, bs=1, num_workers=1,
                mask_threshold=0.2, area_threshold=25,
                blob_filter_func=None, polygon_tfm=None):
        '''
        ill (ImagePolygonLabelList)
        tfms (list) tfm list
        bs (int) batch size
        num_works (int) works's num
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
        from tvdl.segmentation import UNet
        from fastprogress.fastprogress import progress_bar

        _, valid_dl = ill.dataloader(tfms, tfms, bs=bs, num_workers=num_workers)
        y_pred = []
        for bx, bx_info in progress_bar(valid_dl):
            if not self.use_onnx:
                bx = bx.to(self.model.device)
                if self.use_fp16:
                    bx = bx.half()
            outputs = self.model.forward(bx)
            outputs = outputs[0] if self.use_onnx else outputs.cpu().numpy()
            pp = UNet.post_process(outputs, bx_info, self.get_class_list(),
                                 mask_threshold, area_threshold,
                                 blob_filter_func, polygon_tfm)
            y_pred.extend(pp)
        return y_pred

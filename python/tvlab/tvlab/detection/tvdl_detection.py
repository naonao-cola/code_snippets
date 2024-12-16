'''
Copyright (C) 2023 TuringVision
'''

from os import path as osp

from ..category import TvdlTrain, TvdlInference

__all__ = ['TvdlDetectionTrain', 'TvdlDetectionInference']


class TvdlDetectionTrain(TvdlTrain):
    ''' Tvdl detection model training
    '''
    SUPPORT_SCHEDULE_KEYS = ['backbone', 'lr', 'bs', 'num_workers', 'monitor',
                             'epochs', 'gpus', 'check_per_epoch', 'img_c',
                             'backbone_out_indices', 'anchor_sizes', 'aspect_ratios']

    def build_model(self):
        from tvdl.detection import YOLOX

        backbone = self._get_param('backbone', 'resnet18')
        lr = self._get_param('lr', 2e-4)
        img_c = self._get_param('img_c', 3)
        backbone_out_indices = self._get_param('backbone_out_indices', (3, 4))

        model = YOLOX(classes=self.classes,
                      backbone=backbone,
                      lr=lr,
                      img_c=img_c,
                      backbone_out_indices=backbone_out_indices)
        return model

    def train(self, ill, train_schedule, train_tfms=[], valid_tfms=[], cbs=[]):
        if 'monitor' not in train_schedule.keys():
            train_schedule["monitor"] = "val_iou"
        super(TvdlDetectionTrain, self).train(ill, train_schedule, train_tfms, valid_tfms, cbs)

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
        from tvlab import bokeh_figs_to_html
        import torch
        from fastprogress.fastprogress import progress_bar

        from .fast_detection import _get_evad

        train, valid = self._ill.split(show=False)

        self.model.load_from_checkpoint(self._checkpoint_callback.best_model_path)
        self.model.eval()

        y_pred = []
        with torch.no_grad():
            for idx, (images, targets) in enumerate(progress_bar(self._valid_dl)):
                y_pred = y_pred + self.model.predict(images, targets, box_tfm)
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

        evad = _get_evad(y_pred, valid.y, iou_threshold, self.classes, bboxes_only)
        fig_list = [
            self._ill.show_dist(need_show=False),
            self._ill.show_split(train, valid, need_show=False),
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


class TvdlDetectionInference(TvdlInference):
    ''' Tvdl detection model inference
    '''

    def predict(self, ill, tfms=None, bs=1, num_workers=1, box_tfm=None):
        '''
        ill (ImageBoxLabelList)
        tfms (list) tfm list
        bs (int) batch size
        num_works (int) works's num
        box_tfm (Callable) box transform function
        output:
            [{'labels': ['a', 'b', ...], 'bboxes': [[l,t,r,b,conf], ...]}, ...]
        '''
        from tvdl.detection import YOLOX
        import torch
        from fastprogress.fastprogress import progress_bar

        _, valid_dl = ill.dataloader(tfms, tfms, bs=bs, num_workers=num_workers)
        y_pred = []
        for bx, bx_info in progress_bar(valid_dl):
            if not self.use_onnx:
                bx = bx.to(self.model.device)
                if self.use_fp16:
                    bx = bx.half()
            outputs = self.model.forward(bx)
            if self.use_onnx:
                outputs = torch.from_numpy(outputs[0])
            image_sizes = [bx.shape[-2:]] * bx.shape[0]
            pp = YOLOX.post_process(outputs, image_sizes, bx_info, self.get_class_list(), box_tfm)
            y_pred.extend(pp)
        return y_pred

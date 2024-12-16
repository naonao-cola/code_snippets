from ..category import TvdlTrain, TvdlInference

__all__ = ["TvdlAnomalyDetectionTrain", "TvdlAnomalyDetectionInference"]


class TvdlAnomalyDetectionTrain(TvdlTrain):
    SUPPORT_SCHEDULE_KEYS = ['lr', 'bs', 'num_workers', 'monitor',
                             'epochs', 'gpus', 'check_per_epoch', 'img_c',
                             'backbone', 'copy_bn', 'backbone_out_indices']

    def __init__(self, work_dir):
        super().__init__(work_dir)


    def build_model(self):
        from tvdl.anomaly_detection import FPS

        backbone = self._get_param('backbone', 'resnet18')
        lr = self._get_param('lr', 1.0)
        img_c = self._get_param('img_c', 3)
        backbone_out_indices = self._get_param('backbone_out_indices', (1, 2, 3))

        model = FPS(backbone=backbone,
                    img_c=img_c,
                    lr=lr,
                    backbone_out_indices=backbone_out_indices,
                    copy_bn=False)
        return model


class TvdlAnomalyDetectionInference(TvdlInference):

    def predict(self, ill, tfms=None, bs=16, num_workers=8, vmin=0.0, vmax=1.0, border_coef=1.618, border_ratio=0.05):
        from tvdl.anomaly_detection import FPS
        from fastprogress.fastprogress import progress_bar

        if not self.model:
            self.load_model()

        _, valid_dl = ill.dataloader(tfms, tfms, bs=bs, num_workers=num_workers)
        y_pred = []
        for bx, bx_info in progress_bar(valid_dl):
            if not self.use_onnx:
                bx = bx.to(self.model.device)
                if self.use_fp16:
                    bx = bx.half()
            outputs = self.model.forward(bx).cpu().numpy()
            outputs = outputs[0] if self.use_onnx else outputs
            pp = FPS.post_process(outputs,
                                  v_min=vmin,
                                  v_max=vmax,
                                  border_coef=border_coef,
                                  border_ratio=border_ratio)
            for yp in pp:
                y_pred.append(yp)
        return y_pred

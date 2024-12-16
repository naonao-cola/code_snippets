import os
import os.path as osp
import cv2
import numpy as np
from ..utils.capp import package_capp, unpackage_capp
from ..segmentation.polygon_label import PolygonLabelList

__all__ = ["FastOcrEnd2EndTrain", "FastOcrEnd2EndInference"]


class FastOcrEnd2EndTrain:
    def __init__(self, work_dir, det_cfg, rec_cfg=None, pretrain_dir=None):
        """
            work_dir: Workspace directory, save weight file, config file.
            det_cfg: The config file for text detection.
            rec_cfg: The config file for text recognition.
            pretrain_dir: The pretrain model weights.
        """
        from .fast_ocr_det import FastOCRDetectionTrain
        from .fast_ocr_rec import FastOCRRecognitionTrain
        if pretrain_dir is None:
            import paddleocr
            base_dir_name = os.path.dirname(paddleocr.__file__)
            pretrain_dir = osp.join(base_dir_name, "pre_models")

        self._work_dir = work_dir
        self.det_trainer = FastOCRDetectionTrain(osp.join(work_dir, "det"),
                                                det_cfg,
                                                osp.join(pretrain_dir, "det"))
        self.rec_trainer = FastOCRRecognitionTrain(osp.join(work_dir, "rec"),
                                                rec_cfg,
                                                osp.join(pretrain_dir, "rec"))

    def train(self, ocrll=None,
              train_schedule={'bs': 4, 'num_workers': 2, 'epochs': 1000, 'lr': 0.001},
              train_tfms=None, valid_tfms=None,
              callback=None, resume=False):
        if train_schedule is not None:
            if len(ocrll) <= 32:
                train_schedule["bs"] = 2
                train_schedule["num_workers"] = 1
        try:
            self.det_trainer.train(ocrll, train_schedule, train_tfms, valid_tfms, callback, resume)
            self.rec_trainer.train(ocrll, train_schedule=train_schedule)
        except KeyboardInterrupt:
            print('Trainning stopped by user request!')

    def evaluate(self):
        return self.det_trainer.evaluate()

    def package_model(self, model_path, import_cmd):
        det_path = osp.join(self._work_dir, "det_model.capp")
        rec_path = osp.join(self._work_dir, "rec_model.capp")
        self.det_trainer.package_model(det_path)
        self.rec_trainer.package_model(rec_path)
        package_capp(model_path, import_cmd, pkg_files={
            "det_model.capp": det_path,
            "rec_model.capp": rec_path,
        })

class FastOcrEnd2EndInference:
    def __init__(self, model_path, work_dir=None):
        from .fast_ocr_det import FastOCRDetectionInference
        from .fast_ocr_rec import FastOCRRecognitionInference
        self._model_info = None
        self._model_dir = unpackage_capp(model_path)

        self.det_inf = FastOCRDetectionInference(osp.join(self._model_dir, "det_model.capp"))
        self.rec_inf = FastOCRRecognitionInference(osp.join(self._model_dir, "rec_model.capp"))

    def get_rotate_crop_image(self, img, points):
        '''
        img_height, img_width = img.shape[0:2]
        left = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 0]))
        top = int(np.min(points[:, 1]))
        bottom = int(np.max(points[:, 1]))
        img_crop = img[top:bottom, left:right, :].copy()
        points[:, 0] = points[:, 0] - left
        points[:, 1] = points[:, 1] - top
        '''
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0], [img_crop_width, 0],
                              [img_crop_width, img_crop_height],
                              [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img

    def predict(self, ocrll, tfms=None, bs=1, num_workers=1, box_tfm=None):
        from fastprogress.fastprogress import progress_bar
        if len(ocrll) <= 32:
            bs = 1
            num_workers = 1

        print("Detection text")
        det_res = self.det_inf.predict(ocrll, tfms=tfms, bs=bs, num_workers=num_workers, box_tfm=box_tfm)

        fn_res = []
        print("Recognition text")
        for img_path, labels in progress_bar(zip(ocrll.x, det_res)):
            img = cv2.imread(img_path)
            img_crop_list = []
            for ply in labels["polygons"]:
                ply = ply[0:-1].copy()
                tmp_box = np.array(ply).reshape((-1, 2)).astype(np.float32)
                img_crop = self.get_rotate_crop_image(img, tmp_box)
                img_crop_list.append(img_crop)

            rec_res, predict_time = self.rec_inf.predict(img_crop_list)
            new_labels = []
            flatten_plys = []
            for its, ply in zip(rec_res, labels["polygons"]):
                if its[1] > 0.3:
                    new_labels.append(its[0])
                    ply[-1] = its[1]
                    flatten_plys.append(ply)

            fn_res.append({
                "polygons": flatten_plys,
                "labels": new_labels,
            })

        return PolygonLabelList(fn_res)

    def evaluate(self, ocrll, tfms=None, bs=1, num_workers=1, box_tfm=None):
        evals = self.det_inf.evaluate(ocrll, tfms=tfms, bs=bs, num_workers=num_workers, box_tfm=box_tfm)
        return evals

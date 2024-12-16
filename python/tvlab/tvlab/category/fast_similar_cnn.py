'''
Copyright (C) 2023 TuringVision
'''
import os, cv2
import os.path as osp
import numpy as np
import logging
from zipfile import ZipFile, ZIP_DEFLATED
import PIL, yaml, time
from uuid import getnode as get_mac

from .fast_category import FastCategoryTrain, FastCategoryInference
from .image_data import ImageLabelList
from .eval_category import EvalCategory
from ..ui import bokeh_figs_to_html

__all__ = ['FastSimilarCnnInference', 'FastSimilarCnnTrain',]

_MODEL_ACTNS_INFO_FILE = 'model_actns.pkl'
_MODEL_INFO_FILE = 'model_info.yml'

class BaseSimilarCnnInf:
    LAYER_LS_FLAG = "layer_ls"
    MODEL_DEFAULT_LAYER = [0,0]
    def img_trans_to_tensor(self, imgs, normalized = True):
        """
        func: transpose img to torch
        imgs: np.array of img data, read by np.asarray(Image.open())
        normalized: bool, normalized or not
        """
        import torch
        trans_img_list = []
        for img in imgs:
            trans_img = np.array([img.transpose((2, 0, 1))])
            trans_img = torch.from_numpy(trans_img.astype(np.float32, copy=False)).cuda()
            if normalized:
                trans_img = trans_img.div_(255)
        trans_img_list.append(trans_img)
        return trans_img_list

    def get_imgs_feature(self, img_list, model, layer_ls):
        """
        func: get eigenvalue of img
        img_list: imgs list or ImageLabelList
        model: class, model class
        layer_ls: list, the layers of model to get feature
        """
        import torch
        from fastai.callbacks.hooks import hook_output

        feature_list = []
        hook_module = model
        for l in layer_ls:
            hook_module = hook_module[l]

        with hook_output(hook_module) as hook:
            model.eval()
            dl = img_list
            with torch.no_grad():
                for (xb, yb) in dl:
                    if isinstance(xb, np.ndarray):
                        xb = self.img_trans_to_tensor([xb])[0]
                    model(xb)
                    feature_list.append((hook.stored).cpu())
            return torch.cat(feature_list).view(len(dl), -1)

    def calculate_feature_score(self, model_feature, img_feature,
                                topk_num = 5, inter_gate = 5):
        """
        func: calculate cosine similarity
        model_feature: np.array of img data, read by np.asarray(Image.open())
        img_feature: bool, normalized or not
        topk_num: int, the top feature nums used to calculate similar score
        inter_gate: int, the max interval num of valid top scores
        """
        import torch
        res_scores_list = []

        t = model_feature.float().cuda()
        v = img_feature.float().cuda()

        w = t.norm(p=2, dim=1, keepdim=True)
        wv = v.norm(p=2, dim=1, keepdim=True)
        total_scores = torch.mm(v, t.t()) / (wv * w.t()).clamp(min=1e-8)
        del t, v, w, wv
        score_sorted, indices = torch.sort(total_scores, dim=-1, descending=True)
        top_scores = score_sorted[:, :topk_num].tolist()
        indices_list = indices[:, :topk_num].tolist()
        del total_scores, score_sorted, indices

        if topk_num  == 1:
            res_scores_list = np.array(top_scores)[...,-1].tolist()
            res_scores_list = [i if i <=1 else 1 for i in res_scores_list]
        else:
            for i in range(len(top_scores)):
                idx = 0
                i_sum = 0
                for j in range(topk_num-1):
                    idx = j
                    i_sum += top_scores[i][j]
                    if top_scores[i][j+1] - top_scores[i][j] > inter_gate:
                        break

                i_score = i_sum / (idx+1)
                i_score = i_score if i_score <=1 else 1
                res_scores_list.append(i_score)

        return res_scores_list

    def evaluate(self, result_path, callback=None):
        pass

class FastSimilarCnnInference(FastCategoryInference, BaseSimilarCnnInf):
    def __init__(self, model_path, work_dir=None):
        import torch
        self._logger = logging.getLogger()
        self._model_actns = None
        super().__init__(model_path, work_dir)
        self._model_feature = self.load_model_feature()

    def _unpack_model(self, model_path):
        raise NotImplementedError("not use any more")
        import torch
        model_dir = os.path.splitext(model_path)[0]
        self._logger.debug("model_path", model_path, "model_dir", model_dir)
        os.makedirs(model_dir, exist_ok=True)

        with ZipFile(model_path, 'r') as fp:
            fp.extractall(path=model_dir)

        return model_dir

    def load_model_feature(self):
        """
        func: load the model feature file
        """
        import pickle
        model_feature = None
        model_feature_file = os.path.join(self._model_dir, _MODEL_ACTNS_INFO_FILE)
        assert os.path.exists(model_feature_file),\
                "model feature file %s not exist"%(model_feature_file)
        with open(model_feature_file, "rb") as f:
            model_feature = pickle.load(f)
        self._model_feature = model_feature
        return model_feature

    def get_learner(self):
        return self._learner

    def predict(self, imgs, tfms=None, bs=None, num_workers=None, with_loss=False):
        """
        func: get the predict conf of imgs
        imgs: ImageLabelList
        """
        imgs_feature_score = None
        if self._learner is None:
            self.load_learner(imgs, tfms, bs, num_workers)
        model_info = self.model_info()
        train_schedule = model_info['train_schedule']
        layer_ls = model_info['train_schedule'].get(self.LAYER_LS_FLAG, self.MODEL_DEFAULT_LAYER)
        imgs_feature = self.get_imgs_feature(imgs, self._learner.model, layer_ls)
        imgs_feature_score = self.calculate_feature_score(self._model_feature, imgs_feature)

        return imgs_feature_score

class FastSimilarCnnTrain(FastCategoryTrain, BaseSimilarCnnInf):
    def __init__(self, work_dir, layer_ls = [0,0]):
        super().__init__(work_dir)
        self._model_feature = None
        self._layer_ls = layer_ls
        self._normalized = True
        self.SUPPORT_SCHEDULE_KEYS.append(self.LAYER_LS_FLAG)

    def train(self, ill, train_schedule, train_tfms, valid_tfms,
        callback=None, resume_from=None, learner_cbs=None):
        """
        func: train model
        ill: ImageLabelList
        train_schedule:
            {'mixup_ratio': 0.2,
             'monitor': 'valid_loss', # one of ['valid_loss', 'accuracy'],
             'steps': [{'epochs': 20, 'lr': 0.01, 'freeze_layer': 2},
                        ...],
             'layer_ls': [0,0],
            }
        """

        super().train(ill, train_schedule, train_tfms, valid_tfms,
                callback=callback, resume_from=resume_from, learner_cbs=learner_cbs)
        self._train_schedule = train_schedule
        self.get_model_feature(reload =True)

    def get_model_feature(self, reload = False):
        """
        func: get the model feature after training
        reload: bool, reproduce the model features or not
        """
        if self._model_feature is not None and not reload:
            return self._model_feature
        model_feature = None
        self._layer_ls = self._train_schedule.get(self.LAYER_LS_FLAG, self.MODEL_DEFAULT_LAYER)
        model_feature = self.get_imgs_feature(self._ill, self._learner.model, self._layer_ls)
        self._model_feature = model_feature
        return model_feature

    def package_model(self, model_path, import_cmd, vendor_info = None):
        import pickle, os
        # 1. generate model_info.yml
        model_info = {
            'import_inferece': import_cmd,
            'classes': self._data.classes,
            'train_schedule': self._train_schedule,
            'date': time.asctime(),
            'description': 'MAC:'+str(get_mac()),
            'vendor_info': vendor_info,
        }

        with open(osp.join(self._work_dir, _MODEL_INFO_FILE), 'wt', encoding='utf-8') as fp:
            yaml.dump(model_info, fp)

        model_feature_file = os.path.join(self._work_dir, _MODEL_ACTNS_INFO_FILE)
        model_feature = self.get_model_feature()
        with open(model_feature_file, "wb") as f:
            pickle.dump(model_feature, f)

        # 2. zip vision dir
        with ZipFile(model_path, 'w', ZIP_DEFLATED) as fp:
            fp.write(osp.join(self._work_dir, _MODEL_INFO_FILE), arcname=_MODEL_INFO_FILE)
            fp.write(osp.join(self._work_dir, 'history.csv'), arcname='history.csv')
            fp.write(osp.join(self._work_dir, 'models/bestmodel.pth'), arcname='model.pth')
            fp.write(osp.join(self._work_dir, _MODEL_ACTNS_INFO_FILE), arcname=_MODEL_ACTNS_INFO_FILE)

    def get_pred(self, ill):
        """
        func: get the predict conf of ill
        ill: ImageLabelList
        """
        imgs_feature = self.get_imgs_feature(ill, self._learner.model, self._layer_ls)
        imgs_feature_score = self.calculate_feature_score(self._model_feature, imgs_feature)
        return imgs_feature_score

    def evaluate(self, result_path, callback=None, preds = None,
                 target = None, class_list = None, conf = None):
        """
        func: get EvalCategory according the input preds, target, class list
        result_path: str, the save path of output figures
        callback: func, the callback func
        preds: dict, the predict data:
            [{"OK": 0.6, "NG": 0.4}, {"OK": 0.7, "NG": 0.3}, ...]
        target: list, the true class list: [0, 0, 1, ...]
        class_list: the class list, ["OK", "NG", ...]
        conf: float, the recommand confidence threshold
        """
        total = {}
        total['precision'] = None
        total['recall'] = None
        total['f1'] = None
        self._evac = None

        if preds is not None and target is not None and class_list is not None:
            recommand_conf = conf + 0.01
            evac = EvalCategory(y_pred=preds, y_true=target,
                                classes=class_list, conf_threshold = recommand_conf)
            fig_list = [
                evac.plot_confusion_matrix(need_show=False),
                evac.plot_bokeh_table(need_show=False)
            ]
            bokeh_figs_to_html(fig_list, html_path=osp.join(result_path, 'index.html'),
                               title='Evaluate Catgory')
            evac.to_pkl(osp.join(result_path, 'evaluate.pkl'))
            total = evac.get_result()['Total']
            self._evac = evac

        return total['precision'], total['recall'], total['f1']

'''
Copyright (C) 2023 TuringVision
'''

import os
import os.path as osp
import yaml
import paddleocr
from ppocr.utils.utility import initial_logger
logger = initial_logger()
from tools.infer.predict_rec import TextRecognizer
from .program import AttrDict, preprocess, build, create_multi_devices_program, train_eval_det_run, check_gpu
from ..utils.capp import package_capp, unpackage_capp

__all__ = ["FastOCRRecognitionTrain", "FastOCRRecognitionInference"]

_MODEL_INFO_FILE = 'model_info.yml'

lang_to_model = {
    "zh": "ch_ppocr_mobile_v1.1_rec_infer",
    "ja": "japan_ppocr_mobile_v1.1_rec_infer",
    "ko": "korean_ppocr_mobile_v1.1_rec_infer",
}
lang_to_char = {
    "zh": "ppocr_keys_v1.txt",
    "ja": "japan_dict.txt",
    "ko": "korean_dict.txt",
}

class FastOCRRecognitionTrain:
    SUPPORT_SCHEDULE_KEYS = ['bs', 'num_workers', 'epochs', 'lr']

    def __init__(self, work_dir, cfg_file, model_dir):
        """
        pretrain_dir: the pretrained recognition net path.
        rec_char_dict_path: Recognition net char set file
        """
        self._work_dir = work_dir
        self._model_dir = model_dir
        os.makedirs(self._work_dir, exist_ok=True)
        config = {
            "ir_optim": True,
            "use_gpu": True,
            "gpu_mem": 6,
            "enable_mkldnn": False,
            "use_zero_copy_run": False,
            "rec_algorithm": "CRNN",
            "rec_model_dir": "",
            "rec_image_shape": "3, 32, 320",
            "rec_char_type": "ch",
            "rec_batch_num": 6,
            "max_text_length": 25,
            "rec_char_dict_path": "",
            "use_space_char": True,
            "use_pdserving": False,
        }
        pretrain_dir = osp.join(self._model_dir, "inference",  lang_to_model["zh"])
        config["rec_model_dir"]= pretrain_dir
        self._pretrain_dir = pretrain_dir

        rec_char_dict_path = osp.join(self._model_dir, "dict", lang_to_char["zh"])
        config["rec_char_dict_path"]= rec_char_dict_path
        self._rec_char_dict_path = rec_char_dict_path

        self.cfg = config
        self._train_schedule = None

    def _check_train_schedule(self, train_schedule):
        for key in train_schedule.keys():
            if key not in self.SUPPORT_SCHEDULE_KEYS:
                print('unsupport:', key)
                print('SUPPORT_SCHEDULE_KEYS:', self.SUPPORT_SCHEDULE_KEYS)
                raise KeyError(key)

    def _set_model_by_lang(self, ocrll):
        import langid
        from tqdm.auto import tqdm

        train_data, _ = ocrll.split()
        lang_count_map = {}
        number_all = 0.001
        for item in tqdm(train_data.y):
            for word in item["labels"]:
                if len(word) > 1:
                    try:
                        lang, conf = langid.classify(word)
                        if lang in lang_count_map:
                            lang_count_map[lang] += 1
                        else:
                            lang_count_map[lang] = 1
                        number_all += 1
                    except:
                        print("unkown word........")

        flag = "zh"
        if "ja" in lang_count_map and (lang_count_map["ja"] / number_all > 0.1):
            flag = "ja"
        elif "ko" in lang_count_map and (lang_count_map["ko"] / number_all > 0.1):
            flag = "ko"
        print("language: ", flag)

        pretrain_dir = osp.join(self._model_dir, "inference",  lang_to_model[flag])
        self.cfg["rec_model_dir"]= pretrain_dir
        self._pretrain_dir = pretrain_dir

        rec_char_dict_path = osp.join(self._model_dir, "dict", lang_to_char[flag])
        self.cfg["rec_char_dict_path"]= rec_char_dict_path
        self._rec_char_dict_path = rec_char_dict_path

    def train(self, ocrll=None,
                train_schedule={'bs': 4, 'num_workers': 2, 'epochs': 1000, 'lr': 0.001},
                train_tfms=None, valid_tfms=None,
                callback=None, resume=False):
        # do train schedule check
        print("Train recognition net .......................................")
        if train_schedule is not None:
            self._check_train_schedule(train_schedule)

        self._set_model_by_lang(ocrll)

    def package_model(self, model_path, import_cmd=None, **kwargs):
        ''' package capp model
        model_path: capp model path
        import_cmd: import cmd for model inference (used by adc gpu server)
        '''
        if import_cmd is None:
            import_cmd = "from tvlab import FastOCRRecognitionInference"

        ext_info = {'train_schedule': self._train_schedule}
        ext_info.update(kwargs)

        with open(osp.join(self._work_dir, 'cfg.yaml'), 'wt') as fp:
            yaml.dump(self.cfg, fp)

        pkg_files = {'cfg.yaml': osp.join(self._work_dir, 'cfg.yaml'),
                     'model': osp.join(self._pretrain_dir, 'model'),
                     'params': osp.join(self._pretrain_dir, 'params'),
                     'rec_char_dict_path': self._rec_char_dict_path}
        package_capp(model_path, import_cmd,
                     ext_info=ext_info, pkg_files=pkg_files)

class FastOCRRecognitionInference:
    def __init__(self, model_path, work_dir=None):
        self._model_info = None
        self._model_dir = unpackage_capp(model_path)
        self._work_dir = self._model_dir if work_dir is None else work_dir

        with open(osp.join(self._model_dir, 'cfg.yaml'), 'rt') as fp:
            self.cfg = yaml.load(fp, Loader=yaml.UnsafeLoader)

        self.use_gpu = self.cfg["use_gpu"]
        check_gpu(self.use_gpu)

        # Todo
        self.cfg["rec_model_dir"] = self._model_dir
        self.cfg["rec_char_dict_path"] = osp.join(self._model_dir, "rec_char_dict_path")
        self.text_recognizer = TextRecognizer(AttrDict(**self.cfg))

    def model_info(self):
        ''' load model info
        '''
        if self._model_info is None:
            with open(osp.join(self._model_dir, _MODEL_INFO_FILE), 'rt') as fp:
                self._model_info = yaml.load(fp, Loader=yaml.FullLoader)
                if isinstance(self._model_info['train_schedule'], (tuple, list)):
                    self._model_info['train_schedule'] = self._model_info['train_schedule'][0]
        return self._model_info

    def predict(self, img_list):
        rec_res, predict_time = self.text_recognizer(img_list)
        return  rec_res, predict_time

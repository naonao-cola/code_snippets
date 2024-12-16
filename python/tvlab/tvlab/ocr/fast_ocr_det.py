'''
Copyright (C) 2023 TuringVision
'''

import os
import os.path as osp
from copy import deepcopy
import yaml
import time
import numpy as np
import paddleocr
from ppocr.utils.utility import enable_static_mode
enable_static_mode()
from .program import preprocess, build, create_multi_devices_program, train_eval_det_run, check_gpu
from ..utils.capp import package_capp, unpackage_capp

__all__ = ['FastOCRDetectionTrain', 'FastOCRDetectionInference']

_MODEL_INFO_FILE = 'model_info.yml'


class FastOCRDetectionTrain:
    ''' DBnet detection model training
    '''
    SUPPORT_SCHEDULE_KEYS = ['bs', 'num_workers', 'epochs', 'lr']

    def __init__(self, work_dir, cfg_file, model_dir):
        self._work_dir = work_dir
        self._model_dir = osp.join(model_dir, "train")
        os.makedirs(self._work_dir, exist_ok=True)
        pretrain_dir = None
        startup_program, train_program, place, config, train_alg_type = preprocess(work_dir, cfg_file)
        if pretrain_dir is not None:
            config["Global"]["pretrain_weights"] = pretrain_dir

        self.startup_program = startup_program
        self.train_program = train_program
        self.place = place
        self.cfg = config
        self.train_alg_type = train_alg_type
        self.final_all_matircs = None
        self._train_schedule = None
        self._infer_model = None

    def _check_train_schedule(self, train_schedule):
        for key in train_schedule.keys():
            if key not in self.SUPPORT_SCHEDULE_KEYS:
                print('unsupport:', key)
                print('SUPPORT_SCHEDULE_KEYS:', self.SUPPORT_SCHEDULE_KEYS)
                raise KeyError(key)

    def move_pre_to_output(self):
        import shutil
        if "ResNet".lower() in self.cfg["Backbone"]["function"].lower():
            from_dir = osp.join(self._model_dir, 'ch_ppocr_server_v1.1_det_train')
        else:
            from_dir = osp.join(self._model_dir, 'ch_ppocr_mobile_v1.1_det_train')
        flist = os.listdir(from_dir)
        self.cfg["Global"]["checkpoints"] = osp.join(from_dir, "best_accuracy")
        for f in flist:
            src = os.path.join(from_dir, f)
            dst = os.path.join(self._work_dir, f)
            shutil.copyfile(src, dst)

    def train(self, ocrll=None,
              train_schedule={'bs': 4, 'num_workers': 2, 'epochs': 1000, 'lr': 0.001},
              train_tfms=None, valid_tfms=None,
              callback=None, resume=False):
        from paddle import fluid
        if train_schedule is None:
            self.move_pre_to_output()
            train_schedule={'bs': 1, 'num_workers': 1, 'epochs': 0, 'lr': 0.001}

        # do train schedule check
        self._check_train_schedule(train_schedule)
        if "epochs" in train_schedule:
            self.cfg["Global"]["epoch_num"] = train_schedule["epochs"]
        if "lr" in train_schedule:
            self.cfg["Optimizer"]["base_lr"] = train_schedule["lr"]

        train_dataset, valid_dataset = ocrll.dbnet_data(self.cfg, train_tfms, valid_tfms,
                                      bs=train_schedule['bs'],
                                      num_workers=train_schedule['num_workers'],
                                      show_dist=False)
        # build train program
        train_build_outputs = build(self.cfg, self.train_program,
            self.startup_program, mode='train')
        train_loader = train_build_outputs[0]
        train_fetch_name_list = train_build_outputs[1]
        train_fetch_varname_list = train_build_outputs[2]
        train_opt_loss_name = train_build_outputs[3]
        model_average = train_build_outputs[-1]

        # build eval program
        eval_program = fluid.Program()
        eval_build_outputs = build(self.cfg, eval_program,
            self.startup_program, mode='eval')
        eval_fetch_name_list = eval_build_outputs[1]
        eval_fetch_varname_list = eval_build_outputs[2]
        eval_program = eval_program.clone(for_test=True)

        train_loader.set_sample_list_generator(train_dataset, places=self.place)
        exe = fluid.Executor(self.place)
        exe.run(self.startup_program)

        # compile program for multi-devices
        train_compile_program = create_multi_devices_program(
            self.train_program, train_opt_loss_name)

        # dump mode structure
        if self.cfg['Global']['debug']:
            if self.train_alg_type == 'rec' and 'attention' in self.cfg['Global'][
                    'loss_type']:
                print('Does not suport dump attention...')
            else:
                from paddle.fluid.contrib.model_stat import summary
                summary(self.train_program)

        try:
            from ppocr.utils.save_load import init_model
            init_model(self.cfg, self.train_program, exe)
        except:
            print("""
            ==============================================================================================
            error: no pretrainded model
                if Not load pretrain models, please load models at first:
            """)

        train_info_dict = {'compile_program':train_compile_program,\
                'train_program':self.train_program,\
                'reader':train_loader,\
                'fetch_name_list':train_fetch_name_list,\
                'fetch_varname_list':train_fetch_varname_list,\
                'model_average': model_average}
        eval_info_dict = {'program':eval_program,\
            'reader':valid_dataset,\
            'fetch_name_list':eval_fetch_name_list,\
            'fetch_varname_list':eval_fetch_varname_list}

        self.exe = exe
        self.eval_info_dict = eval_info_dict
        if self.train_alg_type == 'det':
            self.final_all_matircs = train_eval_det_run(self.cfg, exe, train_info_dict, eval_info_dict, callback=callback)

    def evaluate(self):
        from .program import eval_det_run
        evals, _ = eval_det_run(self.exe, self.cfg, self.eval_info_dict, "eval")
        return evals

    def package_model(self, model_path, import_cmd=None, **kwargs):
        ''' package capp model
        model_path: capp model path
        import_cmd: import cmd for model inference (used by adc gpu server)
        '''
        if import_cmd is None:
            import_cmd = "from tvlab import FastOCRDetectionInference"

        ext_info = {'train_schedule': self._train_schedule}
        ext_info.update(kwargs)

        with open(osp.join(self._work_dir, 'cfg.yaml'), 'wt') as fp:
            yaml.dump(self.cfg, fp)

        pkg_files = {'cfg.yaml': osp.join(self._work_dir, 'cfg.yaml'),
                     'model.pdparams': osp.join(self._work_dir, 'best_accuracy.pdparams')}
        package_capp(model_path, import_cmd,
                     ext_info=ext_info, pkg_files=pkg_files)


class FastOCRDetectionInference:
    def __init__(self, model_path, work_dir=None):
        self._model_info = None
        self._model_dir = unpackage_capp(model_path)
        self._work_dir = self._model_dir if work_dir is None else work_dir

        with open(osp.join(self._model_dir, 'cfg.yaml'), 'rt') as fp:
            self.cfg = yaml.load(fp, Loader=yaml.UnsafeLoader)

        self.use_gpu = self.cfg["Global"]["use_gpu"]
        self.checkpoints = osp.join(self._model_dir, "model.pdparams")
        check_gpu(self.use_gpu)

    def model_info(self):
        ''' load model info
        '''
        if self._model_info is None:
            with open(osp.join(self._model_dir, _MODEL_INFO_FILE), 'rt') as fp:
                self._model_info = yaml.load(fp, Loader=yaml.FullLoader)
                if isinstance(self._model_info['train_schedule'], (tuple, list)):
                    self._model_info['train_schedule'] = self._model_info['train_schedule'][0]
        return self._model_info

    def _inference(self, ocrll, tfms=None, bs=None, num_workers=None, box_tfm=None):
        from paddle import fluid
        from ppocr.utils.utility import create_module
        from fastprogress.fastprogress import progress_bar
        model_info = self.model_info()
        if bs is None:
            bs = model_info['train_schedule']['bs']
            bs = min(len(ocrll), bs)

        if num_workers is None:
            num_workers = model_info['train_schedule']['num_workers']

        place = fluid.CUDAPlace(0) if self.use_gpu else fluid.CPUPlace()
        exe = fluid.Executor(place)

        det_model = create_module(self.cfg['Architecture']['function'])(params=self.cfg)


        startup_prog = fluid.Program()
        eval_prog = fluid.Program()
        with fluid.program_guard(eval_prog, startup_prog):
            with fluid.unique_name.guard():
                _, eval_outputs = det_model(mode="test")
                fetch_name_list = list(eval_outputs.keys())
                eval_fetch_list = [eval_outputs[v].name for v in fetch_name_list]

        eval_prog = eval_prog.clone(for_test=True)
        exe.run(startup_prog)

        # load checkpoints
        if self.checkpoints:
            path = self.checkpoints
            fluid.load(eval_prog, path, exe)
        else:
            raise Exception("{} not exists!".format(self.checkpoints))

        ocrll.split(valid_pct=1.0)
        _, test_reader = ocrll.dbnet_data(self.cfg, tfms, tfms,
                                         bs=bs, num_workers=num_workers,
                                         show_dist=False)
        tackling_num = 0
        result = []
        for data in progress_bar(test_reader()):
            img_num = len(data)
            tackling_num = tackling_num + img_num
            img_list = []
            ratio_list = []
            gt_list = []
            for ino in range(img_num):
                img_list.append(data[ino][0])
                ratio_list.append(data[ino][1])
                gt_list.append(data[ino][2])

            img_list = np.concatenate(img_list, axis=0)
            outs = exe.run(eval_prog,\
                feed={'image': img_list},\
                fetch_list=eval_fetch_list)

            global_params = self.cfg['Global']
            postprocess_params = deepcopy(self.cfg["PostProcess"])
            postprocess_params.update(global_params)
            postprocess = create_module(postprocess_params['function'])\
                (params=postprocess_params)
            if self.cfg['Global']['algorithm'] == 'EAST':
                dic = {'f_score': outs[0], 'f_geo': outs[1]}
            elif self.cfg['Global']['algorithm'] == 'DB':
                from .db_process import DBPostProcessTS
                postprocess = DBPostProcessTS(params=postprocess_params)
                dic = {'maps': outs[0]}
            elif self.cfg['Global']['algorithm'] == 'SAST':
                dic = {
                    'f_score': outs[0],
                    'f_border': outs[1],
                    'f_tvo': outs[2],
                    'f_tco': outs[3]
                }
            else:
                raise Exception(
                    "only support algorithm: ['EAST', 'DB', 'SAST']")
            dt_boxes_list, dt_scores_list = postprocess(dic, ratio_list)
            for ino in range(img_num):
                dt_boxes = dt_boxes_list[ino]
                dt_scores = dt_scores_list[ino]
                item = {'labels': [], 'polygons': [],}
                for idx, box in enumerate(dt_boxes):
                    item["labels"].append("")
                    tpp = np.array(box).reshape((-1, )).tolist()
                    tpp.append(dt_scores[idx])
                    item["polygons"].append(tpp)
                result.append(item)

        return result

    def predict(self, ocrll, tfms=None, bs=None, num_workers=None, box_tfm=None):
        result = self._inference(ocrll, tfms, bs, num_workers, box_tfm)
        return ocrll.y.__class__(result)

    def evaluate(self, ocrll, tfms=None, bs=None, num_workers=None, box_tfm=None):
        from ..segmentation.eval_segmentation import EvalSegmentation
        ypred = self._inference(ocrll, tfms, bs, num_workers, box_tfm)
        ytrue = ocrll.y.tfm_label(lambda l: 'object')
        evals = EvalSegmentation(ypred, ytrue, polygons_only=True)
        return evals

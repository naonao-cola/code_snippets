'''
Copyright (C) 2023 TuringVision

Multi-task image category interface for model training and inference
'''

from .fast_category import FastCategoryTrain, FastCategoryInference

__all__ = ['MultiTaskCategoryInference', 'MultiTaskCategoryTrain']

class MultiTaskCategoryInference(FastCategoryInference):
    def _load_learner(self, data):
        from fastai.vision import models
        from .multi_task_learner import create_multi_task_learner

        if self._learner is None:
            # 1. load model info
            model_info = self.model_info()
            train_schedule = model_info['train_schedule']

            basemodel = train_schedule['basemodel']
            if isinstance(basemodel, str):
                base_arch = eval('models.'+basemodel)
            elif isinstance(basemodel, (tuple, list)) and isinstance(basemodel[0], str):
                base_arch = list()
                for m in basemodel:
                    base_arch.append(eval('models.'+m))

            neck_nf = train_schedule.get('neck_nf', 512)
            num_groups = train_schedule.get('num_groups', 3)
            loss_weight = train_schedule.get('loss_weight', None)
            if loss_weight is None:
                loss_weight = train_schedule.get('loss_weights', None)
            sigmoid = train_schedule.get('sigmoid', False)

            learner = create_multi_task_learner(data, base_arch, neck_nf=neck_nf,
                                                num_groups=num_groups,
                                                sigmoid=sigmoid,
                                                loss_weight=loss_weight)

            # 2. create learner (need basemodel)
            learner = learner.load(open(model_info['model_path'], 'rb'))
            self._learner = learner
        else:
            self._learner.data = data
        return self._learner


class MultiTaskCategoryTrain(FastCategoryTrain):
    SUPPORT_SCHEDULE_KEYS = ['basemodel', 'neck_nf', 'num_groups',
                             'loss_weight', 'loss_weights', 'sigmoid',
                             'label_smoothing',
                             'bs', 'num_workers', 'mixup_ratio',
                             'optimizer', 'monitor', 'wd', 'class_weight',
                             'bn_final', 'steps', 'batch_n_cls',
                             'apinet', 'loss_rk_coeff', 'loss_p_margin']

    def train(self, ill, train_schedule, train_tfms, valid_tfms,
              callback=None, resume_from=None, learner_cbs=None):
        '''
        ill: ImageLabelList or ImageMultiLabelList
        train_tfms: list of transform function for train dataset
        valid_tfms: list of transform function for valid dataset
        train_schedule:
            {'basemodel': ['densenet161', 'resnet18'] or 'densenet161',
             'neck_nf': [384, 128] or 512,
             'num_groups': 3,
             'loss_weight': [0.1, 0,9],
             'bs': 16, 'num_workers':1, 'mixup_ratio': 0.2,
             'optimizer': 'adam', # one of ['adam', 'rms', 'sgd']
             'sigmoid': False, # when sigmoid is True, convert the classification task to a regression task (0 ~ 1.0)
                               # labelset: ['defect_level_0', 'defect_level_1',   ... , 'defect_level_n']
                               # model target:     0.0      ,        1/n      ,   ... ,      1.0
                               #
                               # If the labelset of the dataset is missing some level, the model target may not be as expected
                               # when ill.labelset is: ['defect_level_0', 'defect_level_2', 'defect_level_3'] # level_0 not in current dataset
                               #         model target:       0.0        ,        0.5      ,       1.0
                               #                                                 ---
                               # We need force set labelset to all levels
                               # ill.labelset = lambda : ['defect_level_0', 'defect_level_1', 'defect_level_2', 'defect_level_3']
                               #        model target is:        0.0       ,       0.33      ,      0.66       ,       1.0
                               #                                                                   ----
             'label_smoothing': Flase,
             'monitor': 'valid_loss', # one of ['valid_loss', 'accuracy']
             'wd': 1e-2,
             'class_weight':[{'NON': 1, 'Other': 10}, {'Particle': 10, 'Damage': 20}]
             'batch_n_cls': 8,
             'apinet': False,
             'loss_rk_coeff': 1.0, # used by apinet
             'loss_p_margin': 0.05, # used by apinet
             'steps': [{
                    'epochs': 20, 'lr': 0.01,
                    'freeze_layer': 2
                }, ...]
            }
        '''
        from fastai.vision import models
        from functools import partial
        from torch import optim
        from .multi_task_learner import create_multi_task_learner

        bs = train_schedule['bs']
        batch_n_cls = train_schedule.get('batch_n_cls', None)
        batch_sampler = None
        if batch_n_cls:
            from .batch_sampler import PairBatchSampler

            n_img = max(1, bs//batch_n_cls)
            n_cls = bs // n_img
            batch_sampler = partial(PairBatchSampler, n_img=n_img, n_cls=n_cls)

        self._check_train_schedule(train_schedule)
        data = ill.databunch(train_tfms=train_tfms,
                             valid_tfms=valid_tfms,
                             path=self._work_dir,
                             bs=bs,
                             num_workers=train_schedule['num_workers'],
                             batch_sampler=batch_sampler,
                             show_dist=False)

        opt = train_schedule.get('optimizer', 'adam')
        mom = 0.9
        alpha = 0.99
        eps = 1e-6

        if opt == 'rms':
            opt_func = partial(optim.RMSprop, alpha=alpha, eps=eps)
        elif opt == 'sgd':
            opt_func = partial(optim.SGD, momentum=mom)
        else:
            opt_func = partial(optim.Adam, betas=(mom, alpha), eps=eps)

        wd = train_schedule.get('wd', 1e-2)
        sigmoid = train_schedule.get('sigmoid', False)

        basemodel = train_schedule['basemodel']
        if isinstance(basemodel, str):
            base_arch = eval('models.'+basemodel)
        elif isinstance(basemodel, (tuple, list)) and isinstance(basemodel[0], str):
            base_arch = list()
            for m in basemodel:
                base_arch.append(eval('models.'+m))

        neck_nf = train_schedule.get('neck_nf', 512)
        num_groups = train_schedule.get('num_groups', 3)
        loss_weight = train_schedule.get('loss_weight', None)
        if loss_weight is None:
            loss_weight = train_schedule.get('loss_weights', None)
        class_weight = train_schedule.get('class_weight', None)

        apinet = train_schedule.get('apinet', False)
        if not apinet:
            batch_n_cls = 0

        loss_rk_coeff = train_schedule.get('loss_rk_coeff', 1.0)
        loss_p_margin = train_schedule.get('loss_p_margin', 0.05)
        label_smoothing = train_schedule.get('label_smoothing', False)

        learner = create_multi_task_learner(data, base_arch, neck_nf=neck_nf,
                                            num_groups=num_groups,
                                            loss_weight=loss_weight,
                                            class_weight=class_weight,
                                            wd=wd, train_bn=True,
                                            sigmoid=sigmoid,
                                            batch_n_cls=batch_n_cls,
                                            loss_rk_coeff=loss_rk_coeff,
                                            loss_p_margin=loss_p_margin,
                                            label_smoothing=label_smoothing,
                                            opt_func=opt_func)
        self.train_from_learner(ill, learner, train_schedule, callback, resume_from, learner_cbs)

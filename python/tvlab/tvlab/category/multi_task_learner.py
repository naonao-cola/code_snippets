'''
Copyright (C) 2023 TuringVision

create learner for multi category task.
'''
import numpy as np
from torch import nn

__all__ = ['create_multi_task_learner']


def create_mini_head(nf, nc, sigmoid=False):
    from fastai.vision import bn_drop_lin
    if sigmoid:
        nc = 1
    layers = []
    layers += bn_drop_lin(nf, nc, True, 0.5, None)
    if not sigmoid:
        layers.append(nn.BatchNorm1d(nc, momentum=0.01))
    else:
        layers.append(nn.Flatten(-2))
    return nn.Sequential(*layers)


def create_backbone(arch, pretrained):
    from fastai.vision import models
    _default_meta    = {'cut':None}
    _resnet_meta     = {'cut':-2}
    _squeezenet_meta = {'cut':-1}
    _densenet_meta   = {'cut':-1}
    _vgg_meta        = {'cut':-1}
    _alexnet_meta    = {'cut':-1}

    model_meta = {
      models.resnet18 :{**_resnet_meta}, models.resnet34: {**_resnet_meta},
      models.resnet50 :{**_resnet_meta}, models.resnet101:{**_resnet_meta},
      models.resnet152:{**_resnet_meta},

      models.squeezenet1_0:{**_squeezenet_meta},
      models.squeezenet1_1:{**_squeezenet_meta},

      models.densenet121:{**_densenet_meta}, models.densenet169:{**_densenet_meta},
      models.densenet201:{**_densenet_meta}, models.densenet161:{**_densenet_meta},
      models.vgg16_bn:{**_vgg_meta}, models.vgg19_bn:{**_vgg_meta},
      models.alexnet:{**_alexnet_meta}}

    def cnn_config(arch):
        return model_meta.get(arch, _default_meta)

    backbone = arch(pretrained)
    backbone = nn.Sequential(*list(backbone.children()))
    meta = cnn_config(arch)
    cut = meta['cut']
    if cut:
        backbone = backbone[:cut]
    return backbone


class MultiTaskModel(nn.Module):
    def __init__(self, base_arch, neck_nf, out_c, pretrained,
                 batch_n_cls=None, sigmoid=False,
                 init=nn.init.kaiming_normal_):
        from fastai.vision import create_head
        from fastai.callbacks.hooks import num_features_model
        super().__init__()

        self.batch_n_cls = batch_n_cls
        self._is_multi_input = False
        self._is_multi_task = False

        if isinstance(base_arch, (tuple, list)):
            self._is_multi_input = True
            self._backbone = [create_backbone(arch, pretrained) for arch in base_arch]
            backbone_nf = [num_features_model(nn.Sequential(*body.children())) * 2
                           for body in self._backbone]

            neck_nf = neck_nf if isinstance(base_arch, (tuple, list)) else [neck_nf] * len(base_arch)

            self._neck = [create_head(in_nf, out_nf, lin_ftrs=[])
                          for in_nf, out_nf in zip(backbone_nf, neck_nf)]

            for i in range(len(base_arch)):
                setattr(self, 'backbone_'+str(i), self._backbone[i])
                setattr(self, 'neck_'+str(i), self._neck[i])

            nf = sum(neck_nf)
        else:
            self._backbone = create_backbone(base_arch, pretrained)
            backbone_nf = num_features_model(nn.Sequential(*self._backbone.children())) * 2
            self._neck = create_head(backbone_nf, neck_nf, lin_ftrs=[])
            nf = neck_nf

        self._activ = nn.ReLU(inplace=True)

        if isinstance(out_c, (tuple, list)):
            self._is_multi_task = True
            self._head = [create_mini_head(nf, nc, sigmoid) for nc in out_c]
            for i in range(len(out_c)):
                setattr(self, 'head_'+str(i), self._head[i])
        else:
            self._head = create_mini_head(nf, out_c, sigmoid)

        self._init_weights(init)

    def _init_weights(self, init):
        from fastai.vision import apply_init
        if self._is_multi_input:
            for neck in self._neck:
                apply_init(neck, init)
        else:
            apply_init(self._neck, init)

        if self._is_multi_task:
            for head in self._head:
                apply_init(head, init)
        else:
            apply_init(self._head, init)

    def split_groups(self, num_groups=3):
        from fastai.vision import flatten_model
        num_groups = max(3, num_groups)
        if self._is_multi_input:
            backbone_groups = list()
            head_layers = list()
            for backbone, neck in zip(self._backbone, self._neck):
                backbone_layers = flatten_model(backbone)
                idxs = np.linspace(0, len(backbone_layers), num_groups, dtype=np.int0)
                groups = [backbone_layers[i:j] for i,j in zip(idxs[:-1], idxs[1:])]
                backbone_groups.append(groups)
                head_layers.append(neck)
            backbone_groups = [[l for group in groups for l in group] for groups in zip(*backbone_groups)]
            layer_groups = [nn.Sequential(*group) for group in backbone_groups]
        else:
            layers = flatten_model(self._backbone)
            idxs = np.linspace(0, len(layers), num_groups, dtype=np.int0)
            layer_groups = [nn.Sequential(*layers[i:j]) for i,j in zip(idxs[:-1],idxs[1:])]
            head_layers = [self._neck]

        head_layers.append(self._activ)

        if self._is_multi_task:
            for head in self._head:
                head_layers.append(head)
        else:
            head_layers.append(self._head)

        layer_groups.append(nn.Sequential(*head_layers))
        return layer_groups

    def forward(self, *x):
        import torch
        if self._is_multi_input:
            x = [backbone(xi) for backbone, xi in zip(self._backbone, x)]
            x = [neck(xi) for neck, xi in zip(self._neck, x)]
            x = torch.cat(x, dim=1)
        else:
            x = self._backbone(x[0])
            x = self._neck(x)

        x = self._activ(x)

        if self._is_multi_task:
            out = tuple([head(x) for head in self._head])
        else:
            if self.batch_n_cls and self.training:
                from .api_net import api_net_training_forward
                out = api_net_training_forward(x, self.batch_n_cls)
                x1self, x2self, x1other, x2other, x1_idxs, x2_idxs = out
                h1self = self._head(x1self)
                h2self = self._head(x2self)
                h1other = self._head(x1other)
                h2other = self._head(x2other)
                out = (h1self, h2self, h1other, h2other, x1_idxs, x2_idxs)
            else:
                out = self._head(x)
        return out


def multi_task_loss(predict, *target, loss_func, loss_weight=None):
    total_loss = None
    loss_weight = loss_weight if loss_weight else [1.0] * len(predict)
    for pred, targs, loss, weight in zip(predict, target, loss_func, loss_weight):
        one_loss = loss(pred, targs) * weight
        if total_loss:
            total_loss += one_loss
        else:
            total_loss = one_loss
    return total_loss


def accuracy(predict, *target):
    from fastai.metrics import accuracy as one_task_accuracy

    acc = None
    for pred, targs in zip(predict, target):
        one_acc = one_task_accuracy(pred, targs)
        if acc:
            acc += one_acc
        else:
            acc = one_acc
    return acc / len(predict)


def accuracy_n(predict, *target, n=0):
    from fastai.metrics import accuracy as one_task_accuracy
    return one_task_accuracy(predict[n], target[n])


def get_all_task_accuracy_func(task_num):
    from functools import partial

    acc_funcs = [accuracy]
    for i in range(task_num):
        acc_func = partial(accuracy_n, n=i)
        acc_func.__name__ = 'accuracy_' + str(i)
        acc_funcs.append(acc_func)
    return acc_funcs


def one_task_sigmoid_loss(predict, target, max_level=1, label_smoothing=False):
    import torch
    from torch.nn.functional import binary_cross_entropy_with_logits

    # convert target to 0 ~ 1
    with torch.no_grad():
        target = target.type(torch.float32)
        target = target / max_level
        if label_smoothing:
            # label smoothing
            target += torch.randn_like(target)/((max_level+1)*8)
            target = torch.clamp(target, 0.0, 1.0)
    return binary_cross_entropy_with_logits(predict, target)


def create_multi_task_learner(data, base_arch, neck_nf=512,
                              pretrained=True, num_groups=3,
                              init=nn.init.kaiming_normal_,
                              loss_weight=None,
                              class_weight=None,
                              sigmoid=False,
                              batch_n_cls=None,
                              loss_rk_coeff=1.0,
                              loss_p_margin=0.05,
                              label_smoothing=False,
                              **kwargs):
    import torch
    from fastai.vision import Learner, CrossEntropyFlat, LabelSmoothingCrossEntropy
    from functools import partial
    from fastai.metrics import accuracy

    # create model
    model = MultiTaskModel(base_arch=base_arch, neck_nf=neck_nf,
                           out_c=data.c, pretrained=pretrained,
                           batch_n_cls=batch_n_cls,
                           sigmoid=sigmoid,
                           init=init)
    layer_groups = model.split_groups(num_groups)

    if isinstance(data.c, (tuple, list)):
        metrics = get_all_task_accuracy_func(len(data.c))

        all_loss_func = list()
        for i, classes in enumerate(data.classes):
            if sigmoid:
                loss = partial(one_task_sigmoid_loss, max_level=len(classes) -1,
                               label_smoothing=label_smoothing)
            else:
                if label_smoothing:
                    loss = LabelSmoothingCrossEntropy()
                else:
                    weight = torch.ones(len(classes), dtype=torch.float32).to('cuda')
                    if class_weight:
                        for key, value in class_weight[i].items():
                            weight[classes.index(key)] = value
                    loss = CrossEntropyFlat(weight=weight)
            all_loss_func.append(loss)

        loss_func = partial(multi_task_loss,
                            loss_func=all_loss_func,
                            loss_weight=loss_weight)
    else:
        metrics = [accuracy]
        if sigmoid:
            loss_func = partial(one_task_sigmoid_loss, max_level=data.c -1,
                                label_smoothing=label_smoothing)
        else:
            if label_smoothing:
                loss_func = LabelSmoothingCrossEntropy()
            else:
                weight = torch.ones(len(data.classes), dtype=torch.float32).to('cuda')
                if class_weight:
                    for key, value in class_weight.items():
                        weight[data.classes.index(key)] = value
                loss_func = CrossEntropyFlat(weight=weight)
        if batch_n_cls:
            from .api_net import api_net_loss
            loss_func = partial(api_net_loss, ce_loss_func=loss_func,
                                loss_rk_coeff=loss_rk_coeff,
                                loss_p_margin=loss_p_margin)

    if sigmoid:
        metrics = []

    if 'metrics' in kwargs:
        kwargs['metrics'] = metrics + kwargs['metrics']
    else:
        kwargs['metrics'] = metrics

    # create learner
    learner = Learner(data, model, layer_groups=layer_groups,
                      loss_func=loss_func, **kwargs)
    if pretrained: learner.freeze()

    return learner

'''
Copyright (C) 2023 TuringVision

timm classification backbones
'''
import torch
from torch import nn

__all__ = ['TimmBackbone']


def is_pool_type(l):
    import re
    return re.search(r'Pool[123]d$', l.__class__.__name__)


def has_pool_type(m):
    if is_pool_type(m): return True
    for l in m.children():
        if has_pool_type(l): return True
    return False


class AdaptiveConcatPool2d(nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`."
    def __init__(self, sz=None):
        super().__init__()
        "Output will be 2*sz or 2 if sz is None"
        self.output_size = sz or 1
        self.ap = nn.AdaptiveAvgPool2d(self.output_size)
        self.mp = nn.AdaptiveMaxPool2d(self.output_size)

    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)


class TimmBackbone(nn.Module):
    def __init__(self, model_name, in_chans=3, pretrained=True, concat_pool=True, **kwargs):
        """
        Args:
            model_name (str): name of model to instantiate
            in_chans (int): default 3
                Number of input (color) channels.
                eg: 1-channel (gray image); 25-channel image (maybe satellite image)
            pretrained (bool): load pretrained ImageNet-1k weights if true.
            concat_pool (bool):
                whether to concat avgpool and maxpool or not.
        Keyword Args:
            drop_rate (float): dropout rate for training (default: 0.0)
            global_pool (str): global pool type (default: 'avg')
            **: other kwargs are model specific
        """
        super().__init__()
        import timm
        self.model = timm.create_model(model_name, in_chans=in_chans, \
            pretrained=pretrained, num_classes=0, **kwargs)
        self.pool = AdaptiveConcatPool2d() if concat_pool else nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x =  self.model.forward_features(x)
        return self.flatten(self.pool(x))

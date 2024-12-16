'''
Copyright (C) 2023 TuringVision

timm detection backbones
'''
from torch import nn

__all__ = ['TimmBackboneWithFPN']


class TimmBackboneWithFPN(nn.Module):
    def __init__(self, model_name, in_chans=3, out_indices=(1, 2, 3, 4),
                 out_channels=256, fpn_last_maxpool=True,
                 pretrained=True, **kwargs):
        """
        Args:
            model_name (str): name of model to instantiate
            in_chans (int): default 3
                Number of input (color) channels.
                eg: 1-channel (gray image); 25-channel image (maybe satellite image)
            pretrained (bool): load pretrained ImageNet-1k weights if true.
            out_channels (int): fpn output feature_map channels
            fpn_last_maxpool (bool): True or False，
                whether fpn with LastLevelMaxPool() or not.
        Keyword Args:
            drop_rate (float): dropout rate for training (default: 0.0)
            global_pool (str): global pool type (default: 'avg')
            **: other kwargs are model specific
        """
        super().__init__()
        import timm
        from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool

        self.backbone = timm.create_model(model_name,
                                          features_only=True,
                                          in_chans=in_chans,
                                          pretrained=pretrained,
                                          out_indices=out_indices,
                                          **kwargs)
        extra_blocks = LastLevelMaxPool() if fpn_last_maxpool else None
        self.fpn = FeaturePyramidNetwork(self.backbone.feature_info.channels(),
                                         out_channels,
                                         extra_blocks)
        self.out_channels = out_channels
        self.featmap_names = [str(i) for i in range(len(out_indices))]
        self.featmap_reductions = [self.backbone.feature_info.info[i]['reduction'] for i in out_indices]
        if fpn_last_maxpool:
            self.featmap_names += ['pool']
            self.featmap_reductions += [int(self.featmap_reductions[-1]*2)]

    def forward(self, x):
        x = self.backbone(x)
        dict_x = {}
        for i, xi in enumerate(x):
            dict_x[str(i)] = xi
        o = self.fpn(dict_x)
        return o

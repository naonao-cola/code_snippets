import torch
from torch import nn

__all__ = ['build_feature_model']

_MODEL_FV_INFO = {
    'densenet161': {8: [6, 7], 16: [8, 9], 32: [10, 11]},
}


class TvFeatureExtractor(nn.Module):
    def __init__(self, basemodel, fv_rf_s):
        import torchvision
        assert basemodel in _MODEL_FV_INFO
        super().__init__()
        self.fv_group_info = _MODEL_FV_INFO[basemodel]
        self.fv_rf_s = sorted(fv_rf_s)
        if basemodel == 'densenet161':
            model = torchvision.models.densenet161(pretrained=True)
            model = nn.Sequential(*list(model.children()))[0]
        net_s = []
        s = 0
        for fv in self.fv_rf_s:
            end_s = self.fv_group_info[fv]
            for end_i in end_s:
                net_s.append(model[s:end_i])
                s = end_i
        self.net_s = net_s
        self.net = nn.Sequential(*net_s)

    def forward(self, x):
        import torch
        out = dict()
        i = 0
        for fv in self.fv_rf_s:
            for _ in self.fv_group_info[fv]:
                x = self.net_s[i](x)
                if fv not in out:
                    out[fv] = [x]
                else:
                    out[fv].append(x)
                i += 1

        for k, v in out.items():
            if len(v) > 1:
                out[k] = torch.cat(v, dim=1)
            else:
                out[k] = v[0]
        return out


class TimmFeatureExtractor(nn.Module):
    def __init__(self, basemodel, fv_rf_s):
        super().__init__()
        import timm
        fv_idx = [2, 4, 8, 16, 32]
        fv_rf_s = sorted(fv_rf_s)
        out_indices = [fv_idx.index(fv) for fv in fv_rf_s]
        self.m = timm.create_model(basemodel, features_only=True, pretrained=True,
                                   out_indices=out_indices)
        self.fv_rf_s = fv_rf_s

    def forward(self, x):
        o = self.m(x)
        out = dict()
        for fv, oi in zip(self.fv_rf_s, o):
            out[fv] = oi
        return out


def build_feature_model(basemodel='densenet161', fv_rf_s=[16]):
    '''
    In:
        basemodel (str): basemodel name
        fv_rf_s (list): list of output feature level
    Out:
        feature_extractor (nn.Module)
    '''
    if basemodel in _MODEL_FV_INFO:
        return TvFeatureExtractor(basemodel, fv_rf_s)
    return TimmFeatureExtractor(basemodel, fv_rf_s)

'''
Copyright (C) 2023 TuringVision

Basic tools for classification.
'''
from torch import nn

__all__ = ['num_features_model', 'freeze', 'flatten_model']

BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)

def children(m:nn.Module):
    "Get children of `m`."
    return list(m.children())


def num_children(m:nn.Module)->int:
    "Get number of children modules in `m`."
    return len(children(m))


class ParameterModule(nn.Module):
    "Register a lone parameter `p` in a module."
    def __init__(self, p:nn.Parameter):
        super().__init__()
        self.val = p

    def forward(self, x): return x


def children_and_parameters(m:nn.Module):
    "Return the children of `m` and its direct parameters not registered in modules."
    children = list(m.children())
    children_p = sum([[id(p) for p in c.parameters()] for c in m.children()],[])
    for p in m.parameters():
        if id(p) not in children_p: children.append(ParameterModule(p))
    return children


flatten_model = lambda m: sum(map(flatten_model,children_and_parameters(m)),[]) if num_children(m) else [m]


def one_param(m: nn.Module):
    "Return the first parameter of `m`."
    return next(m.parameters())


def in_channels(m:nn.Module):
    "Return the shape of the first weight layer in `m`."
    for l in flatten_model(m):
        if hasattr(l, 'weight'): return l.weight.shape[1]
    raise Exception('No weight layer')


def dummy_batch(m: nn.Module, size:tuple=(64,64)):
    "Create a dummy batch to go through `m` with `size`."
    ch_in = 3
    try:
        ch_in = in_channels(m)
    except Exception as e:
        pass
    return one_param(m).new(1, ch_in, *size).requires_grad_(False).uniform_(-1.,1.)


def dummy_eval(m:nn.Module, size:tuple=(64,64)):
    "Pass a `dummy_batch` in evaluation mode in `m` with `size`."
    return m.eval()(dummy_batch(m, size))


def num_features_model(m:nn.Module)->int:
    "Return the number of output features for `model`."
    try_sz = [128, 224, 256, 384, 448, 512, 1024]
    for sz in try_sz:
        try:
            x = dummy_eval(m, (sz, sz))
            return x.shape[1]
        except Exception as e:
            if sz >= 1024: raise



def _make_trainable(module) -> None:
    """Unfreezes a given module.

    Args:
        module: The module to unfreeze
    """
    for param in module.parameters():
        param.requires_grad = True
    module.train()


def _recursive_freeze(module, train_bn: bool = True) -> None:
    """Freezes the layers of a given module.

    Args:
        module: The module to freeze
        train_bn: If True, leave the BatchNorm layers in training mode
    """
    children = list(module.children())
    if not children:
        if not (isinstance(module, BN_TYPES) and train_bn):
            for param in module.parameters():
                param.requires_grad = False
            module.eval()
        else:
            # Make the BN layers trainable
            _make_trainable(module)
    else:
        for child in children:
            _recursive_freeze(child, train_bn=train_bn)


def freeze(module, n = None, train_bn: bool = True) -> None:
    """Freezes the layers up to index n (if n is not None).

    Args:
        module: The module to freeze (at least partially)
        n: Max depth at which we stop freezing the layers. If None, all
            the layers of the given module will be frozen.
        train_bn: If True, leave the BatchNorm layers in training mode
    """
    children = list(module.children())
    n_max = len(children) if n is None else int(n)

    for child in children[:n_max]:
        _recursive_freeze(child, train_bn=train_bn)

    for child in children[n_max:]:
        _make_trainable(child)

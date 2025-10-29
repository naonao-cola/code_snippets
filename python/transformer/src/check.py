import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def check_torchtext():

    try:
        import torchtext
    except ImportError:
        version = torch.__version__
        print("pytorch version is ",version)
        version=version.split(".")
        a = int(version[0])
        b = int(version[1])
        c = int(version[2][0])
        if int(a)==2:
            b = int(b)+15
            torchtext_version = f'0.{b}.{c}'
        if int(a)==1:
            b = int(b)+1
            torchtext_version = f'0.{b}.{c}'
        print('需要安装的torchtext版本为:', torchtext_version)
        ##!pip install torchtext=={torchtext_version}

def check_device():
    if torch.cuda.is_available():
        print('GPU is available.')
        print('Number of GPU:', torch.cuda.device_count())
        print('GPU name:', torch.cuda.get_device_name(0))
    else:
        print('GPU is not available')

check_torchtext()
check_device()
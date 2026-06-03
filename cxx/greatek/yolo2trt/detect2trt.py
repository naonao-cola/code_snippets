from ultralytics import YOLO
from pathlib import Path
from pathlib import Path
from PIL import Image
import glob
import json
from ultralytics import settings
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 下面老是报错 shape 不一致


"""
检测模型修改源码 链接 https://blog.csdn.net/qq_40672115/article/details/134276907



分割模型修改源码 链接 https://blog.csdn.net/qq_40672115/article/details/134277752
评论区
推理端的forwa 需要修改为  return (torch.cat([x.permute(0, 2, 1), mc], 1).permute(0, 2, 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))



关键点模型 https://blog.csdn.net/qq_40672115/article/details/134278117


"""

MODEL_YAML = r"E:\demo\py\yolo2trt\model\workclothes\best.pt"
settings.update({"runs_dir": "E:/demo/py/yolo2trt/detect2trt.py"})


# 转模型时,需要对模型进行修改,否则会报错,不同类型进行不同修改

def export2trt():
    model = YOLO(Path(MODEL_YAML))
    model.export(format="onnx",dynamic=False,simplify=True)  # 将模型导出为 TensorRT 格式


export2trt()

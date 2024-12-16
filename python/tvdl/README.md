# TVDL
深度学习模型库

## 代码释放
- 闭源方式
生成`whl`的python包:
``` python setup.py bdist_wheel BSO```

- 开源方式
生成`whl`的python包:
``` python setup.py bdist_wheel```

## 代码下载

## 环境安装

## 模块说明
```
tvdl
├── __init__.py
├── classification                   # 各种分类模型实现
│   ├── __init__.py
│   ├── adversarial                    # 基于对抗训练的分类模型
│   ├── backbones                      # 分类模型的骨干网络
│   ├── basic.py                       # 基础工具
│   ├── domain_adaptation              # 基于领域自适应的分类模型
│   ├── heads                          # 分类的头部网络
│   ├── semi_supervised                # 基于半监督的分类模型
│   └── transfer_learning.py           # 迁移学习分类模型
├── detection                       # 各种目标检测模型实现
│   └── __init__.py
├── regression                      # 各种回归模型实现
│   └── __init__.py
├── segmentation                    # 各种分割模型实现
│   └── __init__.py
├── utils                           # 工具类
│   └── __init__.py
└── version.py                      # 版本信息
```

## 使用步骤
1.准备dataloaders

```
ill = ImageLabelList.xxx
train_dl, valid_dl = ill.dataloader(...)
```

2.准备模型

```
from tvdl.classification.xxx import XxxModel
model = XxxModel(...)
```

3.开始训练`Trainer`

```
import pytorch_lightning as pl
trainer = pl.Trainer(default_root_dir=TASK_PATH, max_epochs=20, gpus=[0], ...)
trainer.fit(model, train_dl, valid_dl)
```

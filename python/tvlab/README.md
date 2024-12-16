# TVLab
TuringVision 视觉算法实验平台。

## 代码释放
- 闭源方式
生成`whl`的python包:
``` python setup.py bdist_wheel BSO```

- 开源方式
生成`whl`的python包:
``` python setup.py bdist_wheel```

## 代码下载
- 权限开通后再clone代码

## 环境安装
```
0. install nvidia driver
1. install anaconda
curl -O https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
bash Anaconda3-2020.11-Linux-x86_64.sh

source ~/.bashrc
conda deactivate

2. create virenv
conda create --name env_name python=3.7
conda activate env_name

3. install fastai
conda install -c pytorch -c fastai fastai=1.0.61

4. install pytorch
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

5. install pkgs
conda install scipy matplotlib ipython jupyter pandas sympy nose pillow bokeh tqdm opencv cython h5py rsa
conda install -c https://conda.anaconda.org/menpo opencv3
conda install eigen
conda config --add channels conda-forge
conda install imgaug albumentations
pip install lxml jedi==0.17.2


6. install detectron2/mmdetection
conda install -c fvcore fvcore
mkdir -p ~/public
cd ~/public
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2

pip install mmcv-full
git clone https://github.com/open-mmlab/mmdetection.git
pip install -v -e mmdetection
```

## 模块说明
```
tvlab
├── __init__.py                 # python模块必须
├── category                    # 图像分类模块代码
│   ├── __init__.py             # python模块必须
│   ├── category_experiment.py  # 分类模型试验对比工具
│   ├── eval_category.py        # 分类任务模型效果评估、可视化
│   ├── fast_category.py        # 基于fastai的分类算法训练、推理，对接ADC系统
│   ├── guided_backprop.py      # guided梯度计算，用于模型可视化分析
│   ├── image_data.py           # 图像分类数据集管理、清洗工具
│   ├── image_similar.py        # 数据集中相似图片搜索、去重
│   └── model_vis.py            # 模型可视化分析工具
├── defect_detector             # 缺陷检测模块代码
│   ├── __init__.py             # python模块必须
│   ├── basic_detector.py       # 缺陷检测基础框架
│   └── phot_detector.py        # PHOT缺陷检测算法
├── detection                   # 目标检测模块代码
│   ├── __init__.py             # python模块必须
│   ├── eval_detect.py          # 检测任务模型效果评估、可视化
│   ├── image_data.py           # 目标检测数据集管理、清洗工具
│   └── pascal_voc_io.py        # pascal voc标注格式读取代码
├── ui                          # 可视化代码
│   ├── __init__.py             # python模块必须
│   ├── bokeh_ui.py             # bokeh相关的画图帮助函数
│   ├── image_cleaner.py        # Jupyter中运行的数据集浏览、清洗工具
│   └── pyplot_ui.py            # pyplot相关的画图帮助函数
├── utils                       # 工具类
│   ├── __init__.py             # python模块必须
│   ├── basic.py                # 公用工具函数
│   ├── mysql_client.py         # mysql简易客户端
│   ├── ftp_client.py           # ftp访问工具
│   └── mt_ftp_loader.py        # 多线程ftp下载工具
├──ocr                          # ocr模块
|   ├──__init__.py              # python必须模块
|   ├──db_process.py            # db_net数据预处理工具
|   ├──fast_ocr_end2end.py      # ocr端到端训练推理接口
|   ├──fast_ocr_det.py          # 文本位置检测
|   ├──fast_ocr_rec.py          # 文字识别
|   ├──image_data.py            # ocr数据集管理、清洗工具
│   └──program.py               # 训练工具
└── version.py                  # 代码版本
```

## 相关类使用步骤
### 分类任务
1. 构建数据集 -> ImageLabelList
2. 数据集清洗 -> ImageCleaner
3. 训练模型 -> FastCategoryTrain
4. 模型结果分析 -> EvacCategory
5. 数据集去重 -> BasicImageSimilar(初级的相似度搜索) -> ImageSimilarPro(精确的相似度搜索)
6. 模型的可视化分析 -> CategoryModelVis
7. 多个模型试验对比分析 -> CategoryExperiment
8. 模型发布，运行推理 -> FastCategoryInference

### ocr使用说明
安装paddle
```
python3 -m pip install paddlepaddle-gpu==1.8.5.post107 -i https://mirror.baidu.com/pypi/simple
```
安装paddleocr v1.1.0
```
git clone https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR

# 修改MANIFEST.in文件，修改后文件如下
    include LICENSE.txt
    include README.md
    recursive-include ppocr *.*
    recursive-include tools/infer *.py
# 安装paddleocr
    python setup.py install
```

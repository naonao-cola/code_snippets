---
name: yolo-vision-tools
description: 使用 Ultralytics YOLO 执行计算机视觉任务，例如检测图像和视频中的人物或物体、图像分类、人体姿态估计，以及跟踪视频中的汽车、人物或动物。
argument-hint: |
  当用户要求使用 YOLO 计算机视觉模型（YOLO26、YOLO11 等）分析图像或视频时，使用此工具。

  ## 支持的任务与触发关键词 (Supported Tasks & Trigger Keywords)

  ### 1. Object Detection (目标检测)
  **English Triggers:**
  - detect objects in this image/video
  - what objects are in this picture/video
  - find [object] in this image/video
  - identify objects in the photo
  - locate objects in the image
  - show me what's in this picture
  - analyze this image for objects
  - object detection on this video
  - yolo detection on this image
  - run yolo on this picture
  
  **中文触发词:**
  - 检测图片/视频中的物体
  - 图片/视频里有什么东西
  - 找到图片中的[物体]
  - 识别图片中的物体
  - 定位图片中的物体
  - 显示图片中的内容
  - 分析图片中的物体
  - 对视频进行目标检测
  - 用yolo检测这张图片
  - 运行yolo分析图片

  ### 2. Instance Segmentation (实例分割)
  **English Triggers:**
  - segment objects in this image
  - extract object masks
  - highlight object contours
  - separate objects from background
  - instance segmentation
  - object segmentation
  - mask detection
  - pixel-level segmentation
  
  **中文触发词:**
  - 分割图片中的物体
  - 提取物体掩码
  - 标出物体轮廓
  - 将物体从背景分离
  - 实例分割
  - 物体分割
  - 掩码检测
  - 像素级分割

  ### 3. Image Classification (图像分类)
  **English Triggers:**
  - classify this image
  - what category is this image
  - recognize image type
  - determine image label
  - image classification
  - categorize this picture
  
  **中文触发词:**
  - 对图片进行分类
  - 图片属于哪一类
  - 识别图片类型
  - 确定图片标签
  - 图像分类
  - 分类这张图片

  ### 4. Pose Estimation (姿态估计)
  **English Triggers:**
  - detect human poses
  - find skeleton/keypoints
  - recognize body posture
  - analyze human movements
  - pose estimation
  - body pose detection
  - human keypoints
  
  **中文触发词:**
  - 检测人体姿态
  - 找出人体骨架/关键点
  - 识别身体姿势
  - 分析人物动作
  - 姿态估计
  - 人体姿态检测
  - 人体关键点

  ### 5. Object Tracking (物体跟踪)
  **English Triggers:**
  - track objects in this video
  - follow movement of objects
  - monitor object trajectories
  - track multiple objects
  - video object tracking
  - follow cars/people/animals
  
  **中文触发词:**
  - 跟踪视频中的物体
  - 追踪物体移动
  - 监测物体轨迹
  - 跟踪多个物体
  - 视频物体跟踪
  - 追踪汽车/行人/动物

  ### 6. Model & Environment (模型与环境)
  **English Triggers:**
  - install yolo
  - setup yolo environment
  - check yolo installation
  - which yolo model to use
  - compare yolo models
  - yolo model selection
  - troubleshoot yolo
  
  **中文触发词:**
  - 安装yolo
  - 设置yolo环境
  - 检查yolo安装
  - 使用哪个yolo模型
  - 比较yolo模型
  - yolo模型选择
  - yolo故障排除

  ### 7. Specific Model Requests (特定模型请求)
  **English Triggers:**
  - use yolo26/yolo11/yolov8
  - run yolo26 on this
  - yolo11 detection
  - latest yolo model
  - yolo nano/small/medium/large
  
  **中文触发词:**
  - 使用yolo26/yolo11/yolov8
  - 用yolo26运行
  - yolo11检测
  - 最新yolo模型
  - yolo nano/small/medium/large版本

  ### 8. General Analysis (通用分析)
  **English Triggers:**
  - analyze this image with yolo
  - yolo vision analysis
  - computer vision analysis
  - vision ai detection
  - ai image analysis
  
  **中文触发词:**
  - 用yolo分析这张图片
  - yolo视觉分析
  - 计算机视觉分析
  - 视觉AI检测
  - AI图片分析

  ## 快速示例 (Quick Examples):
  - "Detect objects in this image" → 触发
  - "检测这张图片中的物体" → 触发
  - "Segment the cars in this video" → 触发
  - "跟踪视频中的人体姿态" → 触发
  
---

# Ultralytics YOLO 视觉工具 (Vision Tools)

Ultralytics YOLO 是一个最先进的计算机视觉框架，支持多种任务，包括目标检测、实例分割、图像分类、姿态估计和定向边界框检测（OBB）。此技能为您提供高效使用 YOLO 的全面指南。

**最新模型**: YOLO26（于 2026 年 1 月发布），具有端到端无 NMS 推理以及优化的边缘部署特性。对于稳定的生产工作负载，推荐使用 YOLO26 和 YOLO11。

## 快速开始 (Quick Start)

### 1. 安装与环境检查

```bash
# 安装/更新 Ultralytics
pip install -U ultralytics

# 验证安装并检查环境
yolo checks
```

`yolo checks` 命令会验证 Python 版本、PyTorch、CUDA、GPU 可用性以及所有依赖项。如需详细的环境故障排除，请参阅 [环境检查](./references/environment_check.md) 或使用提供的环境检查脚本：`python scripts/check_environment.py`。

### 2. 基本使用示例

#### Python 接口
```python
from ultralytics import YOLO

# 加载模型 (YOLO 会自动从模型推断任务类型)
model = YOLO("yolo26n.pt")  # 或者你自定义的模型路径

# 对不同数据源进行预测
# 默认情况下，输出结果保存在 workspace/yolo-vision 文件夹中
results = model("image.jpg")                     # 图像文件 → 保存在 yolo-vision/outputs/images/
results = model("video.mp4", stream=True)        # 流式视频 → 保存在 yolo-vision/outputs/videos/
results = model("https://example.com/image.jpg") # URL → 保存在 yolo-vision/outputs/images/
results = model(0, show=True)                    # 摄像头实时显示 → 保存在 yolo-vision/outputs/videos/

# 自定义输出目录 (可选)
results = model("image.jpg", project="/custom/path")  # 保存到自定义目录
```

#### CLI 命令行接口
```bash
# 基本语法: yolo TASK MODE ARGS
# 默认情况下，输出结果保存在 workspace/yolo-vision 文件夹中
yolo predict model=yolo26n.pt source="image.jpg"  # → 保存在 yolo-vision/runs/detect/predict/

# 特定任务示例
yolo detect predict model=yolo26n.pt source="video.mp4"  # → 保存在 yolo-vision/runs/detect/predict/
yolo segment predict model=yolo26n-seg.pt source="image.jpg"  # → 保存在 yolo-vision/runs/segment/predict/
yolo pose predict model=yolo26n-pose.pt source="image.jpg"  # → 保存在 yolo-vision/runs/pose/predict/

# 自定义输出目录 (可选)
yolo predict model=yolo26n.pt source="image.jpg" project="/custom/path"  # 保存到自定义目录
```

### 3. 模型选择 (Model Selection)

为了快速上手，请使用以下默认模型：
- **检测 (Detection)**: `yolo26n.pt` (nano), `yolo26s.pt` (small), `yolo26m.pt` (medium)
- **分割 (Segmentation)**: `yolo26n-seg.pt`, `yolo26s-seg.pt`, `yolo26m-seg.pt`
- **分类 (Classification)**: `yolo26n-cls.pt`, `yolo26s-cls.pt`, `yolo26m-cls.pt`
- **姿态估计 (Pose Estimation)**: `yolo26n-pose.pt`, `yolo26s-pose.pt`, `yolo26m-pose.pt`
- **定向检测 (Oriented Detection)**: `yolo26n-obb.pt`, `yolo26s-obb.pt`, `yolo26m-obb.pt`

完整模型列表与选择指南：[模型名称](./references/model_names.md) | [模型选择](./references/model_selection.md)

## 核心工作流 (Core Workflow)

### 第一步：理解 YOLO 任务
YOLO 支持五种主要的计算机视觉任务。请为您的应用选择正确的任务：
- **检测 (Detection)**: 使用边界框识别和定位物体
- **分割 (Segmentation)**: 为物体生成像素级的掩码
- **分类 (Classification)**: 对整张图像进行分类
- **姿态估计 (Pose Estimation)**: 检测关键点以进行姿态分析
- **定向检测 (Oriented Detection)**: 使用角度参数检测旋转物体 (OBB)

详细对比：[任务类型](./references/task_types.md)

### 第二步：选择合适的模型
选择模型时需考虑以下因素：
- **速度 vs 精度**: Nano (最快) → X (最准)
- **硬件限制**: GPU 内存、CPU 性能
- **应用需求**: 实时处理 vs 批量处理

指南：[模型选择](./references/model_selection.md)

### 第三步：配置参数
常用配置参数：
- `conf`: 置信度阈值 (默认: 0.25)
- `iou`: 用于 NMS 的 IoU 阈值 (默认: 0.7)
- `imgsz`: 输入图像尺寸 (默认: 640)
- `device`: 设备 ID (`0` 表示第一个 GPU，`cpu` 表示纯 CPU)
- `save`: 将结果保存到磁盘
- `show`: 实时显示结果

完整示例：[配置示例](./references/configuration_samples.md)

### 第四步：处理结果
YOLO 返回包含以下内容的 `Results` 对象：
- `boxes`: 边界框、置信度分数、类别标签
- `masks`: 分割掩码 (用于分割任务)
- `keypoints`: 姿态关键点 (用于姿态估计任务)
- `probs`: 分类概率 (用于分类任务)
- `obb`: 定向边界框 (用于 OBB 任务)

## 进阶主题 (Advanced Topics)

### 训练自定义模型
```python
from ultralytics import YOLO

# 加载模型
model = YOLO("yolo26n.pt")

# 在自定义数据集上训练
results = model.train(data="dataset.yaml", epochs=100, imgsz=640)
```

训练指南：[训练基础](./references/training_basics.md) | [数据集准备](./references/dataset_preparation.md)

### 数据集准备与增强
如果你有自定义数据集或需要处理高分辨率大图：
```bash
# 拆分数据集
python scripts/dataset_tools.py

# 处理高分辨率遥感/工业图像 (避免 resize 导致小目标丢失)
python scripts/yolo_cli.py data --action crop --img-dir ./images --ann-dir ./labels
```
查看 `references/dataset_preparation.md` 获取更多信息。

### 模型可解释性分析 (热力图)
模型推理出错时，生成 GradCAM 热力图分析判断依据：
```bash
python scripts/yolo_cli.py viz --model best.pt --img bad_case.jpg
```

### 安装选项
提供多种安装方式：
- **pip**: `pip install -U ultralytics`
- **Conda**: `conda install -c conda-forge ultralytics`
- **Docker**: 预构建的 GPU/CPU 环境镜像
- **从源码安装**: 适用于开发和定制

详细说明：[安装指南](./references/installation_guide.md)

### 性能优化
- **流式模式 (Streaming Mode)**: 对视频/长序列使用 `stream=True` 以减少内存消耗
- **批处理 (Batch Processing)**: 同时处理多张图像以提高效率
- **硬件加速**: 配置 CUDA、TensorRT 或 OpenVINO 以获得最佳性能

## 参考文档 (Reference Documentation)

| 文档 | 描述 |
|----------|-------------|
| [环境检查](./references/environment_check.md) | 全面的环境验证与故障排除 |
| [安装指南](./references/installation_guide.md) | 所有安装方法（pip、Conda、Docker、源码） |
| [任务类型](./references/task_types.md) | YOLO 任务和用例的详细比较 |
| [模型名称](./references/model_names.md) | 完整的 YOLO26 模型列表及其规格 |
| [模型选择](./references/model_selection.md) | 根据需求选择模型的策略 |
| [配置示例](./references/configuration_samples.md) | 针对各种场景的参数配置示例 |
| [数据集准备](./references/dataset_preparation.md) | 准备自定义数据集进行训练的指南 |
| [训练基础](./references/training_basics.md) | 在自定义数据上训练 YOLO 模型的基础知识 |
| [参数参考](./references/parameter_reference.md) | 所有 YOLO 配置参数的完整参考 |

## 实用脚本 (Utility Scripts)

为了节省 token 消耗并提供开箱即用的工具，`scripts/` 目录中提供了以下 Python 脚本：

| 脚本 | 描述 | 用法示例 |
|--------|-------------|---------------|
| **check_environment.py** | 全面的环境诊断 | `python scripts/check_environment.py` |
| **config_templates.py** | 开箱即用的配置模板 | `from scripts.config_templates import get_production_config` |
| **dataset_tools.py** | 数据集准备和转换工具 | `from scripts.dataset_tools import coco_to_yolo` |
| **training_helpers.py** | 训练、评估和模型管理 | `from scripts.training_helpers import evaluate_model` |
| **quick_tests.py** | 快速功能测试 | `python scripts/quick_tests.py --test environment` |
| **model_utils.py** | 模型选择和验证实用工具 | `from scripts.model_utils import select_model` |

**使用脚本的好处:**
- **节省 tokens**: 从文档中提取出大型代码块
- **开箱即用**: 无需从文档中复制粘贴代码
- **模块化**: 只导入你需要的模块
- **易于维护**: 脚本可以独立更新

## 故障排除 (Troubleshooting)

### 常见问题

**Q: 安装后找不到 `yolo` 命令？**
A: 尝试运行 `python -m ultralytics yolo` 或检查 Python 环境变量 PATH。

**Q: 如何使用特定的 GPU？**
A: 设置 `device=0` (第一个 GPU) 或 `device=cpu` (仅使用 CPU 模式)。

**Q: 模型下载速度慢？**
A: 设置 `ULTRALYTICS_HOME` 环境变量以控制缓存位置。

**Q: 如何过滤特定类别？**
A: 使用 `classes` 参数: `classes=[0, 2, 5]` (类别索引)。

**Q: 长视频内存溢出？**
A: 使用 `stream=True` 将视频作为生成器处理。

**Q: 支持实时网络摄像头吗？**
A: 支持，使用 `source=0` (默认摄像头) 并配合 `show=True` 进行实时显示。

### 获取帮助
- 运行 `yolo checks` 来诊断环境问题
- 查看官方文档: https://docs.ultralytics.com
- 查看配置参考: https://docs.ultralytics.com/usage/cfg/

## 输出目录约定 (Output Directory Convention)

### 默认输出位置
当使用 YOLO 处理图像或视频时，如果用户没有指定输出目录，所有生成的文件将自动保存在工作区的 `yolo-vision` 文件夹中。

### 文件组织结构
`yolo-vision` 文件夹的结构如下：

```
yolo-vision/
├── inputs/            # 原始输入文件 (为了参考而复制)
├── outputs/           # 带有检测结果的处理后文件
│   ├── images/        # 检测后的图像
│   ├── videos/        # 检测后的视频
│   └── previews/      # 预览图
├── reports/           # 分析报告和统计数据
│   ├── json/          # JSON 格式报告
│   ├── markdown/      # Markdown 格式报告
│   └── csv/           # CSV 格式数据
├── models/            # 下载的 YOLO 模型
│   ├── yolo26/        # YOLO26 模型
│   ├── yolo11/        # YOLO11 模型
│   └── custom/        # 自定义训练的模型
└── logs/              # 处理日志和调试信息
```

### 自动创建文件夹
该技能将自动：
1. 创建 `yolo-vision` 文件夹（如果不存在）
2. 根据需要创建所有子目录
3. 按日期和任务类型组织文件
4. 生成基于时间戳的文件名，便于追踪

### 使用示例
```python
# 未指定输出目录 - 使用默认的 yolo-vision 文件夹
results = model("image.jpg")  # 输出保存在 yolo-vision/outputs/images/

# 自定义输出目录
results = model("image.jpg", save_dir="/custom/path")  # 使用指定路径
```

### 好处
1. **一致性**: 所有 YOLO 输出都集中在一个可预测的位置
2. **组织性**: 文件按类型自动分类
3. **备份**: 原始文件被保留以供参考
4. **可复现性**: 轻松查找并对比以前的分析结果
5. **干净的工作区**: 防止主工作区目录变得杂乱

### 用户覆盖
用户仍可在需要时指定自定义输出目录：
- 通过在 Python 代码中提供 `save_dir` 参数
- 通过在 CLI 命令中使用 `--project` 标志
- 通过设置 `ULTRALYTICS_PROJECT` 环境变量

---

## 🤖 Agent 执行 SOP (标准操作流程)

**⚠️ [P10 强制红线]** 作为大模型，在执行 YOLO 相关任务时，**必须**严格遵循以下闭环 SOP。不要一上来就写代码，先查环境，后出方案，用数据说话。

### Step 1: 环境与硬件嗅探 (Dive Deep)
- **嗅探 GPU**: 运行 `nvidia-smi` 检查 CUDA 是否可用、显存大小、显卡利用率。
- **嗅探环境**: 运行 `python scripts/check_environment.py` 或检查当前目录的依赖文件。如果 Ultralytics 未安装，优先使用 `uv pip install ultralytics` 或 `pip install` 进行安装。
- **防呆检查**: 如果显存小于 4GB 或只有 CPU，强制降级使用 `n` (nano) 或 `s` (small) 尺寸模型，并在旁白中向用户警告。

### Step 2: 方案设计与防呆 (Working Backwards)
- **确定模型**: 根据用户意图精准选择模型（例如：目标检测用 `yolo26n.pt`，实例分割用 `yolo26n-seg.pt`，姿态估计用 `yolo26n-pose.pt`）。参考 `references/model_selection.md`。
- **依赖隔离**: 优先生成独立的执行脚本进行测试，不要随意污染全局环境。
- **防御性编程**: 在 Python 脚本中加入 `try-except` 块，捕获 `CUDA Out of Memory` 异常，提供友好的回退机制（如降低 `batch` 或缩小 `imgsz`）。

### Step 3: 代码生成与执行 (Bias for Action)
- 编写 Python 脚本或 CLI 命令，执行 YOLO 任务。
- 确保输出路径遵循上面的 "输出目录约定"，统一存放在 `yolo-vision/outputs/` 下，保持用户工作区整洁。

### Step 4: 结果验证与闭环 (Deliver Results)
- 执行脚本后，**必须**读取输出日志或使用 `ls` 验证输出文件是否真实生成。没有证据的交付叫自嗨。
- 如果发生报错，立即启动 RCA（根因分析），更换参数或降级模型再次重试。
- 成功后，向用户总结产出，并提供结果文件的相对路径。

---

**许可证说明**: Ultralytics YOLO 在开源使用时遵循 AGPL-3.0 许可证，商业应用需使用企业许可证。详情请查看 https://ultralytics.com/license。

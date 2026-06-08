# YOLO 参数参考 (Parameter Reference)

本文档基于官方 Ultralytics 的 `args_default.yaml` 文件，提供了所有 YOLO 配置参数的全面参考。

**注意**: 有关实际的配置示例，请参阅 [配置示例](./configuration_samples.md)。本文档作为所有可用参数的完整参考。

## 参数类别 (Parameter Categories)

YOLO 参数分为以下几类：

1. **全局参数 (Global Parameters)** - 任务和模式设置
2. **训练参数 (Training Parameters)** - 模型训练配置
3. **验证/测试参数 (Validation/Test Parameters)** - 评估设置
4. **预测参数 (Prediction Parameters)** - 推理配置
5. **导出参数 (Export Parameters)** - 模型导出设置
6. **超参数 (Hyperparameters)** - 优化和数据增强
7. **日志参数 (Logging Parameters)** - 实验跟踪

## 1. 全局参数 (Global Parameters)

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `task` | str | `detect` | YOLO 任务: `detect`, `segment`, `classify`, `pose`, `obb` |
| `mode` | str | `train` | YOLO 模式: `train`, `val`, `predict`, `export`, `track`, `benchmark` |

## 2. 训练参数 (Training Parameters)

### 基础训练设置

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `model` | str | - | 模型文件路径 (例如 `yolo26n.pt`) 或架构文件 (例如 `yolo26n.yaml`) |
| `data` | str | - | 数据集配置文件路径 (例如 `coco8.yaml`) |
| `epochs` | int | `100` | 训练轮数 |
| `time` | float | - | 最大训练时间(小时) (如果设置则覆盖 epochs) |
| `patience` | int | `100` | 早停机制：在 N 轮没有验证提升后停止 |
| `batch` | int/float | `16` | 批次大小（整数），或 0.0-1.0 的浮点数表示 AutoBatch 占用 GPU 内存的比例 |
| `imgsz` | int/list | `640` | 图像尺寸: 整数表示方形，或 [高度, 宽度] 表示矩形 |
| `save` | bool | `True` | 保存训练检查点和预测结果 |
| `save_period` | int | `-1` | 每 N 轮保存一次检查点 (-1 = 仅保存最后一次) |
| `cache` | bool/str | `False` | 缓存图像: `True`/'ram' 表示在内存缓存，'disk' 表示在磁盘缓存 |
| `device` | int/str/list | - | 设备: 0 或 [0,1,2,3] (CUDA), 'cpu', 'mps', 或自动选择 |
| `workers` | int | `8` | 数据加载线程数 (如果是 DDP 则是每个 RANK 的线程数) |
| `project` | str | - | 结果目录的项目名称 |
| `name` | str | - | 实验名称 (结果保存在 'project/name' 中) |
| `exist_ok` | bool | `False` | 覆盖现有的 'project/name' 目录 |
| `pretrained` | bool/str | `True` | 使用预训练权重 (bool) 或从路径加载 (str) |
| `optimizer` | str | `auto` | 优化器: SGD, Adam, AdamW 等，或 'auto' |
| `verbose` | bool | `True` | 在训练/验证期间打印详细日志 |
| `seed` | int | `0` | 随机种子以保证可重复性 |
| `deterministic` | bool | `True` | 启用确定性操作 |
| `single_cls` | bool | `False` | 将所有类别视为单一类别 |
| `rect` | bool | `False` | 矩形训练批次 |
| `cos_lr` | bool | `False` | 使用余弦学习率调度器 |
| `close_mosaic` | int | `10` | 在最后 N 轮禁用马赛克数据增强 |
| `resume` | bool | `False` | 从最后一个检查点恢复训练 |
| `amp` | bool | `True` | 自动混合精度 (AMP) 训练 |
| `fraction` | float | `1.0` | 使用训练数据集的比例 |
| `profile` | bool | `False` | 在训练期间分析 ONNX/TensorRT 的速度 |
| `freeze` | int/list | - | 冻结前 N 层或特定的层索引 |
| `multi_scale` | float | `0.0` | 多尺度范围，作为 imgsz 的比例 |
| `compile` | bool/str | `False` | 启用带有后端的 torch.compile() |

### 特定任务的训练参数

#### 分割 (Segmentation)
| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `overlap_mask` | bool | `True` | 在训练期间合并实例掩码 |
| `mask_ratio` | int | `4` | 掩码下采样比例 |

#### 分类 (Classification)
| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `dropout` | float | `0.0` | 分类头的 Dropout 比例 |

## 3. 验证/测试参数 (Validation/Test Parameters)

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `val` | bool | `True` | 在训练期间进行验证 |
| `split` | str | `val` | 用于验证的数据集划分: 'val', 'test', 或 'train' |
| `save_json` | bool | `False` | 将结果保存为 JSON 文件 |
| `save_hybrid` | bool | `False` | 保存标签的混合版本 |
| `conf` | float | `0.001` | 验证的置信度阈值 |
| `iou` | float | `0.6` | 验证的 IoU 阈值 |
| `max_det` | int | `300` | 每张图像的最大检测数 |
| `half` | bool | `True` | 使用半精度 (FP16) |
| `dnn` | bool | `False` | 使用 OpenCV DNN 进行 ONNX 推理 |
| `plots` | bool | `True` | 在训练/验证期间保存图表 |
| `rect` | bool | `False` | 当 mode='val' 时进行矩形验证 |

## 4. 预测参数 (Prediction Parameters)

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `source` | str | - | 来源: 文件、目录、URL、屏幕、PIL、OpenCV、numpy、torch 等 |
| `conf` | float | `0.25` | 检测的置信度阈值 |
| `iou` | float | `0.7` | NMS 的 IoU 阈值 |
| `imgsz` | int/list | `640` | 推理图像尺寸 |
| `max_det` | int | `300` | 每张图像的最大检测数 |
| `device` | int/str/list | - | 推理设备 |
| `show` | bool | `False` | 显示结果 |
| `save` | bool | `False` | 将结果保存到 'runs/detect' |
| `save_txt` | bool | `False` | 将结果保存为文本文件 |
| `save_conf` | bool | `False` | 在文本文件中保存置信度 |
| `save_crop` | bool | `False` | 保存裁剪的检测结果 |
| `show_labels` | bool | `True` | 在结果上显示标签 |
| `show_conf` | bool | `True` | 在结果上显示置信度 |
| `vid_stride` | int | `1` | 视频帧步长 |
| `stream_buffer` | bool | `False` | 缓冲所有流式帧 |
| `line_width` | int/float | - | 边界框线条宽度 |
| `visualize` | bool | `False` | 可视化模型特征 |
| `augment` | bool | `False` | 应用测试时数据增强 (TTA) |
| `agnostic_nms` | bool | `False` | 类别无关的 NMS |
| `retina_masks` | bool | `False` | 使用高分辨率掩码 (用于分割) |
| `boxes` | bool | `True` | 显示边界框 (用于分割) |
| `format` | str | `torchscript` | 导出的格式 |

## 5. 导出参数 (Export Parameters)

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `format` | str | `torchscript` | 导出格式: torchscript, onnx, openvino, tensorrt, coreml, saved_model, pb, tflite, paddle, ncnn |
| `imgsz` | int/list | `640` | 导出图像尺寸 |
| `keras` | bool | `False` | 导出为 Keras SavedModel |
| `optimize` | bool | `False` | TorchScript: 针对移动端优化 |
| `half` | bool | `False` | FP16 量化 |
| `int8` | bool | `False` | INT8 量化 |
| `dynamic` | bool | `False` | ONNX/TensorRT: 动态轴 |
| `simplify` | bool | `False` | ONNX: 简化模型 |
| `opset` | int | - | ONNX: opset 版本 |
| `workspace` | int/float | `4` | TensorRT: 工作空间大小 (GB) |
| `nms` | bool | `False` | CoreML: 添加 NMS |

## 6. 超参数 (Hyperparameters)

### 学习率参数
| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `lr0` | float | `0.01` | 初始学习率 |
| `lrf` | float | `0.01` | 最终学习率系数 |
| `momentum` | float | `0.937` | SGD 动量/Adam beta1 |
| `weight_decay` | float | `0.0005` | 优化器权重衰减 |
| `warmup_epochs` | float | `3.0` | 预热轮数 |
| `warmup_momentum` | float | `0.8` | 预热初始动量 |
| `warmup_bias_lr` | float | `0.1` | 预热初始偏置学习率 |

### 数据增强参数
| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `hsv_h` | float | `0.015` | HSV-色调增强 (比例) |
| `hsv_s` | float | `0.7` | HSV-饱和度增强 (比例) |
| `hsv_v` | float | `0.4` | HSV-明度增强 (比例) |
| `degrees` | float | `0.0` | 图像旋转 (+/- 角度) |
| `translate` | float | `0.1` | 图像平移 (+/- 比例) |
| `scale` | float | `0.5` | 图像缩放 (+/- 增益) |
| `shear` | float | `0.0` | 图像剪切 (+/- 角度) |
| `perspective` | float | `0.0` | 图像透视 (+/- 比例) |
| `flipud` | float | `0.0` | 图像上下翻转 (概率) |
| `fliplr` | float | `0.5` | 图像左右翻转 (概率) |
| `mosaic` | float | `1.0` | 图像马赛克 (概率) |
| `mixup` | float | `0.0` | 图像 mixup (概率) |
| `copy_paste` | float | `0.0` | 实例复制粘贴 (概率) |
| `erasing` | float | `0.4` | 随机擦除 (概率) |
| `crop_fraction` | float | `1.0` | 裁剪图像的比例 |

### 损失参数
| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `label_smoothing` | float | `0.0` | 标签平滑 epsilon |
| `box` | float | `7.5` | 边界框损失增益 |
| `cls` | float | `0.5` | 分类损失增益 |
| `dfl` | float | `1.5` | DFL 损失增益 |
| `pose` | float | `12.0` | 姿态损失增益 (仅用于姿态估计) |
| `kobj` | float | `1.0` | 关键点目标损失增益 (仅用于姿态估计) |
| `nbs` | int | `64` | 标称批次大小 |

## 7. 日志和跟踪参数 (Logging and Tracking Parameters)

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `entity` | str | - | Weights & Biases 实体 |
| `upload_dataset` | bool | `False` | 上传数据集到 W&B |
| `bbox_interval` | int | `-1` | W&B: 边界框记录间隔 |
| `artifact_alias` | str | `latest` | W&B: artifact 别名 |

## 参数使用示例 (Parameter Usage Examples)

### 基础训练
```python
from ultralytics import YOLO

model = YOLO('yolo26n.pt')
model.train(
    data='coco8.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device='cuda:0',
    lr0=0.01,
    augment=True,
)
```

### 带有自定义参数的推理
```python
results = model.predict(
    source='image.jpg',
    conf=0.5,      # 更高的置信度阈值
    iou=0.45,      # 针对密集场景使用更低的 IoU
    imgsz=1280,    # 更高的分辨率
    augment=True,  # 测试时数据增强
    max_det=100,   # 限制检测数
)
```

### 优化导出
```python
model.export(
    format='onnx',
    imgsz=[640, 480],
    half=True,      # FP16 量化
    simplify=True,  # 简化模型
    dynamic=True,   # 动态轴
)
```

## 参数选择指南 (Parameter Selection Guidelines)

### 速度与精度的权衡
- **速度优先**: 降低 `imgsz`，启用 `half`，禁用 `augment`
- **精度优先**: 提高 `imgsz`，禁用 `half`，启用 `augment`

### 针对不同场景
- **密集场景**: 降低 `iou`，提高 `conf`
- **小目标**: 降低 `conf`，提高 `imgsz`
- **实时处理**: 启用 `half`，降低 `imgsz`，禁用 `augment`

### 硬件考虑因素
- **GPU 内存有限**: 减小 `batch`，启用 `half`
- **仅 CPU**: 禁用 `half`，使用较小的 `imgsz`
- **多 GPU**: 为 `device` 使用列表 (例如 `device=[0,1]`)

## 相关文档 (Related Documentation)

- [配置示例](./configuration_samples.md) - 实际配置示例
- [训练基础](./training_basics.md) - 训练工作流和最佳实践
- [Ultralytics 官方文档](https://docs.ultralytics.com/usage/cfg/) - 完整的参数参考
- [模型选择](./model_selection.md) - 为您的任务选择合适的模型

## 参数管理实用脚本 (Utility Scripts for Parameter Management)

如需即用型的配置模板和参数工具，请使用提供的脚本：

```bash
# 运行配置模板示例
python scripts/config_templates.py

# 测试参数配置
python scripts/quick_tests.py --test performance --model yolo26n.pt
```

**脚本位置**: `scripts/config_templates.py`

**其他用于参数管理的脚本**:
- `scripts/training_helpers.py` - 训练参数配置
- `scripts/dataset_tools.py` - 数据集准备参数
- `scripts/model_utils.py` - 模型选择参数

**优势**:
- 通过提取文档中的代码来节省 tokens
- 即用型的配置模板
- 跨所有任务一致的参数处理
- 模块化设计，只需导入需要的内容

**使用示例**:
```python
from scripts.config_templates import (
    get_basic_config,
    get_production_config,
    get_config_for_scenario,
    merge_configs,
    print_config
)

# 获取特定场景的配置
config = get_config_for_scenario('production')

# 打印配置
print_config(config, "Production Configuration")

# 合并配置
base = get_basic_config()
custom = {'conf': 0.4, 'imgsz': 1280}
merged = merge_configs(base, custom)
```
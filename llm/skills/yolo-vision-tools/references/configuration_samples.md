# YOLO 配置示例 (Configuration Samples)

本文档提供了 Ultralytics YOLO 的各种配置示例，涵盖了从基础到高级的使用场景。

**最新模型**: 示例使用的是 YOLO26 模型，但只需替换模型名称即可适用于 YOLO11 及其他版本。

**注意**: 有关全面的配置模板和即用型函数，请参阅 `scripts/` 目录下的 `config_templates.py` 脚本。

## 快速开始 (Quick Start)

### 最简单的用法
```python
from ultralytics import YOLO

# 加载模型
model = YOLO('yolo26n.pt')  # 预训练模型

# 基础推理
results = model('bus.jpg')                     # 单张图像
```

## 基础配置参数 (Basic Configuration Parameters)

### 常用参数说明
```python
# 基础配置参数
config = {
    'conf': 0.25,      # 置信度阈值 (0-1)
    'iou': 0.7,        # NMS IoU 阈值 (0-1)
    'imgsz': 640,      # 输入图像尺寸
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
    'show': False,     # 显示结果
    'save': False,     # 保存结果
}
```

**如需完整的参数参考和模板，请使用:**
```python
from scripts.config_templates import get_basic_config, get_production_config
basic_config = get_basic_config()
production_config = get_production_config()
```

## 按场景分类的配置示例

### 1. 图像处理 (Image Processing)

#### 单图检测
```python
# 基础图像处理
results = model.predict(source='image.jpg', conf=0.5, save=True)
```

#### 批量图像处理
```python
# 处理多张图像
results = model.predict(source='images/', batch=16, workers=8)
```

**如需特定场景的配置，请使用:**
```python
from scripts.config_templates import get_image_processing_config
config = get_image_processing_config()
```

### 2. 视频处理 (Video Processing)

#### 标准视频处理
```python
# 处理视频文件
results = model.predict(source='video.mp4', save=True)
```

#### 使用流式处理实时视频
```python
# 内存友好的视频处理
results = model.predict(source='video.mp4', stream=True, show=True)
```

**如需视频配置，请使用:**
```python
from scripts.config_templates import get_video_processing_config
config = get_video_processing_config(stream=True)
```

### 3. 实时摄像头流 (Real-time Camera Streams)

#### 摄像头处理
```python
# 处理默认摄像头
results = model.predict(source=0, show=True)
```

**如需实时配置，请使用:**
```python
from scripts.config_templates import get_webcam_config, get_realtime_config
webcam_config = get_webcam_config()
realtime_config = get_realtime_config()
```

### 4. 特殊场景 (Special Scenarios)

#### 弱光/挑战性环境
```python
# 针对挑战性环境的调整
results = model.predict(source='low_light.jpg', conf=0.15, imgsz=1280)
```

#### 密集场景处理
```python
# 针对密集场景的优化
results = model.predict(source='crowd.jpg', conf=0.4, iou=0.3, max_det=1000)
```

**如需特殊场景配置，请使用:**
```python
from scripts.config_templates import (
    get_low_light_config,
    get_crowded_scene_config,
    get_small_object_config
)
```

## 性能优化配置 (Performance Optimization)

### 1. GPU 加速
```python
# GPU 优化
config = {
    'device': 'cuda:0',
    'half': True,      # FP16 半精度
    'batch': 32,
    'workers': 16,
}
```

### 2. CPU 优化
```python
# CPU 优化
config = {
    'device': 'cpu',
    'batch': 1,
    'workers': 4,
    'half': False,
}
```

**如需性能配置，请使用:**
```python
from scripts.config_templates import (
    get_gpu_optimized_config,
    get_cpu_optimized_config,
    get_memory_efficient_config
)
```

## 特定任务配置 (Task-Specific Configurations)

### 1. 目标检测 (Object Detection)
```python
# 检测配置
config = {'task': 'detect', 'conf': 0.25, 'iou': 0.7}
```

### 2. 实例分割 (Instance Segmentation)
```python
# 分割配置
config = {'task': 'segment', 'retina_masks': True, 'boxes': True}
```

### 3. 姿态估计 (Pose Estimation)
```python
# 姿态估计配置
config = {'task': 'pose', 'kpt_shape': [17, 3]}
```

**如需特定任务配置，请使用:**
```python
from scripts.config_templates import (
    get_detection_config,
    get_segmentation_config,
    get_pose_config,
    get_classification_config,
    get_obb_config
)
```

## 训练配置 (Training Configurations)

### 基础训练
```python
# 基础训练配置
train_config = {
    'data': 'dataset.yaml',
    'epochs': 100,
    'imgsz': 640,
    'batch': 16,
}
```

### 高级训练
```python
# 包含超参数的高级训练
train_config = {
    'data': 'dataset.yaml',
    'epochs': 100,
    'lr0': 0.01,        # 初始学习率
    'lrf': 0.01,        # 最终学习率系数
    'augment': True,    # 启用数据增强
}
```

**如需训练配置，请使用:**
```python
from scripts.config_templates import get_training_config, get_advanced_training_config
from scripts.training_helpers import get_basic_training_config, get_advanced_training_config
```

## 配置模板 (Configuration Templates)

### 生产环境模板
```python
# 生产级配置
from scripts.config_templates import get_production_config
config = get_production_config()
```

### 开发/调试模板
```python
# 开发配置
from scripts.config_templates import get_development_config
config = get_development_config()
```

### 实时应用模板
```python
# 实时配置
from scripts.config_templates import get_realtime_config
config = get_realtime_config()
```

## 使用配置脚本 (Using Configuration Scripts)

### 1. 导入配置模板
```python
from scripts.config_templates import (
    get_basic_config,
    get_production_config,
    get_realtime_config,
    get_config_for_scenario,
    merge_configs
)

# 获取特定场景的配置
config = get_config_for_scenario('production')

# 合并配置
base = get_basic_config()
custom = {'conf': 0.4, 'imgsz': 1280}
merged = merge_configs(base, custom)
```

### 2. 打印配置
```python
from scripts.config_templates import print_config

config = get_production_config()
print_config(config, "Production Configuration")
```

### 3. 可用场景
```python
from scripts.config_templates import get_config_for_scenario

# 可用场景:
# - 'production', 'realtime', 'development'
# - 'image', 'video', 'webcam'
# - 'low_light', 'crowded', 'small_objects'
# - 'gpu', 'cpu', 'memory'
# - 'detection', 'segmentation', 'pose', 'classification', 'obb'
# - 'training', 'advanced_training'

config = get_config_for_scenario('realtime')
```

## 故障排除配置 (Troubleshooting Configurations)

### 常见问题与修复

#### 内存不足 (Memory Issues)
```python
# 减少内存使用
config = {
    'stream': True,     # 对大输入至关重要
    'batch': 1,         # 最小批次大小
    'imgsz': 320,       # 降低分辨率
    'half': True,       # 半精度
}
```

#### 性能缓慢 (Slow Performance)
```python
# 速度优化
config = {
    'imgsz': 320,       # 降低分辨率
    'half': True,       # FP16
    'batch': 32,        # 批处理
    'augment': False,   # 禁用增强
}
```

#### 检测质量低 (Low Detection Quality)
```python
# 提高检测质量
config = {
    'imgsz': 1280,      # 提高分辨率
    'augment': True,    # 启用数据增强
    'conf': 0.1,        # 降低初始阈值
    'max_det': 1000,    # 增加检测数
}
```

## 完整参数参考 (Complete Parameter Reference)

有关所有可用参数的完整列表，请参阅:
- [参数参考 (Parameter Reference)](./parameter_reference.md) - 完整的参数文档
- `scripts/config_templates.py` - 即用型配置模板
- `scripts/parameter_reference.md` - 详细的参数规范

### 关键参数组

| 参数组 | 关键参数 | 典型值 |
|-----------------|----------------|----------------|
| **输入控制** | `source`, `imgsz`, `batch`, `workers` | 因应用而异 |
| **检测控制** | `conf`, `iou`, `max_det`, `agnostic_nms` | `conf=0.25`, `iou=0.7` |
| **输出控制** | `save`, `save_txt`, `save_conf`, `show` | 布尔标志 |
| **性能** | `device`, `half`, `dnn`, `stream` | 设备相关 |
| **高级** | `augment`, `visualize`, `retina_masks` | 特殊功能 |

## 最佳实践 (Best Practices)

1. **从简单开始**: 先使用默认参数，然后根据结果进行调整。
2. **使用模板**: 利用 `scripts/config_templates.py` 中的配置模板。
3. **性能分析**: 使用 `yolo benchmark` 测试不同的配置。
4. **验证更改**: 在验证数据上测试参数更改的效果。
5. **记录配置**: 记录成功的配置参数。
6. **考虑硬件**: 根据可用的硬件资源调整参数。

## 相关文档 (Related Documentation)

- [参数参考](./parameter_reference.md) - 完整的参数参考
- [模型选择](./model_selection.md) - 选择合适的模型
- [任务类型](./task_types.md) - 了解不同的 YOLO 任务
- [训练基础](./training_basics.md) - 训练配置指南
- [Ultralytics 官方文档](https://docs.ultralytics.com/usage/cfg/) - 完整的参数参考

## 实用脚本 (Utility Scripts)

如需即用型的配置模板和工具：

```bash
# 运行配置示例
python scripts/config_templates.py

# 在代码中导入
from scripts.config_templates import (
    get_basic_config,
    get_production_config,
    get_realtime_config,
    get_config_for_scenario,
    merge_configs,
    print_config
)
```

**脚本位置**: `scripts/config_templates.py`
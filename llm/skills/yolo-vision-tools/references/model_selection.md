# YOLO 模型选择指南 (Model Selection Guide)

## 概览 (Overview)

本指南帮助您根据应用需求、硬件限制和性能要求选择合适的 Ultralytics YOLO 模型。

**关键决策**: 在 YOLO26（最新特性）和 YOLO11（稳定生产）之间进行选择。两者都支持所有五种任务：检测、分割、分类、姿态估计和定向检测。

## 快速决策流程图

```mermaid
graph TD
    A[开始模型选择] --> B{YOLO 版本?};
    B --> C[最新特性];
    B --> D[生产环境稳定];
    C --> E[选择 YOLO26];
    D --> F[选择 YOLO11];
    
    E --> G{主要需求?};
    F --> G;
    
    G --> H[速度优先];
    G --> I[精度优先];
    G --> J[平衡性能];
    
    H --> K{设备类型?};
    K --> L[移动端/边缘设备];
    K --> M[桌面端/服务器];
    L --> N[选择 Nano (n)];
    M --> O[选择 Small (s)];
    
    I --> P{硬件资源?};
    P --> Q[GPU 显存充足];
    P --> R[资源受限];
    Q --> S[选择 XLarge (x)];
    R --> T[选择 Large (l)];
    
    J --> U{任务类型?};
    U --> V[通用目标检测];
    U --> W[分割/分类/姿态];
    V --> X[选择 Medium (m)];
    W --> Y[选择特定任务的 Medium 模型];
```

## 模型规格对比

### YOLO26 模型 (最新一代)

| 模型 | 尺寸 | 相对速度 | 相对精度 | 大约参数量 | 大约体积 | 最佳适用场景 |
|-------|------|----------------|-------------------|----------------|--------------|----------|
| **yolo26n** | Nano | ⚡⚡⚡⚡⚡ (最快) | ⭐ (最低) | ~2.5M | ~5MB | 移动端、边缘计算、实时处理 |
| **yolo26s** | Small | ⚡⚡⚡⚡ | ⭐⭐ | ~9M | ~15MB | 通用目标、平衡需求 |
| **yolo26m** | Medium | ⚡⚡⚡ | ⭐⭐⭐ | ~25M | ~40MB | 服务器应用、中等精度要求 |
| **yolo26l** | Large | ⚡⚡ | ⭐⭐⭐⭐ | ~50M | ~85MB | 精度优先、高质量检测 |
| **yolo26x** | XLarge | ⚡ | ⭐⭐⭐⭐⭐ (最高) | ~100M | ~170MB | 研究、极端精度需求 |

### YOLO11 模型 (生产环境稳定)

| 模型 | 尺寸 | 相对速度 | 相对精度 | 最佳适用场景 |
|-------|------|----------------|-------------------|----------|
| **yolo11n** | Nano | ⚡⚡⚡⚡⚡ | ⭐ | 生产环境移动端/边缘应用 |
| **yolo11s** | Small | ⚡⚡⚡⚡ | ⭐⭐ | 通用生产环境工作负载 |
| **yolo11m** | Medium | ⚡⚡⚡ | ⭐⭐⭐ | 平衡的生产环境精度 |
| **yolo11l** | Large | ⚡⚡ | ⭐⭐⭐⭐ | 高精度生产环境 |
| **yolo11x** | XLarge | ⚡ | ⭐⭐⭐⭐⭐ | 极致精度生产环境 |

> **注意**: 规格参数为近似值。有关最新数据，请参考 [Ultralytics 官方文档](https://docs.ultralytics.com/models/)。

## 按任务类型选择

### 目标检测 (Object Detection)
- **通用场景**: `yolo26s.pt` 或 `yolo11s.pt`
- **实时应用**: `yolo26n.pt` 或 `yolo11n.pt`
- **高精度**: `yolo26l.pt` 或 `yolo11l.pt`
- **研究/基准测试**: `yolo26x.pt` 或 `yolo11x.pt`

### 实例分割 (Instance Segmentation)
- **通用分割**: `yolo26m-seg.pt` 或 `yolo11m-seg.pt`
- **实时分割**: `yolo26n-seg.pt` 或 `yolo11n-seg.pt`
- **医疗/精密应用**: `yolo26l-seg.pt` 或 `yolo11l-seg.pt`

### 图像分类 (Image Classification)
- **通用分类**: `yolo26m-cls.pt` 或 `yolo11m-cls.pt`
- **移动端分类**: `yolo26n-cls.pt` 或 `yolo11n-cls.pt`
- **细粒度分类**: `yolo26l-cls.pt` 或 `yolo11l-cls.pt`

### 姿态估计 (Pose Estimation)
- **通用姿态**: `yolo26m-pose.pt` 或 `yolo11m-pose.pt`
- **实时姿态**: `yolo26n-pose.pt` 或 `yolo11n-pose.pt`
- **高精度姿态**: `yolo26l-pose.pt` 或 `yolo11l-pose.pt`

### 定向边界框检测 (Oriented Bounding Box Detection)
- **通用 OBB**: `yolo26m-obb.pt` 或 `yolo11m-obb.pt`
- **航空/卫星图像**: `yolo26l-obb.pt` 或 `yolo11l-obb.pt`
- **文档分析**: `yolo26s-obb.pt` 或 `yolo11s-obb.pt`

## 硬件考量

### GPU 显存需求
- **Nano/Small**: 2-4GB 显存
- **Medium**: 4-8GB 显存  
- **Large/XLarge**: 8+ GB 显存

### 仅 CPU 部署
对于纯 CPU 环境：
1. 使用较小的模型 (nano 或 small)
2. 考虑将 batch size 设为 1
3. 启用 `half=False` 以兼容 CPU
4. 使用 OpenVINO 或 ONNX 运行时进行优化

### 移动端/边缘设备部署
对于移动端和边缘设备：
1. 使用 Nano 模型
2. 考虑使用 TensorRT、CoreML 或 NCNN 部署
3. 通过量化 (INT8/FP16) 进行优化
4. 在 Android 上使用 TensorFlow Lite

## 应用场景

### 图像处理流水线
- **批处理 (Batch Processing)**: 使用 Medium 或 Large 模型，batch size > 1
- **交互式应用**: 使用 Small 模型，实现实时响应
- **质量控制系统**: 使用 Large 模型以追求最大精度

### 视频处理
- **实时视频**: Nano 或 Small 模型
- **离线视频分析**: 使用 Medium 模型并开启批处理
- **高质量生产环境**: 使用 Large 模型，并进行细致优化

### 特殊场景
- **低光照条件**: 较大的模型通常表现更好
- **小目标检测**: 更高的输入分辨率 + 较大的模型
- **拥挤场景**: 更高的置信度阈值 + 较大的模型

## 选择策略

### 分步选择流程

1. **定义需求**
   - 实时处理 vs 批处理
   - 精度阈值
   - 硬件限制
   - 部署环境

2. **选择 YOLO 版本**
   - YOLO26 用于最新特性和边缘优化
   - YOLO11 用于经过验证的生产环境稳定性

3. **选择模型尺寸**
   - Nano: 速度最快，资源占用最小
   - Small: 适用于大多数应用的良好平衡点
   - Medium: 服务器应用的默认选择
   - Large: 当精度至关重要时
   - XLarge: 用于研究、基准测试和极致精度

4. **通过测试验证**
   - 在具有代表性的样本数据上进行测试
   - 在目标硬件上测量推理速度
   - 评估精度指标
   - 根据结果进行调整

### Python 选择逻辑示例

```python
def select_model(task="detect", speed_priority="balanced", hardware="gpu"):
    """选择合适模型的辅助函数"""
    
    # YOLO 版本选择
    version = "yolo26"  # 或者 "yolo11" (用于生产环境)
    
    # 基于速度优先级选择尺寸
    if speed_priority == "max_speed":
        size = "n"
    elif speed_priority == "speed":
        size = "s"  
    elif speed_priority == "balanced":
        size = "m"
    elif speed_priority == "accuracy":
        size = "l"
    elif speed_priority == "max_accuracy":
        size = "x"
    else:
        size = "m"
    
    # 任务后缀
    if task == "detect":
        suffix = ""
    elif task == "segment":
        suffix = "-seg"
    elif task == "classify":
        suffix = "-cls"
    elif task == "pose":
        suffix = "-pose"
    elif task == "obb":
        suffix = "-obb"
    else:
        suffix = ""
    
    return f"{version}{size}{suffix}.pt"

# 使用示例
model_name = select_model(task="segment", speed_priority="balanced")
print(f"选择的模型: {model_name}")  # 输出: yolo26m-seg.pt
```

## 性能优化提示

1. **输入尺寸 (Input Size)**: 较小的 `imgsz` 可提高速度，较大的 `imgsz` 可提高精度
2. **批大小 (Batch Size)**: 较大的 batch 提高吞吐量，较小的 batch 降低延迟
3. **精度 (Precision)**: 使用 FP16 提高速度，使用 FP32 获取最大精度
4. **置信度阈值 (Confidence)**: 根据应用需求调整 `conf`
5. **设备选择 (Device)**: 使用 GPU 获取速度，使用 CPU 获取兼容性

## 常见问题 (Common Questions)

**Q: 我应该使用 YOLO26 还是 YOLO11？**
A: 使用 YOLO26 获取最新特性和边缘部署优化。使用 YOLO11 获得生产环境的稳定性。

**Q: 如何在速度和精度之间做出选择？**
A: 在您的特定数据和硬件上测试两者。性能差异因应用而异。

**Q: 以后可以更换模型吗？**
A: 可以，YOLO 模型共享相同的 API。您可以轻松地在模型之间切换。

**Q: 模型尺寸对我的应用有多重要？**
A: 非常重要。较大的模型需要更多资源，但提供更好的精度。

**Q: 我应该训练自己的模型吗？**
A: 对于自定义物体或特定领域的应用，是的。对于通用物体，预训练模型通常已经足够。

## 进一步阅读

- [模型名称参考](./model_names.md) - 所有 YOLO 模型的完整列表
- [任务类型](./task_types.md) - 计算机视觉任务的详细解释
- [配置示例](./configuration_samples.md) - 参数调优示例
- [Ultralytics 模型文档](https://docs.ultralytics.com/models/) - 官方模型规格

## 实用脚本

如需使用开箱即用的模型选择和管理工具，请使用 `model_utils.py` 脚本：

```bash
# 运行模型工具示例
python scripts/model_utils.py

# 在您的代码中导入
from scripts.model_utils import (
    select_model,                    # 根据需求选择模型
    get_model_recommendation,        # 获取详细推荐
    get_model_specifications,        # 获取模型规格
    print_model_comparison,          # 打印对比表格
    validate_model_file,             # 验证模型文件
    compare_models_for_task,         # 比较特定任务的模型
    load_model_with_fallback,        # 加载带有回退选项的模型
    get_model_cache_info,            # 获取缓存信息
    clear_model_cache                # 清理旧缓存文件
)

# 使用示例
model_name = select_model(task="detect", speed_priority="balanced")
recommendation = get_model_recommendation({
    'task': 'segment',
    'realtime': True,
    'accuracy_needed': 'medium'
})
print_model_comparison("yolo26")
```

**脚本位置**: `scripts/model_utils.py`

**模型管理的附加脚本**:
- `scripts/quick_tests.py` - 快速模型测试
- `scripts/training_helpers.py` - 训练与评估
- `scripts/config_templates.py` - 配置模板

**优势**:
- 通过提取文档中的大型代码块来节省 tokens
- 开箱即用的模型选择和管理函数
- 所有模型相关任务的 API 保持一致
- 模块化设计，只导入您需要的部分

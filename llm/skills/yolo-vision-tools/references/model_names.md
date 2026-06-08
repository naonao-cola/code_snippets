# YOLO 模型名称参考 (Model Names Reference)

## 概览 (Overview)

本文档列出了所有 Ultralytics YOLO 预训练模型的名称，并按任务类型进行了分类。这些模型可以通过 `YOLO()` 函数或 CLI 命令行直接加载。

**最新模型**: YOLO26（发布于 2026 年 1 月）是最新一代的无 NMS 端到端推理模型。对于稳定的生产环境，建议使用 YOLO26 和 YOLO11。

## 模型命名约定 (Model Naming Convention)

YOLO 模型遵循以下命名模式：
- **版本前缀**: `yolo26` (最新), `yolo11` (稳定), `yolo10`, `yolov8`, 等
- **尺寸标识符**: `n` (nano), `s` (small), `m` (medium), `l` (large), `x` (extra large)
- **任务后缀**: `.pt` (目标检测), `-seg` (实例分割), `-cls` (图像分类), `-pose` (姿态估计), `-obb` (定向边界框检测)
- **文件扩展名**: `.pt` (PyTorch 模型文件)

## YOLO26 模型 (最新)

### 目标检测 (Object Detection)
| 模型名称 | 尺寸 | 描述 |
|------------|------|-------------|
| `yolo26n.pt` | Nano | 最快、最小，适合移动端/边缘设备 |
| `yolo26s.pt` | Small | 速度和精度的平衡，通用 |
| `yolo26m.pt` | Medium | 中等精度，适用于服务器应用 |
| `yolo26l.pt` | Large | 高精度，准确率优先场景 |
| `yolo26x.pt` | XLarge | 最高精度，适用于研究/极端要求 |

### 定向边界框检测 (Oriented Bounding Box Detection)
| 模型名称 | 尺寸 | 描述 |
|------------|------|-------------|
| `yolo26n-obb.pt` | Nano | 旋转目标检测，最小版本 |
| `yolo26s-obb.pt` | Small | 旋转目标检测，平衡版本 |
| `yolo26m-obb.pt` | Medium | 旋转目标检测，中等精度 |
| `yolo26l-obb.pt` | Large | 旋转目标检测，高精度 |
| `yolo26x-obb.pt` | XLarge | 旋转目标检测，最高精度 |

### 实例分割 (Instance Segmentation)
| 模型名称 | 尺寸 | 描述 |
|------------|------|-------------|
| `yolo26n-seg.pt` | Nano | 实例分割，最快版本 |
| `yolo26s-seg.pt` | Small | 实例分割，平衡版本 |
| `yolo26m-seg.pt` | Medium | 实例分割，中等精度 |
| `yolo26l-seg.pt` | Large | 实例分割，高精度 |
| `yolo26x-seg.pt` | XLarge | 实例分割，最高精度 |

### 图像分类 (Image Classification)
| 模型名称 | 尺寸 | 描述 |
|------------|------|-------------|
| `yolo26n-cls.pt` | Nano | 图像分类，最快版本 |
| `yolo26s-cls.pt` | Small | 图像分类，平衡版本 |
| `yolo26m-cls.pt` | Medium | 图像分类，中等精度 |
| `yolo26l-cls.pt` | Large | 图像分类，高精度 |
| `yolo26x-cls.pt` | XLarge | 图像分类，最高精度 |

### 姿态估计 (Pose Estimation)
| 模型名称 | 尺寸 | 描述 |
|------------|------|-------------|
| `yolo26n-pose.pt` | Nano | 姿态估计，最快版本 |
| `yolo26s-pose.pt` | Small | 姿态估计，平衡版本 |
| `yolo26m-pose.pt` | Medium | 姿态估计，中等精度 |
| `yolo26l-pose.pt` | Large | 姿态估计，高精度 |
| `yolo26x-pose.pt` | XLarge | 姿态估计，最高精度 |

## YOLO11 模型 (稳定生产版本)

YOLO11 在所有任务中都提供了出色的性能，推荐用于生产环境。

| 任务 | Nano | Small | Medium | Large | XLarge |
|------|------|-------|--------|-------|--------|
| 检测 (Detection) | `yolo11n.pt` | `yolo11s.pt` | `yolo11m.pt` | `yolo11l.pt` | `yolo11x.pt` |
| 分割 (Segmentation) | `yolo11n-seg.pt` | `yolo11s-seg.pt` | `yolo11m-seg.pt` | `yolo11l-seg.pt` | `yolo11x-seg.pt` |
| 分类 (Classification) | `yolo11n-cls.pt` | `yolo11s-cls.pt` | `yolo11m-cls.pt` | `yolo11l-cls.pt` | `yolo11x-cls.pt` |
| 姿态 (Pose) | `yolo11n-pose.pt` | `yolo11s-pose.pt` | `yolo11m-pose.pt` | `yolo11l-pose.pt` | `yolo11x-pose.pt` |
| 定向检测 (OBB) | `yolo11n-obb.pt` | `yolo11s-obb.pt` | `yolo11m-obb.pt` | `yolo11l-obb.pt` | `yolo11x-obb.pt` |

## 其他支持的 YOLO 版本

Ultralytics 还支持以下 YOLO 版本（仅限推理，不包括训练）：

- **YOLOv10**: 无 NMS 训练，效率-精度架构
- **YOLOv9**: 可编程梯度信息 (PGI) 实现
- **YOLOv8**: 多功能，支持分割、姿态和分类
- **YOLOv5**: 改进的性能和速度权衡
- **YOLOv3**: 经典的实时目标检测

## 使用示例 (Usage Examples)

### Python 代码
```python
from ultralytics import YOLO

# 加载目标检测模型
model = YOLO('yolo26s.pt')  # 或 'yolo11s.pt'

# 加载实例分割模型
model = YOLO('yolo26m-seg.pt')

# 加载姿态估计模型
model = YOLO('yolo26l-pose.pt')
```

### CLI 命令
```bash
# 使用目标检测模型进行预测
yolo predict model=yolo26s.pt source='image.jpg'

# 使用实例分割模型进行预测
yolo predict model=yolo26m-seg.pt source='image.jpg'

# 使用特定的任务模式
yolo detect predict model=yolo26s.pt source='image.jpg'
yolo segment predict model=yolo26m-seg.pt source='image.jpg'
```

## 重要注意事项 (Important Notes)

1. **模型下载**: 在首次使用时，模型会自动从 Ultralytics 服务器下载。
2. **文件扩展名**: 所有模型文件都具有 `.pt` 扩展名，在引用时请包含它。
3. **任务兼容性**: 确保模型类型与您的任务（检测、分割等）相匹配。
4. **硬件要求**: 更大的模型（l, x）需要更多的 GPU 内存和计算资源。
5. **许可证**: 开源使用 AGPL-3.0，商业使用需企业许可证。

## 相关文档 (Related Documentation)

- [模型选择指南](./model_selection.md) - 如何为您的需求选择合适的模型
- [任务类型](./task_types.md) - 计算机视觉任务的详细解释
- [Ultralytics 官方文档](https://docs.ultralytics.com/) - 完整的 API 参考和教程

## 模型管理实用脚本 (Utility Scripts for Model Management)

如需模型选择、验证和管理工具，请使用提供的脚本：

```bash
# 运行模型工具示例
python scripts/model_utils.py

# 测试模型下载和验证
python scripts/quick_tests.py --test download --model yolo26n.pt

# 比较特定任务的模型性能
python scripts/quick_tests.py --test performance --model yolo26n.pt
```

**脚本位置**: `scripts/model_utils.py`

**其他与模型相关的脚本**:
- `scripts/quick_tests.py` - 模型测试和验证
- `scripts/training_helpers.py` - 模型训练和评估
- `scripts/config_templates.py` - 模型配置模板

**优势**:
- 通过提取文档中的代码来节省 tokens
- 即用型的模型管理函数
- 跨所有模型相关任务的一致 API
- 模块化设计，只需导入需要的内容
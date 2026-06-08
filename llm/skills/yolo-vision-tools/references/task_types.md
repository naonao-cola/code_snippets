# YOLO 任务类型详解 (Task Types Explained)

Ultralytics YOLO 支持多种计算机视觉任务，每种任务适用于不同的应用场景。本文档详细解释了每种任务类型及其用例。

**支持的任务**: YOLO26 和 YOLO11 支持所有五种任务：目标检测、实例分割、图像分类、姿态估计和定向边界框检测（OBB）。

## 任务概览 (Task Overview)

| 任务类型 | 输出描述 | 用例 | 模型示例 | 复杂度 |
|-----------|-------------------|-----------|---------------|------------|
| **目标检测 (Object Detection)** | 边界框 + 类别标签 + 置信度 | 通用物体识别 | yolo26n.pt | ★★☆☆☆ |
| **实例分割 (Instance Segmentation)** | 像素级掩码 + 类别标签 | 精确形状分析 | yolo26n-seg.pt | ★★★☆☆ |
| **图像分类 (Image Classification)** | 图像类别 + 置信度 | 整个场景分类 | yolo26n-cls.pt | ★☆☆☆☆ |
| **姿态估计 (Pose Estimation)** | 关键点坐标 + 置信度 | 人体/物体姿态分析 | yolo26n-pose.pt | ★★★★☆ |
| **定向检测 (Oriented Detection)** | 旋转边界框 + 类别标签 | 倾斜物体检测 | yolo26n-obb.pt | ★★★☆☆ |

## 1. 目标检测 (Object Detection)

### 概览
目标检测是最基础且最常用的计算机视觉任务，用于识别图像中物体的位置（边界框）和类别。

### 输出格式
- **边界框**: [中心x, 中心y, 宽度, 高度] 或 [x1, y1, x2, y2] 格式
- **类别标签**: COCO 数据集包含 80 个类别（人、车、动物等）
- **置信度得分**: 0-1 之间的预测置信度

### 用例
- 安防监控: 行人/车辆检测
- 自动驾驶: 交通标志、行人检测
- 工业质检: 缺陷产品检测
- 零售分析: 货架商品检测

### 代码示例

```python
from ultralytics import YOLO

# 加载检测模型
model = YOLO('yolo26n.pt')  # 或 yolo26s.pt, yolo26m.pt 等

# 检测图像中的物体
results = model('bus.jpg')
boxes = results[0].boxes  # 边界框对象
print(f"检测到 {len(boxes)} 个物体")

# 显示结果
for box in boxes:
    print(f"类别: {model.names[int(box.cls)]}, 置信度: {box.conf.item():.2f}")
    print(f"位置: {box.xywh[0].tolist()}")
```

**CLI 命令:**
```bash
yolo detect predict model=yolo26n.pt source='bus.jpg'
```

## 2. 实例分割 (Instance Segmentation)

### 概览
实例分割在目标检测的基础上，为每个物体提供像素级的掩码，能够精确分离重叠的物体。

### 输出格式
- **边界框**: 同目标检测
- **类别标签**: 同目标检测
- **分割掩码**: 标记物体像素的二值掩码矩阵
- **多边形轮廓**: 掩码的轮廓点

### 用例
- 医学图像: 细胞分割、器官分割
- 自动驾驶: 可行驶区域分割
- 遥感图像: 建筑轮廓提取
- 机器人: 用于抓取的精确物体轮廓

### 代码示例

```python
from ultralytics import YOLO

# 加载分割模型
model = YOLO('yolo26n-seg.pt')

# 分割图像
results = model('street.jpg')

# 获取分割结果
masks = results[0].masks  # 掩码对象
if masks is not None:
    print(f"分割了 {len(masks)} 个实例")
    
    # 获取第一个物体的掩码
    mask = masks[0].data.cpu().numpy()
    contours = masks[0].xy  # 轮廓点列表
```

**CLI 命令:**
```bash
yolo segment predict model=yolo26n-seg.pt source='street.jpg'
```

## 3. 图像分类 (Image Classification)

### 概览
图像分类将整张图像归入预定义的类别中，不提供物体的位置信息。

### 输出格式
- **Top-k 类别**: 最可能的 k 个类别（默认 k=5）
- **置信度得分**: 每个类别的预测概率
- **特征向量**: 可选的特征嵌入（Feature embeddings）

### 用例
- 内容审核: NSFW/暴力内容识别
- 场景识别: 室内/室外、白天/黑夜
- 野生动物识别: 物种分类
- 产品分类: 产品类型识别

### 代码示例

```python
from ultralytics import YOLO

# 加载分类模型
model = YOLO('yolo26n-cls.pt')

# 分类图像
results = model('cat.jpg')

# 获取分类结果
probs = results[0].probs  # 概率对象
top5 = probs.top5  # 前 5 个类别的索引
top5conf = probs.top5conf  # 前 5 个类别的置信度

print("前 5 个预测结果:")
for idx, conf in zip(top5, top5conf):
    print(f"  {model.names[idx]}: {conf:.2%}")
```

**CLI 命令:**
```bash
yolo classify predict model=yolo26n-cls.pt source='cat.jpg'
```

## 4. 姿态估计 (Pose Estimation)

### 概览
姿态估计检测物体的关键点（如人体关节），用于分析姿势、动作和行为。

### 输出格式
- **关键点坐标**: [x, y, 可见度] 格式
- **骨架连接**: 关键点之间的关系
- **置信度得分**: 每个关键点的可见度得分

### 用例
- 体育分析: 运动员动作评估
- 健康监测: 老人跌倒检测
- 交互设计: 手势识别
- 安防: 异常行为检测

### 代码示例

```python
from ultralytics import YOLO

# 加载姿态模型
model = YOLO('yolo26n-pose.pt')

# 估计姿态
results = model('yoga.jpg')

# 获取关键点
keypoints = results[0].keypoints  # 关键点对象
if keypoints is not None:
    print(f"检测到 {len(keypoints)} 个人的姿态")
    
    # 获取第一个人的关键点
    kpts = keypoints[0].xy.cpu().numpy()
    confs = keypoints[0].conf.cpu().numpy()
    
    # 显示关键点坐标
    for i, (x, y) in enumerate(kpts):
        print(f"关键点 {i}: ({x:.1f}, {y:.1f}), 置信度: {confs[i]:.2f}")
```

**CLI 命令:**
```bash
yolo pose predict model=yolo26n-pose.pt source='yoga.jpg'
```

## 5. 定向边界框检测 (Oriented Bounding Box Detection)

### 概览
定向边界框检测为边界框添加了旋转角度，使其更适合检测倾斜或旋转的物体。

### 输出格式
- **旋转边界框**: [中心x, 中心y, 宽度, 高度, 角度]
- **类别标签**: 同目标检测
- **置信度得分**: 预测置信度

### 用例
- 遥感图像: 倾斜的建筑物、车辆
- 文档分析: 倾斜的文本区域
- 工业检测: 旋转的机械零件
- 自动驾驶: 倾斜的停车位

### 代码示例

```python
from ultralytics import YOLO

# 加载 OBB 模型
model = YOLO('yolo26n-obb.pt')

# 检测旋转物体
results = model('aerial.jpg')

# 获取旋转边界框
obb = results[0].obb  # OBB 对象
if obb is not None:
    print(f"检测到 {len(obb)} 个旋转物体")
    
    # 获取第一个旋转框的参数
    box = obb.xywhr[0]  # [x, y, w, h, 角度]
    print(f"中心: ({box[0]:.1f}, {box[1]:.1f})")
    print(f"尺寸: {box[2]:.1f}×{box[3]:.1f}, 角度: {box[4]:.1f} rad")
```

**CLI 命令:**
```bash
yolo obb predict model=yolo26n-obb.pt source='aerial.jpg'
```

## 任务选择指南

### 如何选择任务类型？

```mermaid
graph TD
    A[开始] --> B{需要物体位置吗?};
    B -->|否| C[使用图像分类];
    B -->|是| D{需要精确形状信息吗?};
    D -->|否| E[使用目标检测];
    D -->|是| F{需要关键点信息吗?};
    F -->|否| G[使用实例分割];
    F -->|是| H[使用姿态估计];
    E --> I{物体经常倾斜/旋转吗?};
    I -->|是| J[使用定向检测 (OBB)];
    I -->|否| K[使用标准目标检测];
```

### 性能考虑

1. **速度**: 分类 > 检测 > 分割 ≈ 姿态 > 定向检测
2. **精度需求**: 根据应用在速度和精度之间权衡
3. **硬件限制**: 移动端优先小模型，服务器端可使用大模型
4. **数据准备**: 分割和姿态任务需要更详细的标注数据

### 多任务组合

在实际应用中，您可以组合多个任务：

```python
# 先检测后分类 (两阶段处理)
det_model = YOLO('yolo26n.pt')
cls_model = YOLO('yolo26n-cls.pt')

# 检测物体
det_results = det_model('scene.jpg')

# 对每个检测到的物体进行分类
for box in det_results[0].boxes:
    crop = box.xyxy  # 裁剪区域
    # 对裁剪区域进行细粒度分类
    cls_result = cls_model(crop)
```

## 模型文件后缀说明

| 后缀 | 任务类型 | 模型示例 |
|--------|-----------|---------------|
| `.pt` | 目标检测 | yolo26n.pt |
| `-seg.pt` | 实例分割 | yolo26n-seg.pt |
| `-cls.pt` | 图像分类 | yolo26n-cls.pt |
| `-pose.pt` | 姿态估计 | yolo26n-pose.pt |
| `-obb.pt` | 定向检测 | yolo26n-obb.pt |

## YOLO 版本兼容性

- **YOLO26**: 支持所有五种任务，包含最新优化
- **YOLO11**: 支持所有五种任务，生产环境稳定
- **YOLOv8**: 支持检测、分割、分类、姿态
- **YOLOv5**: 主要用于检测，对其他任务支持有限

## 进阶主题

### 特定任务配置

每个任务都有特定的配置参数：
- **检测**: `conf`, `iou`, `classes`, `agnostic_nms`
- **分割**: `mask_ratio`, `retina_masks`
- **分类**: `topk`, `temperature`
- **姿态**: `kpt_shape`, `skeleton`
- **OBB**: `angle_range`, `rotate`

详细示例请参阅 [配置示例](./configuration_samples.md)。

### 自定义任务训练

为特定任务训练自定义模型：
1. 准备特定任务的数据集（检测/分割使用 COCO 格式，分类使用 ImageNet 格式等）
2. 配置特定任务的训练参数
3. 使用合适的模型架构

详细训练指南：[训练基础](./training_basics.md)

## 进一步学习

- [Ultralytics 任务文档](https://docs.ultralytics.com/tasks/)
- [COCO 数据集类别](https://docs.ultralytics.com/datasets/detect/coco/)
- [自定义任务训练指南](https://docs.ultralytics.com/guides/training/)
- [模型选择指南](./model_selection.md)
- [配置示例](./configuration_samples.md)

## 任务测试实用脚本

如需快速测试不同的 YOLO 任务，请使用 `quick_tests.py` 脚本：

```bash
# 测试所有任务 (需要模型文件)
python scripts/quick_tests.py --test all

# 测试特定任务
python scripts/quick_tests.py --test detection
python scripts/quick_tests.py --test segmentation --model yolo26n-seg.pt
python scripts/quick_tests.py --test classification --model yolo26n-cls.pt
python scripts/quick_tests.py --test pose --model yolo26n-pose.pt
python scripts/quick_tests.py --test obb --model yolo26n-obb.pt
```

**脚本位置**: `scripts/quick_tests.py`

**特定任务配置的附加脚本**:
- `scripts/config_templates.py` - 特定任务的配置模板
- `scripts/model_utils.py` - 特定任务的模型选择
- `scripts/training_helpers.py` - 特定任务的训练配置

**优势**:
- 通过提取文档中的代码来节省 tokens
- 快速测试不同的 YOLO 任务
- 一致的测试方法
- 开箱即用，无需编写测试代码

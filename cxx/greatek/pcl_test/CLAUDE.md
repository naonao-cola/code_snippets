# 项目说明

## 构建命令

本项目使用 [xmake](https://xmake.io) 构建：

```bash
xmake                    # 调试构建（默认）
xmake build -m release   # 发布构建
xmake clean              # 清理构建产物
xmake run pcl_test       # 运行主程序
xmake run test_8         # 运行激光雷达采集工具
```

**构建目标：**
- `pcl_test` — 主推理 + 点云处理管道
- `test_8` — 激光雷达数据采集工具

## 架构概览

多模态感知管道：RGB相机 + 深度/IR相机 + 激光雷达，配合GPU加速的深度学习推理。

**数据流：**
1. 传感器输入 → 2. YOLOv8分割推理 → 3. PCL点云处理 → 4. CUDA距离计算

**关键文件：**
- `src/test_5_trt.cpp` — 主入口：TensorRT推理 + PCL处理
- `src/test_8.cpp` — 激光雷达数据采集
- `src/ox_seg.cpp` / `include/ox_seg.h` — 分割模块
- `include/yolov8_seg.hpp` — YOLOv8分割类
- `include/common.hpp` — TensorRT工具、日志、数据结构

**模型**（`model/`目录）：
- `.onnx` 源模型；`.engine` TensorRT编译模型
- `config.yaml` 运行参数配置：相机内参、模型路径、置信度/IOU阈值

## 依赖环境

Windows x64 需安装：
- CUDA 11.8 + cuDNN 8.9.7
- TensorRT 8.6.1
- ONNX Runtime 1.22.1
- OpenCV 4.8.x
- PCL (含VTK)

库路径硬编码在 `xmake.lua` 中，不同机器需修改路径。

## 代码风格

- 函数保持短小精悍，避免过度设计
- 修改时最小化改动范围
- 适当使用设计模式
- 复杂逻辑用 Mermaid 图说明

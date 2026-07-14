# 第三部分：TensorRT 部署 (TensorRT Deployment)

> **定位：** 本模块关注如何将视觉深度学习模型（如 YOLO 等）通过 TensorRT 在 C++ 环境中进行高并发、低延迟的推理部署。核心思想提炼自基于生产环境的 `algo_ai` 架构。

> **💡 代码参考**：有关 TRT Builder、异步推理控制器、GPU 内存池以及 YOLO 的 CUDA 前后处理细节，请参考模板脚本：
> - `../scripts/templates/4_deployment_tensorrt/trt_1_builder.cpp`
> - `../scripts/templates/4_deployment_tensorrt/trt_2_memory_tensor.hpp`
> - `../scripts/templates/4_deployment_tensorrt/trt_3_infer_controller.hpp`
> - `../scripts/templates/4_deployment_tensorrt/models/` (包含各类模型 100% 完整的 C++ / CUDA 前后处理实现源码)

## 1. 模型转换与引擎构建 (TRT Builder)
- **ONNX 解析**：使用 `nvonnxparser` 解析 ONNX 模型，并结合 `nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH` 显式指定 Batch 维度。
- **动态 Batch 支持**：通过配置 `OptimizationProfile`，为输入 Tensor 设置 `kMIN`, `kOPT`, `kMAX` 维度的形状，从而允许在运行时动态调整 Batch Size。
- **精度模式**：支持 `FP32` 与 `FP16`。当开启 `FP16` 时，通过 `builder->platformHasFastFp16()` 检查硬件兼容性，并设置 `BuilderFlag::kFP16` 标志。

## 2. 混合内存池与张量管理 (Memory & Tensor)
- **MixMemory 架构**：统一管理 Pinned CPU Memory (`cudaMallocHost`) 和 GPU Device Memory (`cudaMalloc`)。调用 `.cpu()` 或 `.gpu()` 时，系统会按需自动分配显存并记录归属权。
- **MonopolyAllocator 独占式分配器**：
  - 由于高并发下频繁调用 `cudaMalloc/cudaFree` 极其耗时，通过 `MonopolyAllocator` 预先分配一定容量（如 2 倍 `max_batch_size`）的内存池。
  - 内部利用 `std::condition_variable` 控制获取与释放，保证在多线程下数据的线程安全性，实现 Zero-Allocation 推理。

## 3. 高并发异步调度器与多流推理 (Inference Controller & Multi-Stream)
- **调度框架 (`AIAlgoBase` & `InferController`)**：
  - **解耦设计**：采用生产者-消费者模型，通过 `MPMCQueue` (多生产者多消费者队列) 连接前处理 (`PreProcessWorker`)、模型推理和后处理 (`PostProcessWorker`)。
  - **流水线架构 (Pipeline)**：外部调用 `commit(Input)` 时，任务被包装为 `Job` 压入队列并返回 `std::shared_future<Output>`。主线程无阻塞，后台 `Worker` 线程负责异步组批 (Dynamic Batching) 和推理。
- **多流推理与多上下文机制 (`MutiContext` 架构)**：
  - **多 Context 复用**：反序列化模型后，`ICudaEngine` 全局唯一。利用它创建多个 `IExecutionContext`（即 `MutiContext` 实例）。在多线程环境中，每个线程绑定独立的 Context 和专属的显存 Workspace，彻底消除并发推理时的锁竞争。
  - **`std::enable_shared_from_this` 技法**：在 `MutiContext::create_context` 内部，通过继承该模板类，能够安全地调用 `shared_from_this()`，将自身的 `shared_ptr` 注册到全局 `InferEngine` 的 `context_vec_` 中进行统一的生命周期管理，避免了“双重释放 (double free)”或悬空指针问题。
  - **独立 Stream 绑定**：每个 `MutiContext` 创建时都会绑定一个独占的 `cudaStream_t`。通过 `enqueueV2(bindingsptr, stream_, nullptr)`，将推理任务提交到不同的流中，配合 TensorRT 的异步执行，真正实现了硬件级的计算流水线重叠。

## 4. 各类模型的前后处理流程 (Pre/Post Processing)
> **💡 源码级参考**：所有视觉模型的完整 C++ / CUDA 前后处理实现（绝非仅有头文件的占位符）均已 100% 无损沉淀至 `../scripts/templates/4_deployment_tensorrt/models/` 目录中。

- **通用预处理 (`models/trt_cuda/preprocess_kernel.cu`)**：
  - 摒弃了低效的 CPU `cv::resize` 和 `cv::cvtColor`。
  - 利用 CUDA Kernel 并行计算仿射变换矩阵（`d2i` 和 `i2d`），一步完成图像缩放、平移对齐、色彩空间转换及归一化 (Normalize)。
- **各模型独占后处理 (YOLO, OCR, Seg, OBB, Anomalib, DEIM)**：
  - **YOLO (`models/trt_app_yolo/`)**：GPU 端 Decode 解析输出特征图，Fast NMS Kernel 利用双重循环将耗时的 IoU 计算下放至 GPU。
  - **YOLOv8-OBB (`models/trt_app_yolo8_obb/`)**：基于高斯协方差矩阵的 ProbIoU 实现旋转框 GPU NMS。
  - **OCR (`models/trt_app_ocr/`)**：DBNet 文本检测的轮廓提取与 CRNN 文本识别的 CTC 贪婪解码实现。
  - **Anomalib (`models/trt_app_anomalib/`)**：异常检测的热力图伪彩色渲染与像素级极值归一化。
  - **DEIM / MSAE / 分割 (`models/trt_app_*/`)**：多输出 Tensor 的跨设备拷贝流转与 OpenCV 轮廓处理逻辑。

## 5. 自定义算子插件 (TensorRT Plugin)
> **💡 代码参考**：有关 TensorRT Plugin 的具体结构与 CUDA Kernel 实现，请查阅 `../scripts/templates/4_deployment_tensorrt/trt_6_plugin.hpp` 和 `.cu`。

- **架构规范**：编写 TRT Plugin 必须实现两个核心类：
  - **`IPluginV2DynamicExt`**：插件本体类。负责 `getOutputDimensions` (推导输出张量维度)、`supportsFormatCombination` (验证数据类型如 FP32/FP16) 以及 `enqueue` (实际调用 CUDA Kernel 执行计算)。
  - **`IPluginCreator`**：插件工厂类。负责在 ONNX Parse 阶段根据算子名称进行拦截匹配，通过 `PluginFieldCollection` 获取权重和参数并实例化插件；同时在反序列化阶段通过 `deserializePlugin` 重建插件。
- **宏注册机制**：必须使用 `REGISTER_TENSORRT_PLUGIN(CustomPluginCreator)` 宏将 Creator 注册到 TRT 的全局插件注册表中。
- **CUDA 异步执行**：在 `enqueue` 中启动 Kernel 时，必须严格传入 TRT 传递过来的 `cudaStream_t`，确保插件的计算与整个模型的异步流水线保持绝对同步。

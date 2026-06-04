# 自动库检索与配置指南

当用户要求添加一个本地库时，请按照以下逻辑进行操作，以确保配置的准确性。

## 1. 目录结构探索
首先使用 `LS` 工具查看目标目录。典型的库结构如下：
- `include/` 或 `inc/`: 存放头文件。
- `lib/`: 存放静态库或导入库 (`.lib`, `.a`)。
- `bin/`: 存放动态库 (`.dll`, `.so`)。
- `build/`: 有些库（如 OpenCV）的编译产物在 `build` 目录下，需要深入查看 `build/x64/vc16/lib` 等。

## 2. 路径提取逻辑
- **头文件**: 寻找包含 `xxx.h` 或 `xxx.hpp` 的最顶层目录。
- **库文件**: 寻找包含大量 `.lib` 或 `.so` 文件的目录。
- **运行时 (rpath)**: 寻找包含 `.dll` 或 `.so` 的目录，通常在 `bin` 目录下，用于 `add_rpathdirs`。

## 3. 链接库提取规则
在提取 `add_links` 时，遵循以下规则：
- **Windows**: 去掉 `.lib` 后缀。例如 `opencv_world481.lib` -> `opencv_world481`。
- **Linux**: 去掉 `lib` 前缀和 `.so/.a` 后缀。例如 `libnvinfer.so` -> `nvinfer`。
- **过滤**: 忽略非库文件（如日志、文本文件）。

## 4. xmake.lua 更新策略
- 优先查找现有的 `target`。
- 如果是新库，可以创建一个 `rule` 或直接在 `target` 中添加。
- 建议使用相对路径（如果库在项目内）或使用 `$(projectdir)`。

## 5. 示例流程
用户问："把我的 TensorRT 加进来，路径是 E:\3rdparty\TensorRT-8.6"
1. `LS("E:\3rdparty\TensorRT-8.6")` -> 发现 `include`, `lib`, `bin`。
2. `LS("E:\3rdparty\TensorRT-8.6\lib")` -> 发现 `nvinfer.lib`, `nvparsers.lib` 等。
3. 生成代码：
   ```lua
   add_includedirs("E:/3rdparty/TensorRT-8.6/include")
   add_linkdirs("E:/3rdparty/TensorRT-8.6/lib")
   add_links("nvinfer", "nvparsers", "nvinfer_plugin", "nvonnxparser")
   add_rpathdirs("E:/3rdparty/TensorRT-8.6/bin")
   ```

# Ultralytics YOLO 安装指南 (Installation Guide)

本指南提供了 Ultralytics YOLO 的全面安装方法，涵盖了各种环境和部署选项。

**最新版本**: YOLO26（2026 年 1 月发布），具有端到端无 NMS 推理特性。对于稳定的生产环境，推荐使用 YOLO26 和 YOLO11。

## 系统要求 (System Requirements)

### 🎯 最低要求 (必须满足)

| 组件 | 最低要求 | 备注 |
|-----------|---------------------|-------|
| **Python** | **3.8+** | 官方支持 Python 3.8, 3.9, 3.10, 3.11, 3.12 |
| **PyTorch** | **1.8.0+** | Windows 用户: 避免使用 torch 2.4.0 (已知有 CPU Bug) |
| **操作系统** | **Linux, Windows, macOS** | 跨平台支持 |

> ⚠️ **重要**: 安装 `ultralytics` 时会自动安装所有必需的依赖项。**请勿手动安装下面列出的库**。

### 📦 核心依赖 (自动安装)
当运行 `pip install ultralytics` 时，将随附安装以下库（最低版本要求）：

| 库 | 最低版本 | 描述 |
|---------|-----------------|-------------|
| `torch` | 1.8.0+ | PyTorch 深度学习框架 |
| `torchvision` | 0.9.0+ | 计算机视觉数据集和模型 |
| `numpy` | 1.23.0+ | 数值计算库 |
| `matplotlib` | 3.3.0+ | 数据可视化 |
| `opencv-python` | 4.6.0+ | 计算机视觉库 |
| `pillow` (PIL) | 7.1.2+ | 图像处理库 |
| `pyyaml` | 5.3.1+ | YAML 配置文件处理 |
| `requests` | 2.23.0+ | HTTP 请求库 |
| `scipy` | 1.4.1+ | 科学计算库 |
| `psutil` | 5.8.0+ | 系统监控和进程管理 |
| `polars` | 0.20.0+ | 高性能数据处理 |
| `ultralytics-thop` | 2.0.18+ | FLOPs 计算工具 |

### 🎪 可选功能 (按需安装)
为了获得特定功能，可以安装额外的依赖包：

```bash
# 开发工具包 (适用于代码贡献者)
pip install "ultralytics[dev]"

# 模型导出包 (支持 ONNX, TensorFlow, CoreML 等)
pip install "ultralytics[export]"

# 解决方案包 (Streamlit, Flask Web 应用)
pip install "ultralytics[solutions]"

# 日志包 (W&B, TensorBoard, MLflow)
pip install "ultralytics[logging]"

# 额外功能包 (Albumentations 数据增强, COCO 评估)
pip install "ultralytics[extra]"
```

### ⚡ GPU 加速 (可选)
如果需要 GPU 加速，请确保：
1. 拥有支持 CUDA 的 NVIDIA GPU
2. 安装了匹配的 NVIDIA 驱动程序
3. 推荐使用 CUDA 12.1+ 版本

### 🚀 快速验证
安装完成后，验证您的环境：

```bash
# 检查版本
python -c "import ultralytics; print(f'Ultralytics version: {ultralytics.__version__}')"

# 简单测试
yolo predict model=yolo26n.pt source='https://ultralytics.com/images/bus.jpg'
```

## 安装方法 (Installation Methods)

### 1. 基础安装 (推荐)

使用 pip 是最简单、最快的方法：

```bash
# 安装最新版本
pip install ultralytics

# 或安装特定版本
pip install ultralytics==8.4.0

# 升级到最新版本
pip install -U ultralytics
```

### 2. 从源码安装 (开发版)

如果您需要最新的开发特性或想要修改源代码：

```bash
# 克隆仓库
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics

# 以可编辑模式安装 (开发环境)
pip install -e .

# 或者直接从 GitHub 安装
pip install git+https://github.com/ultralytics/ultralytics.git

# 从特定分支安装
pip install git+https://github.com/ultralytics/ultralytics.git@main
```

### 3. Conda 安装 (Anaconda/Miniconda)

```bash
# 创建新环境 (可选)
conda create -n yolo python=3.10
conda activate yolo

# 通过 conda-forge 安装
conda install -c conda-forge ultralytics

# 或者先安装 PyTorch，再安装 ultralytics
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install ultralytics

# 一次性安装所有包 (CUDA 环境)
conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=11.8 ultralytics
```

### 4. Docker 安装 (容器化部署)

```bash
# 拉取官方镜像
docker pull ultralytics/ultralytics:latest

# 运行支持 GPU 的容器
docker run -it --ipc=host --runtime=nvidia --gpus all ultralytics/ultralytics:latest

# 挂载本地目录并运行
docker run -it --rm --gpus all -v $(pwd):/workspace ultralytics/ultralytics:latest

# 指定 GPU 设备
docker run -it --ipc=host --runtime=nvidia --gpus '"device=2,3"' ultralytics/ultralytics:latest
```

可用的 Docker 镜像：
- `ultralytics/ultralytics:latest` - 推荐用于训练的 GPU 镜像
- `ultralytics/ultralytics:latest-cpu` - 仅限 CPU 的推理版本
- `ultralytics/ultralytics:latest-arm64` - 为 ARM64 (树莓派) 优化
- `ultralytics/ultralytics:latest-jetson` - 专为 NVIDIA Jetson 设备定制
- `ultralytics/ultralytics:latest-conda` - 基于 Miniconda3 的镜像

有关更多 Docker 选项，请参阅 [Ultralytics Docker 指南](https://docs.ultralytics.com/guides/docker-quickstart/)。

### 5. 无头服务器安装 (Headless Server)

对于没有显示器的服务器环境（云 VM、Docker 容器、CI/CD 流水线）：

```bash
# 使用无头变体 (无 GUI 依赖)
pip install ultralytics-opencv-headless

# 这两个包提供相同的功能和 API
# 无头变体排除了 OpenCV 的 GUI 组件
```

## GPU 支持配置

### CUDA 环境设置 (NVIDIA GPU)

```bash
# 方法 1: 使用官方推荐命令 (自动匹配 CUDA 版本)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics

# 方法 2: 指定 CUDA 版本
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 安装后检查 CUDA 是否可用
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

### 仅 CPU 安装

```bash
# 仅 CPU 版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics

# 或通过 conda 安装 CPU 版 PyTorch
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

## 高级安装方法

### 从 Fork 安装

```bash
# 在 GitHub 上 Fork Ultralytics 仓库
# 克隆您的 Fork
git clone https://github.com/YOUR_USERNAME/ultralytics.git
cd ultralytics

# 创建修改分支
git checkout -b my-custom-branch

# 修改 pyproject.toml 或其他文件
# 从您的分支安装
pip install git+https://github.com/YOUR_USERNAME/ultralytics.git@my-custom-branch
```

### 使用 requirements.txt

```bash
# requirements.txt
git+https://github.com/YOUR_USERNAME/ultralytics.git@my-custom-branch
flask
# 其他项目依赖

# 安装依赖
pip install -r requirements.txt
```

## 安装验证

安装后，验证是否成功：

```python
import ultralytics
print(f"Ultralytics version: {ultralytics.__version__}")

# 简单测试
from ultralytics import YOLO

# 加载预训练模型
model = YOLO('yolo26n.pt')

# 快速推理测试
results = model('https://ultralytics.com/images/bus.jpg')
print(f"检测到 {len(results[0].boxes)} 个物体")
```

命令行验证：

```bash
# 显示版本信息
yolo version

# 运行快速测试
yolo predict model=yolo26n.pt source='https://ultralytics.com/images/bus.jpg'

# 全面的环境检查
yolo checks
```

## 故障排除 (Troubleshooting)

### 常见问题

1. **ImportError: libcudart.so.xx.x: cannot open shared object file**
   - 确保 CUDA 正确安装且版本匹配
   - 检查 `LD_LIBRARY_PATH` 环境变量
   - 验证 NVIDIA 驱动程序是最新的

2. **PyTorch 版本不兼容**
   ```bash
   # 重新安装匹配的 PyTorch 版本
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

3. **安装缓慢或超时**
   ```bash
   # 使用国内镜像源 (如清华源)
   pip install ultralytics -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

4. **找不到 `yolo` 命令**
   ```bash
   # 使用 Python 模块语法
   python -m ultralytics yolo version
   
   # 或检查 Python 环境变量 PATH
   which python
   pip show ultralytics
   ```

5. **模型下载内存/存储问题**
   ```bash
   # 设置缓存目录
   export ULTRALYTICS_HOME=/path/to/cache
   ```

### 环境检查脚本

使用内置的环境检查脚本：

```bash
# 运行全面环境检查
python scripts/check_environment.py

# 或使用内置检查
yolo checks
```

## 平台特定注意事项

### Windows
- 推荐: 使用 Python 3.10 或 3.11
- 避免使用 PyTorch 2.4.0 (已知有 CPU Bug)
- 为了更好的兼容性，可使用 Windows Subsystem for Linux (WSL2)

### macOS
- M1/M2/M3 Apple Silicon: 使用 PyTorch nightly builds 获得最佳性能
- Intel Macs: 仅支持 CPU 运行

### Linux
- 兼容性最好的平台
- 使用系统包管理器管理底层依赖 (apt, yum, dnf)

### ARM64 (Raspberry Pi, Jetson)
- 使用专门为 ARM64 提供的 Docker 镜像
- 考虑模型量化以提升性能
- 使用较小的模型 (nano, small) 进行实时推理

## 进一步操作

安装成功后：
1. 运行 `yolo checks` 验证环境
2. 浏览 [任务类型](./task_types.md) 了解 YOLO 功能
3. 查看 [模型选择](./model_selection.md) 选择合适的模型
4. 查看 [配置示例](./configuration_samples.md) 了解参数调整

## 验证用的实用脚本

如需进行全面的环境验证，请使用提供的脚本：

```bash
# 快速环境检查
python scripts/check_environment.py

# 快速功能测试
python scripts/quick_tests.py --test environment

# 测试基础推理
python scripts/quick_tests.py --test inference
```

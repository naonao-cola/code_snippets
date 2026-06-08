# Ultralytics YOLO 环境检查指南 (Environment Check Guide)

本文档提供了检查 Ultralytics YOLO 环境的全面方法，帮助您验证安装是否正确、环境配置是否合理以及 GPU/CUDA 是否可用。

## 1. 快速检查 (推荐)

### 使用官方 `yolo checks` 命令
这是最全面的检查方法，它会输出 Python 版本、PyTorch、CUDA、GPU 信息以及所有依赖库的状态。

```bash
yolo checks
```

**预期输出示例:**
```
Ultralytics 8.4.21 🚀 Python-3.11.2 torch-2.10.0+cu128 CUDA:0 (NVIDIA GeForce RTX 3060, 6144MiB)
Setup complete ✅ (72 checks passed in 1.4s)

OS                     Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.35
Environment            Docker
Python                 3.11.2
Install                pip
Path                   /usr/local/lib/python3.11/site-packages/ultralytics
RAM                    31.2GB
Disk                   464.5GB
CPU                    Intel Xeon (4) @ 2.200GHz
CPU count              4
GPU                    NVIDIA GeForce RTX 3060 Laptop GPU
GPU count              1
CUDA                   12.8

numpy                  ✅ 2.4.3>=1.23.0
matplotlib             ✅ 3.10.8>=3.3.0
opencv-python          ✅ 4.13.0.92>=4.6.0
pillow                 ✅ 12.1.1>=7.1.2
pyyaml                 ✅ 6.0.3>=5.3.1
requests               ✅ 2.32.5>=2.23.0
scipy                  ✅ 1.14.1>=1.4.1
torch                  ✅ 2.10.0>=2.5.1
torchvision            ✅ 0.20.0>=0.20.1
...
```

**关键信息解读:**
- 第一行: 显示 YOLO 版本、Python 版本、PyTorch 版本、CUDA 版本、GPU 型号和显存
- `Setup complete`: 环境已准备就绪
- 依赖状态: ✅ 表示已安装且版本符合要求

## 2. 基础验证方法

### 2.1 Python 导入测试
验证 Ultralytics 包是否可导入并检查版本：

```bash
python -c "import ultralytics; print(f'Ultralytics version: {ultralytics.__version__}')"
```

**输出示例:**
```
Ultralytics version: 8.4.21
```

### 2.2 包信息检查
查看已安装的 ultralytics 包的详细信息：

```bash
pip show ultralytics
```

如果存在多个 Python 环境，请使用：

```bash
python -m pip show ultralytics
```

**输出包括:** 版本号、安装路径、依赖项等。

### 2.3 包列表检查
检查 ultralytics 是否存在于已安装的包列表中：

```bash
pip list | grep -i ultralytics
```

**或使用 conda (如果适用):**
```bash
conda list | grep -i ultralytics
```

## 3. CLI 命令行工具验证

### 3.1 检查 `yolo` 命令是否可用
```bash
yolo --help
```

**预期输出:** 显示帮助信息，包括可用任务（detect, segment, classify, pose, obb）和模式（train, val, predict, export, track, benchmark）。

### 3.2 简单推理测试
使用最小的预训练模型进行快速推理测试：

```bash
yolo predict model=yolo26n.pt source='https://ultralytics.com/images/bus.jpg' verbose=False
```

如果环境正常，它会下载模型（首次运行）并输出检测结果。

## 4. GPU/CUDA 环境验证

### 4.1 检查 PyTorch 的 CUDA 支持
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
```

**输出示例:**
```
PyTorch version: 2.10.0+cu128
CUDA available: True
GPU count: 1
GPU 0: NVIDIA GeForce RTX 3060 Laptop GPU
```

### 4.2 检查 CUDA 版本兼容性
```bash
python -c "import torch; print(f'CUDA version (PyTorch): {torch.version.cuda}'); print(f'CUDA runtime version: {torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}')"
```

### 4.3 验证 GPU 显存可用性
```bash
python -c "import torch; print(f'GPU memory total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB'); print(f'GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB'); print(f'GPU memory cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB')"
```

## 5. 依赖库完整性检查

### 5.1 关键依赖版本验证
```bash
python -c "
import torch, torchvision, cv2, PIL, numpy, pandas
print(f'torch: {torch.__version__}')
print(f'torchvision: {torchvision.__version__}')
print(f'opencv-python: {cv2.__version__}')
print(f'Pillow: {PIL.__version__}')
print(f'numpy: {numpy.__version__}')
print(f'pandas: {pandas.__version__}')
"
```

### 5.2 检查缺失的依赖项
Ultralytics 的最低版本要求：
- `torch>=2.5.1`
- `torchvision>=0.20.1`
- `opencv-python>=4.10.0`
- `pillow>=10.3.0`

检查版本是否满足要求：
```bash
pip list | grep -E "torch|torchvision|opencv-python|pillow"
```

## 6. 环境问题诊断 (Troubleshooting)

### 6.1 常见问题与解决方案

**问题: 找不到 `yolo` 命令**
```
bash: yolo: command not found
```
**解决方案:**
1. 检查 Python 环境: `which python` 和 `which pip`
2. 确保 ultralytics 安装在当前环境中: `pip list | grep ultralytics`
3. 使用完整路径: `python -m ultralytics yolo checks`
4. 检查 PATH 环境变量是否包含 Python 脚本目录

**问题: CUDA 不可用**
```
CUDA available: False
```
**解决方案:**
1. 确认安装了兼容 CUDA 的 PyTorch: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`
2. 检查 NVIDIA 驱动: `nvidia-smi`
3. 验证 CUDA toolkit 安装: `nvcc --version`
4. 确保 PyTorch 的 CUDA 版本与系统 CUDA 版本兼容

**问题: 依赖版本冲突**
```
ImportError: cannot import name 'xxx' from 'ultralytics'
```
**解决方案:**
1. 升级到最新版本: `pip install -U ultralytics`
2. 重新安装: `pip uninstall ultralytics -y && pip install ultralytics`
3. 检查依赖兼容性: `pip check`

### 6.2 完整环境报告
生成用于故障排除的完整环境报告：

```bash
python -c "
import sys, platform, torch, ultralytics
print('='*60)
print('Ultralytics YOLO Environment Report')
print('='*60)
print(f'Python: {sys.version}')
print(f'Platform: {platform.platform()}')
print(f'Ultralytics: {ultralytics.__version__}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
print('='*60)
"
```

## 7. 自动化检查脚本

### 7.1 使用提供的检查脚本
使用随附的 Python 脚本进行全面的环境检查：

```bash
# 运行环境检查脚本
python scripts/check_environment.py

# 此脚本提供有关以下内容的详细信息：
# - Python 环境
# - PyTorch 配置和 CUDA 可用性
# - Ultralytics 安装状态
# - 关键依赖版本
# - 系统资源 (CPU, RAM, 磁盘)
# - 生成可用于分享的 JSON 报告
```

### 7.2 简易检查脚本
将以下内容保存为 `quick_check.py` 用于基础验证：

```python
#!/usr/bin/env python3
"""
Ultralytics YOLO Quick Environment Check
Run: python quick_check.py
"""

import sys
import platform
import subprocess

def run_command(cmd):
    """执行命令并返回输出"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else f"Error: {result.stderr.strip()}"
    except Exception as e:
        return f"Exception: {str(e)}"

def main():
    print("="*60)
    print("Ultralytics YOLO 快速环境检查")
    print("="*60)
    
    # 1. 系统信息
    print("\n1. 系统信息:")
    print(f"   Python: {sys.version.split()[0]}")
    print(f"   Platform: {platform.platform()}")
    
    # 2. Ultralytics 检查
    print("\n2. Ultralytics 检查:")
    import_test = run_command('python -c "import ultralytics; print(\"OK\")"')
    version_test = run_command('python -c "import ultralytics; print(ultralytics.__version__)"')
    print(f"   导入测试: {import_test}")
    print(f"   版本: {version_test}")
    
    # 3. PyTorch 检查
    print("\n3. PyTorch 检查:")
    torch_test = run_command('python -c "import torch; print(\"OK\")"')
    torch_version = run_command('python -c "import torch; print(torch.__version__)"')
    cuda_test = run_command('python -c "import torch; print(torch.cuda.is_available())"')
    print(f"   导入测试: {torch_test}")
    print(f"   版本: {torch_version}")
    print(f"   CUDA 可用: {cuda_test}")
    
    # 4. yolo 命令行工具检查
    print("\n4. yolo 命令行工具:")
    yolo_check = run_command('which yolo 2>/dev/null || echo "Not found"')
    print(f"   命令可用性: {yolo_check}")
    
    print("\n" + "="*60)
    print("检查完成!")
    print("="*60)

if __name__ == "__main__":
    main()
```

## 8. 性能测试

### 8.1 推理速度基准测试
测试不同模型的推理速度：

```bash
# 测试 nano 模型
yolo benchmark model=yolo26n.pt imgsz=640 device=0

# 测试 medium 模型
yolo benchmark model=yolo26m.pt imgsz=640 device=0

# 比较 CPU vs GPU
yolo benchmark model=yolo26n.pt imgsz=640 device=cpu
yolo benchmark model=yolo26n.pt imgsz=640 device=0
```

### 8.2 显存使用测试
检查推理期间的显存使用情况：

```bash
# 监控推理时的 GPU 显存
yolo predict model=yolo26m.pt source='image.jpg' device=0
# 在另一个终端中使用 nvidia-smi 检查显存使用情况
```

## 9. YOLO 版本特定检查

### 9.1 YOLO26 特定检查
YOLO26 引入了新特性，可能需要特定的检查：

```bash
# 检查 YOLO26 模型加载
yolo predict model=yolo26n.pt source='image.jpg'

# 测试无 NMS 推理 (YOLO26 特性)
yolo predict model=yolo26n.pt source='image.jpg' nms=False
```

### 9.2 多任务模型验证
验证模型是否支持特定任务：

```bash
# 检测模型检查
yolo detect predict model=yolo26n.pt source='image.jpg'

# 分割模型检查
yolo segment predict model=yolo26n-seg.pt source='image.jpg'

# 姿态模型检查
yolo pose predict model=yolo26n-pose.pt source='image.jpg'

# 分类模型检查
yolo classify predict model=yolo26n-cls.pt source='image.jpg'

# OBB 模型检查
yolo obb predict model=yolo26n-obb.pt source='image.jpg'
```

## 10. 网络与下载检查

### 10.1 模型下载测试
测试模型下载功能：

```bash
# 测试模型下载 (如果未下载，将被缓存)
yolo predict model=yolo26n.pt source='https://ultralytics.com/images/bus.jpg' verbose=False

# 检查缓存位置
python -c "from ultralytics import YOLO; import os; print(f'模型缓存: {os.path.expanduser(\"~/.cache/ultralytics\")}')"
```

### 10.2 网络连接检查
```bash
# 测试连接到 Ultralytics 服务器
curl -I https://ultralytics.com/images/bus.jpg

# 测试 GitHub API (用于源码安装)
curl -I https://api.github.com/repos/ultralytics/ultralytics
```

## 总结

定期的环境检查确保 YOLO 任务的稳定执行。建议使用 `yolo checks` 进行全面检查，并在出现问题时参考本文档中的针对性验证方法。

**检查优先级:**
1. ✅ `yolo checks` - 全面检查
2. ✅ Python 导入测试 - 基础验证
3. ✅ GPU/CUDA 验证 - 性能相关
4. ✅ 依赖版本检查 - 兼容性验证

**附加工具:**
- 使用 `scripts/check_environment.py` 获取详细的诊断信息
- 运行 `yolo benchmark` 进行性能测试
- 使用 `nvidia-smi` 监控 GPU 利用率

**提示:** 环境检查通过后，即可继续执行 YOLO 任务（如推理、训练等）。有关安装帮助，请参阅 [安装指南](./installation_guide.md)；有关理解 YOLO 功能，请参阅 [任务类型](./task_types.md)。

## 实用脚本

如需进行全面的环境检查，请使用 `check_environment.py` 脚本：

```bash
# 运行全面环境检查
python scripts/check_environment.py

# 此脚本提供：
# - 详细的 Python 环境信息
# - PyTorch 和 CUDA 配置检查
# - Ultralytics 安装验证
# - 关键依赖版本验证
# - 系统资源分析 (CPU, RAM, 磁盘)
# - 生成用于排错的 JSON 报告
```

**脚本位置**: `scripts/check_environment.py`

**额外的测试脚本**:
- `scripts/quick_tests.py` - 快速功能测试
- `scripts/model_utils.py` - 模型验证工具

**使用脚本的好处**:
- 通过提取文档中的代码来节省 tokens
- 一致且全面的检查
- 生成 JSON 报告以便于故障排除
- 开箱即用，无需复制粘贴命令

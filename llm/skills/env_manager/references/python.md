# Python 环境创建、源配置与包名验证指南

旨在加速 Python 环境准备，并确保通过联网搜索获取准确的 Pip 安装包名称，避免因包名不一致导致的安装失败。

## 1. 核心功能

1. **包名自动验证 (重要)**：在安装前，如果包名可能存在歧义（如 `opencv`, `yaml`, `pil`, `sklearn`），**必须先调用联网搜索**确认其在 PyPI 上的准确安装名称。
2. **快速创建 Conda 环境**：自动处理环境名和 Python 版本。
3. **Pip 国内源配置**：全局或单次命令配置（清华、阿里、中科大等）。
4. **Conda 国内源配置**：配置镜像站以加速 Conda 通道。

## 2. 包名验证逻辑
对于用户提出的安装请求，按以下步骤操作：
1. **识别模糊包名**：判断用户提供的包名是否为常用别名（例如：用户说 `opencv`，实际包名是 `opencv-python`；用户说 `yaml`，实际是 `pyyaml`）。
2. **环境依赖隔离诊断**：检查当前目录是否存在 `pyproject.toml` 或 `uv.lock`。如果有，必须使用对应工具（Poetry 或 uv）执行添加包操作，**绝对禁止**直接使用 pip 破坏原生依赖树。
3. **联网搜索确认**：如果不确定准确名称，**必须使用 WebSearch** 搜索 `pip install <package_name> correct name`。
4. **执行安装**：使用确认后的准确包名，并配合镜像源。

### 常见别名对照表 (示例)
- `opencv` -> `opencv-python`
- `pil` -> `Pillow`
- `yaml` -> `pyyaml`
- `sklearn` -> `scikit-learn`
- `tensorrt` -> (需根据 CUDA 版本确认具体 wheel 路径或包名)

## 3. 操作模式与验证

### 验证并安装包
```powershell
# 步骤 1: 联网搜索 (如果包名不确定)
# WebSearch: "pip install opencv correct package name"

# 步骤 2: 安装确认后的包 (带镜像源)
conda run -n <env_name> python -m pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 创建环境并安装基础包
当用户要求创建环境时，优先使用以下模式：
```powershell
# 1. 创建环境
conda create -n <env_name> python=<version> -y

# 2. 激活并安装包 (带清华源)
conda run -n <env_name> python -m pip install <packages> -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 环境验证
安装完成后，必须运行简单的 Python 脚本验证关键包（尤其是 GPU 相关包）是否可用：
```powershell
# 验证 GPU 可用性
conda run -n <env_name> python -c "import torch; print(torch.cuda.is_available())"
# 验证 TensorRT
conda run -n <env_name> python -c "import tensorrt; print(tensorrt.__version__)"
```

## 4. 常用国内源地址与配置

### Pip 镜像源
- **清华 (推荐)**: `https://pypi.tuna.tsinghua.edu.cn/simple`
- **阿里**: `https://mirrors.aliyun.com/pypi/simple/`
- **中科大**: `https://mirrors.ustc.edu.cn/pypi/web/simple/`
- **华为**: `https://mirrors.huaweicloud.com/repository/pypi/simple/`
- **豆瓣**: `https://pypi.doubanio.com/simple/`

### Conda 镜像源配置 (推荐清华大学或北京外国语大学)

**清华大学镜像源 (Tsinghua)**
```powershell
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
conda config --set show_channel_urls yes
```

**北京外国语大学镜像源 (BFSU)**
```powershell
conda config --add channels https://mirrors.bfsu.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.bfsu.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.bfsu.edu.cn/anaconda/cloud/conda-forge/
conda config --set show_channel_urls yes
```

**华为镜像源 (Huawei)**
```powershell
conda config --add channels https://mirrors.huaweicloud.com/repository/anaconda/pkgs/main/
conda config --add channels https://mirrors.huaweicloud.com/repository/anaconda/pkgs/free/
conda config --set show_channel_urls yes
```

**配置全局 Pip 镜像源 (永久生效)**
```powershell
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

## 5. 安装优先级与依赖冲突解决

### 现代包管理生态补充 (P10 视野)
除了传统的 Conda/Pip，必须根据项目特性评估是否引入现代工具：
- **uv**: 极速的 Python 包安装器，直接替代 pip。如果你发现 pip 依赖解析太慢，立刻使用 `uv pip install`。
- **Poetry**: 如果是一个需要发布的库或大型应用，推荐使用 Poetry 生成 `pyproject.toml` 锁定依赖树，而不是手写 `requirements.txt`。

### 安装工具优先级 (Conda vs Pip vs UV)
在 Conda 环境中，应遵循以下优先级：
- **优先使用 Conda (`conda install`)**:
    - 适用于复杂的二进制包（如 `pytorch`, `cudatoolkit`, `opencv`, `tensorflow`）。
    - Conda 能够更好地管理非 Python 依赖（如 CUDA 库、C++ 运行时）。
    - 建议：`conda install <package> -c conda-forge`。
- **次选使用 Pip (`pip install`)**:
    - 适用于仅 Python 的包、最新的开发版或 Conda 仓库中不存在的包。
    - 注意：一旦开始使用 Pip 安装包，后续应尽量避免再用 Conda 修改该环境，以防依赖链损坏。

### 依赖冲突解决策略
如果安装时报错（如 `Conflict detected` 或 `ERROR: ResolutionImpossible`）：
1. **指定版本范围**：例如针对 NumPy 2.0 冲突，强制使用旧版：`pip install "numpy<2"`。
2. **清理缓存并重试**：`pip cache purge` 或 `conda clean --all`。
3. **分步安装**：先安装核心大包（如 PyTorch），再安装周边依赖，观察在哪一步报错。
4. **使用 `conda-forge` 通道**：该通道通常比 `defaults` 通道包更全、版本更新，能解决很多搜索不到包的问题。
5. **创建纯净环境**：如果依赖已经完全搞乱，直接 `conda remove -n <env> --all` 重开。

## 6. 进阶环境维护
- **导出 Pip 依赖**: `pip freeze > requirements.txt`
- **导出 Conda 环境**: `conda env export > environment.yml`
- **克隆环境**: `conda create -n <new_env> --clone <old_env>`
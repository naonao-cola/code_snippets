# Chandra OCR 2 环境配置指南

本文档记录了当前开发环境的所有细节，以便在其他机器上快速复现。

## 1. 基础环境
- **操作系统**: Linux (Ubuntu/CentOS)
- **Python 版本**: 3.10.x
- **Conda 环境路径**: `/home/greatek/software/miniconda3/envs/chandra`
- **CUDA 版本**: 12.1+ (推荐使用 12.1 以匹配 vLLM 编译版)

## 2. 环境恢复步骤

### 方式 A: 使用 Conda 一键恢复 (推荐)
```bash
# 从配置文件直接创建环境
conda env create -f environment.yml -n chandra_new
conda activate chandra_new
```

### 方式 B: 手动分布搭建
1. **创建 Python 3.10 环境**:
   ```bash
   conda create -n chandra python=3.10 -y
   conda activate chandra
   ```

2. **安装 PyTorch (CUDA 12.1)**:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

3. **安装项目核心依赖**:
   ```bash
   pip install chandra-ocr vllm streamlit
   ```

4. **处理包冲突 (关键)**:
   - 确保 `pydantic >= 2.12.0`
   - 确保 `gradio >= 6.0` (以兼容 Pydantic 和 Pillow)

## 3. 模型配置
- **模型名称**: `datalab-to/chandra-ocr-2`
- **本地存储路径**: `./models/chandra-ocr-2`
- **配置文件**: 创建 `local.env` 并写入:
  ```text
  MODEL_CHECKPOINT=/你的路径/models/chandra-ocr-2
  ```

## 4. 运行命令总结

### 启动 vLLM 服务端 (宿主机)
```bash
vllm serve ./models/chandra-ocr-2 \
    --host 0.0.0.0 --port 8000 \
    --served-model-name chandra \
    --max-model-len 18000 --dtype bfloat16 \
    --gpu-memory-utilization 0.9 --max-num-seqs 4 \
    --enable-chunked-prefill --enable-prefix-caching \
    --trust-remote-code --no-enforce-eager \
    --mm-processor-kwargs '{"min_pixels": 3136, "max_pixels": 6291456}'
```

### 启动网页应用 (Streamlit)
```bash
export STREAMLIT_CONFIG_DIR=$(pwd)/.streamlit
streamlit run test02.py --browser.gatherUsageStats false
```

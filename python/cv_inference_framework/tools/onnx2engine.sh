#!/bin/bash

# 设置ONNX文件目录和Engine文件目录
onnx_dir="model_zoo/model_files/onnx"
engine_dir="model_zoo/model_files/engine"


# 遍历 onnx 目录中的所有 onnx 文件
for onnx_file in "$onnx_dir"/*.onnx; do
    # 提取文件名（不带路径）
    base_name=$(basename "$onnx_file" .onnx)
    
    # 设置对应的 engine 文件路径
    engine_file="$engine_dir/$base_name.engine"
    
    # 检查 engine 文件是否已经存在，若存在则跳过
    if [ -f "$engine_file" ]; then
        echo "Engine file $engine_file already exists, skipping conversion."
        continue
    fi
    
    # 执行 trtexec 命令转换 ONNX 到 TensorRT 引擎
    echo "Converting $onnx_file to $engine_file"
    trtexec --onnx="$onnx_file" --saveEngine="$engine_file"
    
    # 检查是否转换成功
    if [ $? -eq 0 ]; then
        echo "Successfully converted $onnx_file to $engine_file"
    else
        echo "Failed to convert $onnx_file"
    fi
done

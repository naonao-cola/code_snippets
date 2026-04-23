下载模型

mkdir -p models && export HF_ENDPOINT=https://hf-mirror.com && hf download datalab-to/chandra-ocr-2 --local-dir ./models/chandra-ocr-2

启动vllm
vllm serve datalab-to/chandra-ocr-2 \
    --served-model-name chandra \
    --max-model-len 18000 \
    --dtype bfloat16 \
    --enable-prefix-caching

运行脚本

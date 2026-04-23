"""
vLLM 终端启动命令 (请在运行本脚本前在另一个终端执行):

/home/greatek/software/miniconda3/envs/chandra/bin/vllm serve /home/greatek/wangww/demo/py/chandra-master/models/chandra-ocr-2 \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name chandra \
    --max-model-len 18000 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 4 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --trust-remote-code \
    --no-enforce-eager \
    --mm-processor-kwargs '{"min_pixels": 3136, "max_pixels": 6291456}'

运行本网页应用命令 (针对只读家目录环境优化):
export STREAMLIT_CONFIG_DIR=$(pwd)/.streamlit
streamlit run test02.py --browser.gatherUsageStats false
"""

import streamlit as st
import requests
from PIL import Image
from chandra.model import InferenceManager
from chandra.model.schema import BatchInputItem
from chandra.settings import settings
import io

st.set_page_config(page_title="Chandra OCR 2 演示", layout="wide")

st.title("📄 Chandra OCR 2 智能识别")
st.markdown("上传图片，通过 vLLM 后端进行 OCR 识别。")

# 检查 vLLM 状态
def check_vllm_health():
    api_base = settings.VLLM_API_BASE.replace("/v1", "")
    try:
        response = requests.get(f"{api_base}/v1/models", timeout=2)
        return response.status_code == 200
    except:
        return False

vllm_ready = check_vllm_health()

if not vllm_ready:
    st.error("⚠️ vLLM 服务未启动！请先在终端运行注释中的启动命令。")
else:
    st.success("✅ vLLM 服务已连接")

    # 图片上传
    uploaded_file = st.file_uploader("选择一张图片...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="已上传图片", use_container_width=True)

        if st.button("开始识别"):
            with st.spinner("正在识别中，请稍候..."):
                try:
                    manager = InferenceManager(method="vllm")
                    batch = [
                        BatchInputItem(
                            image=image,
                            prompt_type="ocr_layout"
                        )
                    ]
                    result = manager.generate(batch)[0]

                    with col2:
                        st.subheader("识别结果 (Markdown)")
                        st.markdown(result.markdown)

                        st.divider()

                        st.subheader("原始输出")
                        st.text_area("JSON/Text 内容", result.markdown, height=400)
                except Exception as e:
                    st.error(f"识别出错: {str(e)}")

st.sidebar.title("关于")
st.sidebar.info(
    "这是一个使用 Chandra-OCR 2 和 vLLM 构建的演示应用。\n\n"
    "模型路径: " + settings.MODEL_CHECKPOINT
)

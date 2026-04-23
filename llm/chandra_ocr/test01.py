import time
import requests
import subprocess
import os
import signal
import sys
from chandra.model import InferenceManager
from chandra.model.schema import BatchInputItem
from PIL import Image
from chandra.settings import settings



"""
参考链接 https://blog.csdn.net/weixin_35886636/article/details/159874400
"""
def wait_for_vllm(url, process, timeout=300):
    print(f"Waiting for vLLM server at {url}...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        # Check if process is still running
        if process and process.poll() is not None:
            print("vLLM process exited unexpectedly.")
            return False
        try:
            # vLLM models endpoint - Use /v1/models which is more standard in recent vLLM versions
            response = requests.get(f"{url}/v1/models", timeout=2)
            if response.status_code == 200:
                print("vLLM server is ready!")
                return True
        except Exception:
            # Fallback to /health for some versions
            try:
                response = requests.get(f"{url}/health", timeout=1)
                if response.status_code == 200:
                    print("vLLM server is ready (via /health)!")
                    return True
            except:
                pass
        time.sleep(5)
    print("Timeout waiting for vLLM server.")
    return False

def start_vllm_on_host():
    vllm_bin = "/home/greatek/software/miniconda3/envs/chandra/bin/vllm"
    model_path = settings.MODEL_CHECKPOINT

    print(f"Starting vLLM server on host with model: {model_path}")

    # Construct the command - Using positional argument for model path as recommended by vLLM
    cmd = [
        vllm_bin, "serve", model_path,
        "--host", "0.0.0.0",
        "--port", "8000",
        "--served-model-name", settings.VLLM_MODEL_NAME,
        "--max-model-len", "18000",
        "--dtype", "bfloat16",
        "--gpu-memory-utilization", "0.9",
        "--max-num-seqs", "4",
        "--enable-chunked-prefill",
        "--enable-prefix-caching",
        "--trust-remote-code",
        "--no-enforce-eager",
        "--mm-processor-kwargs", '{"min_pixels": 3136, "max_pixels": 6291456}'
    ]

    # Start the process
    process = subprocess.Popen(
        cmd,
        preexec_fn=os.setsid # Create a process group to kill it properly later
    )

    return process

def main():
    vllm_process = None
    try:
        # Check if server is already running
        api_base = settings.VLLM_API_BASE.replace("/v1", "")
        try:
            requests.get(f"{api_base}/models")
            print("vLLM server is already running.")
        except:
            vllm_process = start_vllm_on_host()

        if wait_for_vllm(api_base, vllm_process):
            manager = InferenceManager(method="vllm")
            batch = [
                BatchInputItem(
                    image=Image.open("/home/greatek/wangww/demo/py/chandra-master/images/IMG_20260414_150426.jpg"),
                    prompt_type="ocr_layout"
                )
            ]
            print("Starting inference...")
            result = manager.generate(batch)[0]
            print("\n--- OCR Result ---")
            print(result.markdown)
        else:
            print("Failed to start or connect to vLLM server.")

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        if vllm_process:
            print("Shutting down vLLM server...")
            os.killpg(os.getpgid(vllm_process.pid), signal.SIGTERM)
            vllm_process.wait()
            print("vLLM server stopped.")

if __name__ == "__main__":
    main()

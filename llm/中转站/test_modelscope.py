import os
import base64
import io
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("MODELSCOPE_API_KEY"),
    base_url=os.getenv("MODELSCOPE_BASE_URL")
)

def encode_image(image_path, max_size=(2048, 2048)):
    """
    将本地图片编码为 Base64 字符串。
    如果图片尺寸超过 max_size，则进行等比例缩放。
    """
    with Image.open(image_path) as img:
        # 如果图片尺寸超过限制，进行缩放
        if img.width > max_size[0] or img.height > max_size[1]:
            print(f"检测到图片尺寸 ({img.width}, {img.height}) 超过限制，正在缩放...")
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            print(f"缩放后尺寸: ({img.width}, {img.height})")

        # 将图片保存到内存中
        img_format = img.format if img.format else 'PNG'
        buffered = io.BytesIO()
        img.save(buffered, format=img_format)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

def test_multimodal(image_path=None):
    """
    测试 ModelScope 的多模态模型 (Qwen-VL)
    支持本地图片或远程 URL
    """
    try:
        print("\n--- 正在测试多模态模型 (Qwen/Qwen3-VL-8B-Instruct) ---")

        if image_path and os.path.exists(image_path):
            # 使用本地图片
            print(f"正在读取本地图片: {image_path}")
            base64_image = encode_image(image_path)
            image_url = f"data:image/png;base64,{base64_image}"
        else:
            # 使用默认远程图片
            print("未指定本地图片或路径不存在，使用默认远程图片...")
            image_url = "https://img.alicdn.com/tfs/TB1p.BGSpXXXXX9XXXXXXXXXXXX-204-52.png"

        response = client.chat.completions.create(
            model="Qwen/Qwen3-VL-8B-Instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "这张图片里有什么？"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                            },
                        },
                    ],
                }
            ],
            max_tokens=300,
        )
        print("回答内容:")
        print(response.choices[0].message.content)
    except Exception as e:
        print(f"多模态测试失败: {e}")

def test_text_stream():
    """
    测试文本流式输出 (Qwen3.5-35B)
    """
    try:
        print("\n--- 正在测试文本模型 (Qwen/Qwen3.5-35B-A3B) ---")
        response = client.chat.completions.create(
            model="Qwen/Qwen3.5-35B-A3B",
            messages=[
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': '请用一句话介绍一下魔塔社区。'}
            ],
            stream=True
        )
        print("回答内容: ", end="")
        for chunk in response:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end='', flush=True)
        print()
    except Exception as e:
        print(f"文本流式测试失败: {e}")

if __name__ == "__main__":
    # 1. 测试普通文本流式
    test_text_stream()

    # 2. 测试多模态 (可以传入本地图片路径，例如: "my_image.png")
    # 如果文件不存在，会自动切换回远程 URL 测试
    local_image = r"F:\360MoveData\Users\13191\Desktop\IMG_20260414_150451.jpg"
    test_multimodal(local_image)

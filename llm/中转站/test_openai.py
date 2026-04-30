import os
from openai import OpenAI
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

# 请在 .env 文件中填写你的 API Key 和 Base URL
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

client = OpenAI(
    api_key=api_key,
    base_url=base_url
)

def test_connection():
    try:
        print(f"正在测试连接到: {base_url}")
        # 使用一个简单的模型列表请求来测试连接
        response = client.models.list()
        print("连接成功！")
        print("可用模型示例:", [model.id for model in response.data[:5]])
    except Exception as e:
        print(f"连接失败: {e}")

def test_chat():
    try:
        print(f"\n正在测试对话接口 (模型: glm-5)...")
        response = client.chat.completions.create(
            model="glm-5",
            messages=[
                {"role": "user", "content": "请简短地介绍一下你自己，并告诉我你擅长做什么。"}
            ]
        )
        print("回答内容:")
        print("-" * 20)
        print(response.choices[0].message.content)
        print("-" * 20)
    except Exception as e:
        print(f"对话测试失败: {e}")

if __name__ == "__main__":
    if not api_key or api_key == "YOUR_API_KEY_HERE":
        print("提示: 请在 .env 文件中设置 OPENAI_API_KEY")

    test_connection()
    test_chat()

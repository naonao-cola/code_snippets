# sdk
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:6006/v1",
    api_key="sk-xxx",  # 随便填写，只是为了通过接口参数校验
)

completion = client.chat.completions.create(
    model="DeepSeek-R1-Distill-Qwen-7B",
    messages=[
        {"role": "user", "content": "什么是深度学习？<think>\n"}
    ]
)

print(completion.choices[0].message)
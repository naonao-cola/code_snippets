from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:6006/v1',
    api_key='xxxxx'
)

resp = client.chat.completions.create(
    model='qwen3-8b',
    messages=[
        # {"role": 'user', "content": "qwen3的大模型有什么特点？<think>\n"}
        {"role": 'user', "content": "什么是深度学习？"}
    ],
    extra_body={
        "chat_template_kwargs": {'enable_thinking': False}
    }
)

print(resp.choices[0].message)
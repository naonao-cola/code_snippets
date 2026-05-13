from openai import OpenAI


client = OpenAI(
    base_url='http://localhost:6006/v1',
    api_key='xxx'
)

resp = client.chat.completions.create(
    model='qwen3-8b',
    messages=[
        {'role': 'user', 'content':  '什么是AI大模型？'}
    ],
    # 只有qwen3才行
    extra_body={
        "chat_template_kwargs": {'enable_thinking': False}
    }
)

print(resp.choices[0].message)
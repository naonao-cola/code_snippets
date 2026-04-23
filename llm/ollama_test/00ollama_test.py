import ollama

client = ollama.Client(host='http://192.168.21.14:11434')  # 替换为你的服务器 IP 和端口

response = client.chat(
    model='qwen2.5:7b',
    messages=[{'role': 'user', 'content': '今天成都天气怎么样'}]
)

print(response['message']['content'])
# https://platform.openai.com/docs/guides/function-calling?api-mode=responses

import ollama
from ollama import Tool
import json

client = ollama.Client(host='http://192.168.21.14:11434')  # 替换为你的服务器 IP 和端口

tool_json_string = """{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Retrieves current weather for the given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City and country e.g. Bogotá, Colombia"
                }
            }
        }
    }
}"""



def get_weather(city_name):
    # 调用api获取天气数据的逻辑
    weather_data = {
        "city": city_name,
        "temperature": "20°C",
        "condition": "Clear Sky"
    }
    return weather_data


def process_query(question):
    tool_dict = json.loads(tool_json_string)
    tools = [tool_dict]

    messages = [{"role": "user", "content": question}]

    response = client.chat(
        model='qwen2.5:7b',
        messages=[{'role': 'user', 'content': question}],
        tools=tools
    
    )

    print(response['message']['content'])
    content = response["message"]

    if dict(content)["tool_calls"]:
        tool_call = dict(dict(content["tool_calls"][0])["function"])
        tool_name = tool_call["name"]
        tool_args = tool_call["arguments"]

        result = eval(tool_name)(tool_args["city"])

        # 将模型返回的调用哪个工具数据和工具执行完成后的数据都存入messages中
        messages.append({
            "role": "tool",
            "content": str(result),
            "tool_call_id": tool_call["name"],
            })

    response = client.chat(
        model='qwen2.5:7b',
        messages=[{'role': 'user', 'content': str(messages)}],
    )
    return response["message"]["content"]

if __name__ == "__main__":
    res = process_query(question = "今天成都天气怎么样")
    res = process_query(question = "今天天气怎么样？")
    # res = process_query(question = "今天是周几")

    print(res)

    
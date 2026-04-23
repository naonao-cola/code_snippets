import ollama
import json

def is_valid_json(s):
    try:
        json.loads(s)
        return True
    except (ValueError, TypeError):
        return False
    
def has_key(json_str, key):
    try:
        data = json.loads(json_str)
        return key in data
    except (ValueError, TypeError):
        return False
    
# tool function
def get_weather(city_name):
    # 调用api获取天气数据的逻辑
    weather_data = {
        "city": city_name,
        "temperature": "20°C",
        "condition": "Clear Sky"
    }
    return weather_data


prompt = """
    用户提问如下："{question}"

    根据问题内容，如果用户询问天气，请调用 get_weather 函数并返回 JSON 格式的天气数据。返回的 JSON 必须包含以下字段：
    - "function": get_weather
    - "city": 用户所描述城市名

    请确保返回的数据是有效的 JSON 格式，不要回答json以外的东西。
    如果问题不是关于天气的，则正常回答。
    如果用户没有提供城市名，则不返回json，返回正常文本回答。
    """

if __name__ == "__main__":
    client = ollama.Client(host='http://192.168.21.14:11434')  # 替换为你的服务器 IP 和端口
    # question = "今天成都天气怎么样？"
    question = "今天天气怎么样？"
    # question = "今天是周几"

    response = client.chat(
        model='qwen2.5:7b',  
        messages=[{'role': 'user', 'content': prompt.format(question=question)}]
    )

    message_content = response['message']['content']
    print(f"1、  ", message_content)

    if is_valid_json(message_content):
        if has_key(message_content, 'function'):
            function_name = json.loads(message_content)['function']
            if function_name == 'get_weather':
                city_name = json.loads(message_content)['city']
                weather_data = get_weather(city_name)
                print(f"2、  ",weather_data)
                
                response = client.chat(
                    model='qwen2.5:7b',  
                    messages=[{'role': 'user', 'content': question + "今天天气情况" + str(weather_data)}]
                )

                message_content = response['message']['content']
                print(f"3、  ",message_content)

    else:
        response = client.chat(
            model='qwen2.5:7b',  
            messages=[{'role': 'user', 'content': question}]
        )

        message_content = response['message']['content']
        print(f"4、  ",message_content)
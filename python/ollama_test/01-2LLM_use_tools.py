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
    

def get_weather(city_name):
    # 调用api获取天气数据的逻辑
    weather_data = {
        "city": city_name,
        "temperature": "20°C",
        "condition": "Clear Sky"
    }
    return weather_data


prompt = '''
    {
      "user_question": "%s",
      "response_instructions": {
        "task": "根据问题内容，如果用户询问天气并且包含城市名，请返回一个包含以下内容的 JSON：",
        "example_if_get_weather": {
          "function": "get_weather",
          "city": "城市名"
        },
        "example_if_no_get_weather": {
          "function": "None"
        },
        "condition_for_get_weather": "如果user_question中没有城市名，则返回city为None",结果不返回json以外的文字。
      }
    }
'''

if __name__ == "__main__":
    client = ollama.Client(host='http://192.168.21.14:11434')  # 替换为你的服务器 IP 和端口
    question = "今天成都天气怎么样？"
    question = "今天天气怎么样？"
    # question = "今天的温度怎么样？"
    # question = "今天成都的温度怎么样？"
    # question = "今天是周几"


    response = client.chat(
        model='qwen2.5:7b',  
        messages=[{'role': 'user', 'content': prompt% question}]
    )

    message_content = response['message']['content']
    message_content = message_content.replace("```json", "").replace("```", "").replace("\n", "")
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

            elif function_name:= 'None':
                
                response = client.chat(
                    model='qwen2.5:7b',  
                    messages=[{'role': 'user', 'content': question}]
                )

                message_content = response['message']['content']
                print(f"4、  ",message_content)
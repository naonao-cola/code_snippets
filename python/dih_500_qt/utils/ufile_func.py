#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import sys
from utils.ulog import ulog
def is_image_file(path):
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']
    return any(path.lower().endswith(ext) for ext in valid_extensions)

def load_json(image_path):
        current_json_path = os.path.splitext(image_path)[0] + ".json"
        if not os.path.exists(current_json_path):
            return
        with open(current_json_path, 'r', encoding='utf-8') as file:
            json_content = file.read()
        return json_content







def get_config_path(dir_name, file_name):
    # 获取当前脚本所在的上层目录
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # 拼接配置文件路径
    config_path = os.path.join(current_dir, 'config', dir_name, file_name)
    dst_path = f"{config_path}.json"
    return dst_path


def find_in_nested(data, target_id, id_key="type_id"):
    """递归查找嵌套结构中匹配id_key的项"""
    if isinstance(data, dict):
        if data.get(id_key) == target_id:
            return data
        for value in data.values():
            result = find_in_nested(value, target_id, id_key)
            if result is not None:
                return result
    elif isinstance(data, list):
        for item in data:
            result = find_in_nested(item, target_id, id_key)
            if result is not None:
                return result
    return None

# if __name__ == "__main__":
#     # 测试代码
#     print(get_config_path("algo", "alg_cfg"))

#!/usr/bin/env python3
"""
Windows 系统信息收集脚本
"""
import platform
import sys
import subprocess
import json
import os
from datetime import datetime

def get_system_info():
    """收集并返回系统信息字典"""
    print("hello---------正在收集系统信息...")
    info = {}

    # 1. 操作系统信息
    info['操作系统'] = {
        '系统': platform.system(),
        '版本': platform.version(),
        '发行版': platform.release(),
        '架构': platform.architecture()[0],
        '机器': platform.machine(),
        '处理器': platform.processor(),
    }

    # 2. Python环境信息
    info['Python环境'] = {
        '版本': sys.version,
        '可执行文件': sys.executable,
        '路径': sys.path
    }

    # 3. 磁盘信息 (Windows特定命令)
    try:
        # 使用 `wmic` 命令获取磁盘信息 (Windows)
        result = subprocess.run(
            ['wmic', 'logicaldisk', 'get', 'size,freespace,caption', '/format:csv'],
            capture_output=True,
            text=True,
            encoding='gbk'  # Windows中文环境常用编码
        )
        drives = []
        for line in result.stdout.strip().split('\n')[1:]:  # 跳过标题行
            if line:
                parts = line.split(',')
                if len(parts) >= 4:
                    drive = parts[-1].strip()
                    try:
                        free = int(parts[-2]) if parts[-2] else 0
                        total = int(parts[-3]) if parts[-3] else 0
                        used = total - free
                        used_percent = (used / total * 100) if total > 0 else 0
                        drives.append({
                            '驱动器': drive,
                            '总空间(GB)': round(total / (1024**3), 2),
                            '可用空间(GB)': round(free / (1024**3), 2),
                            '已用百分比(%)': round(used_percent, 2)
                        })
                    except (ValueError, ZeroDivisionError):
                        continue
        info['磁盘信息'] = drives
    except Exception as e:
        info['磁盘信息'] = f"获取失败: {e}"

    # 4. 网络信息 (简化的IP配置)
    try:
        result = subprocess.run(['ipconfig'], capture_output=True, text=True, encoding='gbk')
        # 简化输出，只取IPv4地址行
        ip_lines = [line.strip() for line in result.stdout.split('\n') if 'IPv4' in line]
        info['网络信息'] = ip_lines if ip_lines else ["未能获取IP信息"]
    except Exception as e:
        info['网络信息'] = [f"获取失败: {e}"]

    # 5. 当前工作目录和用户
    info['当前状态'] = {
        '工作目录': os.getcwd(),
        '用户名': os.getenv('USERNAME') or os.getenv('USER'),
        '时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    return info

if __name__ == '__main__':
    print("=" * 50)
    print("            Windows 系统信息报告")
    print("=" * 50)
    sys_info = get_system_info()

    for category, data in sys_info.items():
        print(f"\n【{category}】")
        if isinstance(data, dict):
            for key, value in data.items():
                print(f"  {key}: {value}")
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    for k, v in item.items():
                        print(f"    {k}: {v}")
                    print()  # 每个磁盘信息后空一行
                else:
                    print(f"  - {item}")
        else:
            print(f"  {data}")
    print("\n" + "=" * 50)
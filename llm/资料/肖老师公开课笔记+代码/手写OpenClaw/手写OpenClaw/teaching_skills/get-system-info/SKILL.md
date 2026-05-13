---
name: get-system-info
description: 当用户提问关于当前计算机的系统信息，例如操作系统详情、Python环境、磁盘空间、网络配置等时，使用此技能。本技能利用一个Python脚本获取结构化信息。
---


# Windows 系统信息查询技能

## 概述
本技能用于获取并报告当前Windows系统的详细信息。为了提供准确、结构化的信息，请运行同目录下的 `get_system_info.py` 脚本。

## 重要提示
- **必须在项目根目录下运行脚本**：脚本使用相对路径，必须在项目根目录下执行
- **使用相对路径**：使用 `python teaching_skills/windows-system-info/get_system_info.py` 而不是绝对路径
- **先验证当前目录**：执行前先用 `cd` 命令查看当前目录

## 操作步骤
1. **检查当前工作目录**：
   - 使用 `cd` 命令（不带参数）查看当前目录
   
2. **验证脚本存在**：
   - 使用 `dir teaching_skills\windows-system-info\get_system_info.py` 验证脚本存在
   
3. **执行查询命令**：
   - 运行系统信息脚本：`python teaching_skills/windows-system-info/get_system_info.py`

## 快捷命令
如果确定当前是项目根目录，可以直接使用：
- `python teaching_skills/windows-system-info/get_system_info.py`

## 错误处理
如果遇到"系统找不到指定的路径"错误：
1. 先用 `cd` 查看当前目录
2. 用 `dir` 查看 `teaching_skills` 目录是否存在
3. 确保在正确的项目根目录下执行
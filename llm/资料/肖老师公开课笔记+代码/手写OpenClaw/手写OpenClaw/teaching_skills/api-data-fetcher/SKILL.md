---
name: api-data-fetcher
description: 当用户需要获取网络数据、查询API信息、获取公开数据（如天气、股票、新闻、汇率等）时，使用此技能。本技能通过调用公开API获取数据，可以接收多种参数进行灵活查询。
---

# API数据查询技能

## 概述
本技能提供了访问多种公开API的功能，可以通过参数指定要查询的数据类型和参数。

## 重要提示
- **必须在项目根目录下运行脚本**：脚本使用相对路径，必须在项目根目录（E:\my_project\project26）下执行
- **使用相对路径**：使用 `python teaching_skills/api-data-fetcher/fetch_api_data.py` 而不是绝对路径
- **先验证当前目录**：执行前先用 `cd` 命令查看当前目录

## 操作步骤
1. **检查当前工作目录**：
   - 使用 `cd` 命令（不带参数）查看当前目录
   - 如果显示的不是项目根目录（E:\my_project\project26），先切换到项目根目录
   
2. **验证脚本存在**：
   - 使用 `dir teaching_skills\api-data-fetcher\fetch_api_data.py` 验证脚本存在
   
3. **执行查询命令**：
   - 查询天气：`python teaching_skills/api-data-fetcher/fetch_api_data.py --api-type weather --city "北京"`
   - 查询汇率：`python teaching_skills/api-data-fetcher/fetch_api_data.py --api-type exchange --base-currency USD --target-currency CNY`
   - 获取新闻：`python teaching_skills/api-data-fetcher/fetch_api_data.py --api-type news --category technology`
   - 获取名言：`python teaching_skills/api-data-fetcher/fetch_api_data.py --api-type quote`
   - 查询IP：`python teaching_skills/api-data-fetcher/fetch_api_data.py --api-type ipinfo --ip-address "8.8.8.8"`

## 快捷命令
如果确定当前是项目根目录，可以直接使用：
- `python teaching_skills/api-data-fetcher/fetch_api_data.py -t weather -c 北京`

## 错误处理
如果遇到"系统找不到指定的路径"错误：
1. 先用 `cd` 查看当前目录
2. 用 `dir` 查看 `teaching_skills` 目录是否存在
3. 确保在正确的项目根目录下执行
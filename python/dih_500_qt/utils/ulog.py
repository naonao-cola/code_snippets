#!/usr/bin/python
# -*- coding: UTF-8 -*-

from loguru import logger
import os

class ulog:
    def __init__(self):
        # 确保日志目录存在
        log_dir = "log"
        os.makedirs(log_dir, exist_ok=True)

        # 配置日志输出到文件，按天轮换，保留最近一周的日志
        logger.add(
            f"{log_dir}/app_{{time:YYYY-MM-DD}}.log",
            rotation="1 day",
            retention="1 week",
            encoding="utf-8",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {file} | {line} | {message}"
        )

    def get_logger(self):
        return logger

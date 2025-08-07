#!/bin/bash

SERVICE_NAME="frpc"      # 要监控的服务名称
LOG_FILE="/home/y/proj/www/log/service_monitor.log"  # 日志文件路径

# 检查服务是否活动（running）
if ! systemctl is-active --quiet "$SERVICE_NAME"; then
  # 记录日志并尝试启动服务
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] 服务未运行，正在启动..." >> "$LOG_FILE"
  sudo systemctl start "$SERVICE_NAME" >> "$LOG_FILE" 2>&1

  # 检查启动是否成功
  if systemctl is-active --quiet "$SERVICE_NAME"; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 服务启动成功" >> "$LOG_FILE"
  else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 服务启动失败！" >> "$LOG_FILE"
  fi
fi

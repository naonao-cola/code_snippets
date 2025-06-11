#!/bin/bash

# 定义记录文件路径
output_file="resource_usage.log"

# 获取 CPU 使用率的函数
get_cpu_usage() {
    # 获取 CPU 的总空闲时间和总时间
    prev_idle=$(grep '^cpu ' /proc/stat | awk '{print $5}')
    prev_total=$(grep '^cpu ' /proc/stat | awk '{print $2+$3+$4+$5+$6+$7+$8+$9+$10+$11+$12}')

    # 等待 100ms 以获取时间差
    sleep 0.1

    # 获取 CPU 的当前空闲时间和总时间
    curr_idle=$(grep '^cpu ' /proc/stat | awk '{print $5}')
    curr_total=$(grep '^cpu ' /proc/stat | awk '{print $2+$3+$4+$5+$6+$7+$8+$9+$10+$11+$12}')

    # 计算 CPU 使用率
    idle=$((curr_idle - prev_idle))
    total=$((curr_total - prev_total))
    cpu_usage_percent=$((100 * (total - idle) / total))

    # 返回 CPU 使用率百分比
    echo "$cpu_usage_percent"
}

# 获取内存使用情况的函数
get_memory_usage() {
    # 获取内存总量和已使用内存（以 MB 为单位）
    memory_total=$(free -m | awk '/^Mem:/ {print $2}')
    memory_used=$(free -m | awk '/^Mem:/ {print $3}')

    # 计算内存使用率（%）
    memory_usage_percent=$(echo "scale=2; $memory_used * 100 / $memory_total" | bc)

    # 返回内存使用情况
    echo "$memory_total $memory_used $memory_usage_percent"
}

# 获取 CPU 和内存占用率最高的进程的函数
get_top_process() {
    # 获取 CPU 占用率最高的进程
    cpu_top_info=$(ps -eo pid,%cpu,%mem,comm --sort=-%cpu | head -n 2 | tail -n 1)
    cpu_top_pid=$(echo "$cpu_top_info" | awk '{print $1}')
    cpu_top_cpu=$(echo "$cpu_top_info" | awk '{print $2}')
    cpu_top_mem=$(echo "$cpu_top_info" | awk '{print $3}')
    cpu_top_name=$(echo "$cpu_top_info" | awk '{print $4}')

    # 获取内存占用率最高的进程
    memory_top_info=$(ps -eo pid,%cpu,%mem,comm --sort=-%mem | head -n 2 | tail -n 1)
    memory_top_pid=$(echo "$memory_top_info" | awk '{print $1}')
    memory_top_cpu=$(echo "$memory_top_info" | awk '{print $2}')
    memory_top_mem=$(echo "$memory_top_info" | awk '{print $3}')
    memory_top_name=$(echo "$memory_top_info" | awk '{print $4}')

    # 返回进程信息
    echo "$cpu_top_pid $cpu_top_cpu $cpu_top_mem $cpu_top_name $memory_top_pid $memory_top_cpu $memory_top_mem $memory_top_name"
}

# 主循环每秒记录一次
while true; do
    # 获取 CPU 和内存使用情况
    cpu_usage=$(get_cpu_usage)
    read -r memory_total memory_used memory_usage_percent <<< $(get_memory_usage)
    read -r cpu_top_pid cpu_top_cpu cpu_top_mem cpu_top_name memory_top_pid memory_top_cpu memory_top_mem memory_top_name <<< $(get_top_process)

    # 获取当前时间
    timestamp=$(date +"%Y-%m-%d %H:%M:%S")

    # 将信息写入日志文件
    echo "$timestamp - CPU Usage: $cpu_usage%, Memory Total: $memory_total MB, Memory Used: $memory_used MB, Memory Usage: $memory_usage_percent%" >> "$output_file"
    echo "  CPU Highest Usage Process: PID: $cpu_top_pid, CPU: $cpu_top_cpu%, MEM: $cpu_top_mem%, Name: $cpu_top_name" >> "$output_file"
    echo "  Memory Highest Usage Process: PID: $memory_top_pid, CPU: $memory_top_cpu%, MEM: $memory_top_mem%, Name: $memory_top_name" >> "$output_file"
    echo "" >> "$output_file"

    # 等待 990ms，加上之前的 100ms，总共 1 秒
    sleep 0.9
done
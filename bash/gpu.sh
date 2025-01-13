nvidia-smi

# 获取 nvidia-smi 输出的 PID 列表
pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)

# 遍历每个 PID
for pid in $pids; do
    # 获取该 PID 的 cwd 路径（使用 sudo 获取权限）
    cwd=$(sudo readlink /proc/$pid/cwd)

    # 打印 PID 和其 cwd 路径
    if [ -n "$cwd" ]; then
        echo "PID: $pid -> cwd: $cwd"
    else
        echo "PID: $pid -> cwd: Not Available"
    fi
done

echo " "
free -h



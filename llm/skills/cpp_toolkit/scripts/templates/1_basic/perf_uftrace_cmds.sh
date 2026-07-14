#!/bin/bash
# ==============================================================================
# 性能分析工具 (perf, FlameGraph, uftrace) 核心命令模板
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. perf 安装与基本使用 (WSL/Linux)
# ------------------------------------------------------------------------------
# 安装依赖
# sudo apt-get install linux-tools-common
# 若遇到 WSL 内核版本问题，需自行编译 WSL2-Linux-Kernel/tools/perf

# 生成采样记录 ( -F 99: 99次/秒, -a: 所有CPU, -g: 调用栈)
perf record -F 99 -a -g ./perf_001

# 指定进程号持续分析 30 秒
# perf record -F 99 -p 2347 -g -- sleep 30

# 查看分析报告
# perf report -i perf.data
perf report --call-graph none -c perf_001

# 统计特定事件 (如 cache-misses, cpu-clock)
perf stat -e cache-misses ./perf_001
perf stat -e cpu-clock ./perf_001

# 导出特定函数的汇编指令耗时
# sudo perf annotate -f main


# ------------------------------------------------------------------------------
# 2. 火焰图 (FlameGraph) 生成
# ------------------------------------------------------------------------------
# 下载工具
# git clone https://github.com/brendangregg/FlameGraph.git
# cd FlameGraph

# 解析 -> 折叠 -> SVG 火焰图
perf script -i perf.data > perf.unfold
./stackcollapse-perf.pl perf.unfold > perf.folded
./flamegraph.pl perf.folded > perf.svg

# 或者一条管道命令搞定
# perf script -i perf.data | stackcollapse-perf.pl | flamegraph.pl > perf.svg


# ------------------------------------------------------------------------------
# 3. uftrace 使用范式 (编译需添加 -pg)
# ------------------------------------------------------------------------------
# 控制台实时查看
uftrace ./perf_001

# 记录 5 次执行，然后再查看
uftrace record ./perf_001 5
uftrace replay

# 过滤控制
uftrace --no-libcall ./perf_001 5          # 隐藏 new/delete 等库函数
uftrace -k ./perf_001 5                    # 记录 kernel 调用
uftrace --depth 3 ./perf_001 5             # 指定深度为 3
uftrace -F allocate2 ./perf_001 5          # 仅追踪 allocate2 及其子函数
uftrace -t 1us ./perf_001 5                # 过滤耗时 < 1us 的函数

# 记录参数与返回值
uftrace -A atoi@arg1/s ./perf_001 3        # 记录 atoi 的字符串参数
uftrace -R atoi@retval ./perf_001 3        # 记录 atoi 的返回值
uftrace -a ./perf_001 3                    # 自动记录参数和返回值

# 性能统计报告
uftrace report -s self                     # 按自身耗时排序 (Total time / Self time)

# 查看 Call Graph (TUI)
# uftrace graph

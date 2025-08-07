#!/bin/sh
chmod 777 main2
export LD_LIBRARY_PATH=./data/alg/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=./data/alg/cl:$LD_LIBRARY_PATH

# 检查参数数量
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <int_value> <string_value>"
    exit 1
fi

# 获取参数
int_val=$1
str_val=$2

./main2 "$int_val" "$str_val"

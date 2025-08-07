/**
 * @FilePath     : /test02/src/test_2.cpp
 * @Description  :
 * @Author       : naonao
 * @Date         : 2025-07-23 17:10:11
 * @Version      : 0.0.1
 * @LastEditors  : naonao
 * @LastEditTime : 2025-07-23 17:38:35
 * @Copyright (c) 2025 by G, All Rights Reserved.
 **/
#include <jemalloc/jemalloc.h>
#include <stdlib.h>

void do_something(size_t i)
{
    je_malloc(i * 100); // 分配内存
}

int main(int argc, char** argv)
{
    for (size_t i = 0; i < 1000; i++) {
        do_something(i);
    }
    // 打印分配器统计信息
    malloc_stats_print(NULL, NULL, NULL);
    return 0;
}
/**
 * @FilePath     : /test_xsimd/src/main.cpp
 * @Description  :
 * @Author       : naonao
 * @Date         : 2024-09-20 22:44:20
 * @Version      : 0.0.1
 * @LastEditors  : naonao
 * @LastEditTime : 2024-09-20 23:36:59
 * @Copyright (c) 2024 by G, All Rights Reserved.
**/
#include "test.h"


// 假设数组长度是已知的
constexpr std::size_t array_size = 429496729; // 1 million elements

// 使用标准C++库计算数组总和
double sum_std(const double* data, std::size_t size) {
    return std::accumulate(data, data + size, 0.0);
}

// 使用 xsimd 计算数组总和
double sum_xsimd(const double* data, std::size_t size) {
    using batch_type = xsimd::batch<double>;
    constexpr std::size_t batch_size = batch_type::size;
    double sum = 0.0;

    // 处理整数倍于 batch_size 的部分
#pragma omp parallel for reduction(+:sum)
    for (long i = 0; i <= size - batch_size; i += batch_size) {
        auto batch = xsimd::load_unaligned(data + i);
        sum += xsimd::reduce_add(batch);
    }

    // 处理剩余的部分
#pragma omp parallel for reduction(+:sum)
    for (long i = (size / batch_size) * batch_size; i < size; ++i) {
        sum += data[i];
    }

    return sum;
}

int main() {
    // 初始化随机数生成器
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(-100.0, 100.0);

    // 创建两个 double 数组并填充随机数据
    std::vector<double> a(array_size);
    for (auto& value : a) {
        value = distribution(generator);
    }

    // 测试标准C++库的性能
    auto start = std::chrono::high_resolution_clock::now();
    double result_std = sum_std(a.data(), a.size());
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_std = end - start;
    std::cout << "Sum with std::accumulate: " << result_std << " in " << elapsed_std.count() << " seconds" << std::endl;

    // 测试 xsimd 的性能
    start = std::chrono::high_resolution_clock::now();
    double result_xsimd = sum_xsimd(a.data(), a.size());
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_xsimd = end - start;
    std::cout << "Sum with xsimd: " << result_xsimd << " in " << elapsed_xsimd.count() << " seconds" << std::endl;

    // 检查结果是否一致
    if (result_std == result_xsimd) {
        std::cout << "The results are the same." << std::endl;
    } else {
        std::cout << "The results differ!" << std::endl;
    }
    test_01();
    return 0;
}
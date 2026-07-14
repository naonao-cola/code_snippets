#include <iostream>
#include <vector>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include "xsimd/xsimd.hpp"

// ============================================================================
// 1. OpenMP 高级用法：自定义规约 (Custom Reduction) 与结构体规约
// ============================================================================
struct compare {
    float val;
    int index;
    int index1;
};

struct compare add_matrix(struct compare X, struct compare Y) {
    struct compare temp;
    if (X.val < Y.val) {
        temp.val = X.val;
        temp.index = X.index;
        temp.index1 = X.index1;
    } else {
        temp.val = Y.val;
        temp.index = Y.index;
        temp.index1 = Y.index1;
    }
    return temp;
}

// 声明基于结构体的自定义归约
#pragma omp declare reduction(p_add_matrix : struct compare : omp_out = add_matrix(omp_out, omp_in)) initializer(omp_priv = {100})

// 声明 std::vector 的自定义归约（拼接）
class Match { /* 伪代码 */ };
#pragma omp declare reduction(omp_insert: std::vector<Match>: omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))

// ============================================================================
// 2. 针对 VS Studio 不支持高级 OpenMP 特性的降级写法（利用 Thread-Local 变量）
// ============================================================================
void matchClass_MSVC_Fallback(std::vector<Match>& matches) {
#ifdef _OPENMP
#pragma omp parallel
    {
#endif
        std::vector<Match> match_private; // 每个线程私有的局部变量
#ifdef _OPENMP
#pragma omp for nowait
#endif
        for (int i = 0; i < 100; ++i) {
            // ... 复杂的模板匹配计算 ...
            // match_private.push_back(candidate);
        }
        
#ifdef _OPENMP
#pragma omp critical // 线程计算完毕后，安全合并到全局结果
        {
#endif
            matches.insert(matches.end(), match_private.begin(), match_private.end());
#ifdef _OPENMP
        }
    }
#endif
}

// ============================================================================
// 3. OpenMP 多重归约与获取线程 ID (结合 OpenCV 实战)
// ============================================================================
cv::Mat duplicate_remove(cv::Mat data, double remain) {
    int length = data.rows;
    cv::Mat matrix = cv::Mat::zeros(length, length, CV_32FC1);
    double min_value = std::numeric_limits<double>::max();
    double max_value = 0.f;

    // 多重归约：同时找最小值和最大值
#pragma omp parallel for reduction(min:min_value) reduction(max:max_value)
    for (int i = 0; i < data.rows; i++) {
        cv::Mat p1 = data.rowRange(i, i + 1);
        for (int j = 0; j < data.rows; j++) {
            cv::Mat p2 = data.rowRange(j, j + 1);
            double diff = cv::norm((p1 - p2)) / (p1.cols * 1.f);
            matrix.ptr<float>(i)[j] = diff;
            if (diff < min_value && diff > 0.0000001) min_value = diff;
            if (diff > max_value) max_value = diff;
        }
    }

    double remain_max = max_value - (max_value - min_value) * remain;
    cv::Mat remain_mat = matrix > remain_max;
    
    std::vector<std::pair<int, int>> d1;
    std::vector<std::vector<std::pair<int, int>>> local_d1(omp_get_max_threads()); // 利用最大线程数创建二维局部数组
    
#pragma omp parallel for
    for (int i = 0; i < remain_mat.rows; ++i) {
        uchar* p = remain_mat.ptr<uchar>(i);
        int thread_id = omp_get_thread_num(); // 获取当前线程 ID 以访问局部数组
        for (int j = 0; j < remain_mat.cols; ++j) {
            if (i == j) continue;
            if (p[j] < 127) {
                local_d1[thread_id].push_back(std::make_pair(i, j));
            }
        }
    }
    
    for (auto& local_vector : local_d1) {
        d1.insert(d1.end(), local_vector.begin(), local_vector.end());
    }
    // ... 后续去重逻辑 ...
    return cv::Mat();
}

// ============================================================================
// 4. SIMD (xsimd) 搭配 OpenMP 的完整处理模式 (含余数处理)
// ============================================================================
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

    // 处理剩余的尾部数据（不能凑齐一个 SIMD batch 的部分）
#pragma omp parallel for reduction(+:sum)
    for (long i = (size / batch_size) * batch_size; i < size; ++i) {
        sum += data[i];
    }

    return sum;
}
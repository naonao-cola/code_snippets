// 编译选项：g++ perf_uftrace_demo.cpp -o perf_001 -pg -O0
#include <chrono>
#include <iostream>
#include <vector>

class ComplexObject
{
public:
    ComplexObject(int value)
        : value_(value)
    {
        // std::cout << "ComplexObject(" << value_ << ") constructed\n";
    }
    ~ComplexObject()
    {
        // std::cout << "ComplexObject(" << value_ << ") destroyed\n";
    }

    int value_;
};

int main()
{
    std::vector<ComplexObject> vec;

    // 使用 push_back 添加元素
    auto start_push_back = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10000000; ++i) {
        vec.push_back(ComplexObject(i));
    }
    auto end_push_back  = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> push_back_time = end_push_back - start_push_back;
    std::cout << "Time taken by push_back: " << push_back_time.count() << " ms\n";

    // 清空 vector 以便重新测试 emplace_back
    vec.clear();

    // 使用 emplace_back 添加元素
    auto start_emplace_back = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10000000; ++i) {
        vec.emplace_back(i);
    }
    auto end_emplace_back  = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> emplace_back_time = end_emplace_back - start_emplace_back;
    std::cout << "Time taken by emplace_back: " << emplace_back_time.count() << " ms\n";

    return 0;
}

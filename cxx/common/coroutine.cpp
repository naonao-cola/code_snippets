#include <coroutine>
#include <iostream>
/**
 * @FilePath     : /code_snippets/cxx/common/temp_2.cpp
 * @Description  :
 * @Author       : naonao
 * @Date         : 2024-12-10 17:01:06
 * @Version      : 0.0.1
 * @LastEditors  : naonao
 * @LastEditTime : 2024-12-10 17:01:07
 * @Copyright (c) 2024 by G, All Rights Reserved.
**/
#include <optional>
#include <stdexcept>
#include <vector>

// 模板化的协程承诺类型
template<typename T>
struct GeneratorPromise {
    using handle_type = std::coroutine_handle<GeneratorPromise<T>>;

    // 当前生成的值
    std::optional<T> current_value;

    // 构造函数
    GeneratorPromise() = default;

    // 当协程创建时调用
    auto get_return_object() {
        return handle_type::from_promise(*this);
    }

    // 协程开始时的挂起
    std::suspend_never initial_suspend() {
        return {};
    }

    // 协程结束时的挂起
    std::suspend_never final_suspend() noexcept {
        return {};
    }

    // 当协程异常时调用
    void unhandled_exception() {
        // 存储异常以便后续处理
        current_value = std::nullopt;
    }

    // 当协程 yield 一个值时调用
    std::suspend_always yield_value(T value) {
        current_value = value;
        return {};
    }

    // 当协程完成时调用
    void return_void() {}
};

// 模板化的协程句柄类型
template<typename T>
using GeneratorHandle = std::coroutine_handle<GeneratorPromise<T>>;

// 模板化的生成器协程类
template<typename T>
class Generator {
public:
    // 使用 GeneratorHandle 来构造
    explicit Generator(GeneratorHandle<T> h) : handle(h) {}

    // 析构函数，销毁协程
    ~Generator() {
        if (handle) {
            handle.destroy();
        }
    }

    // 迭代器概念
    class iterator {
    public:
        using iterator_category = std::input_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = const T*;
        using reference = const T&;

        iterator(GeneratorHandle<T> h) : handle(h) {}

        reference operator*() const {
            if (!handle || !handle.promise().current_value) {
                throw std::runtime_error("Attempted to dereference an invalid iterator");
            }
            return handle.promise().current_value.value();
        }

        iterator& operator++() {
            if (!handle) {
                return *this;
            }

            handle.resume();
            if (handle.done() || !handle.promise().current_value) {
                handle = nullptr;
            }
            return *this;
        }

        bool operator==(const iterator& other) const {
            return handle == other.handle;
        }

        bool operator!=(const iterator& other) const {
            return !(*this == other);
        }

    private:
        GeneratorHandle<T> handle;
    };

    // 开始迭代
    iterator begin() {
        if (handle) {
            handle.resume();
        }
        return iterator{handle};
    }

    // 结束迭代
    static iterator end() {
        return iterator{nullptr};
    }

private:
    GeneratorHandle<T> handle;
};

// 协程管理器类
class CoroutineManager {
public:
    // 添加协程
    template<typename T>
    void add_coroutine(Generator<T>&& gen) {
        coroutines.emplace_back(std::move(gen));
    }

    // 启动所有协程
    void start_all() {
        for (auto& coroutine : coroutines) {
            coroutine.begin();
        }
    }

    // 检查所有协程是否完成
    bool all_done() const {
        for (const auto& coroutine : coroutines) {
            if (!coroutine.handle.done()) {
                return false;
            }
        }
        return true;
    }

    // 清理所有协程
    void cleanup() {
        coroutines.clear();
    }

private:
    std::vector<Generator<int>> coroutines;
};

// 协程函数，生成不同类型的数值
template<typename T>
Generator<T> generate_numbers(T start, T end) {
    for (T i = start; i < end; ++i) {
        co_yield i;
    }
}

int main() {
    CoroutineManager manager;

    // 添加整数协程
    manager.add_coroutine(generate_numbers<int>(1, 10));

    // 添加浮点数协程
    manager.add_coroutine(generate_numbers<double>(1.0, 5.0));

    // 启动所有协程
    manager.start_all();

    // 迭代并打印整数协程的结果
    for
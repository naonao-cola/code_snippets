
#include <atomic>
/**
 * @FilePath     : /connector_algo/src/custom/spinlock.h
 * @Description  :
 * @Author       : naonao 1319144981@qq.com
 * @Version      : 0.0.1
 * @LastEditors  : naonao 1319144981@qq.com
 * @LastEditTime : 2024-04-19 11:37:47
 * @Copyright    : G AUTOMOBILE RESEARCH INSTITUTE CO.,LTD Copyright (c) 2024.
 **/

class USpinLock {
public:
    void lock()
    {
        // memory_order_acquire
        while (flag_.test_and_set(std::memory_order_acquire)) {
        }
    }

    void unlock()
    {
        // memory_order_release
        flag_.clear(std::memory_order_release);
    }

    bool tryLock()
    {
        return !flag_.test_and_set();
    }

private:
    std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
};
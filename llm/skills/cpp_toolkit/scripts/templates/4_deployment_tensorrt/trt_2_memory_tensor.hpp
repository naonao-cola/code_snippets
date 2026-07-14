/*
 * 混合内存封装 (MixMemory) 与 Tensor
 * 提取自 algo_ai/include/private/trt/trt_common/trt-tensor.hpp
 *
 * 核心思想：
 * 统一管理 CPU (Pinned Memory) 和 GPU (Device Memory)，并在获取指针时按需自动触发数据拷贝。
 */

#include <memory>
#include <vector>
#include <cuda_runtime_api.h>

namespace TRT {

    class MixMemory {
    public:
        MixMemory(int device_id = 0) : device_id_(device_id) {}
        ~MixMemory() { release_all(); }

        void* gpu(size_t size) {
            if (gpu_size_ < size) {
                release_gpu();
                cudaSetDevice(device_id_);
                cudaMalloc(&gpu_, size);
                gpu_size_ = size;
            }
            return gpu_;
        }

        void* cpu(size_t size) {
            if (cpu_size_ < size) {
                release_cpu();
                cudaSetDevice(device_id_);
                cudaMallocHost(&cpu_, size); // Pinned memory
                cpu_size_ = size;
            }
            return cpu_;
        }

        void release_gpu() {
            if (owner_gpu_ && gpu_) {
                cudaFree(gpu_);
                gpu_ = nullptr;
                gpu_size_ = 0;
            }
        }

        void release_cpu() {
            if (owner_cpu_ && cpu_) {
                cudaFreeHost(cpu_);
                cpu_ = nullptr;
                cpu_size_ = 0;
            }
        }

        void release_all() {
            release_cpu();
            release_gpu();
        }

    private:
        void* cpu_ = nullptr;
        size_t cpu_size_ = 0;
        bool owner_cpu_ = true;

        void* gpu_ = nullptr;
        size_t gpu_size_ = 0;
        bool owner_gpu_ = true;

        int device_id_ = 0;
    };

    // MonopolyAllocator: 针对 Tensor 的独占式分配池，避免频繁分配显存，结合线程池实现高并发推理。
    template<class _ItemType>
    class MonopolyAllocator {
    public:
        class MonopolyData {
        public:
            std::shared_ptr<_ItemType>& data() { return data_; }
            void release() { manager_->release_one(this); }
        private:
            explicit MonopolyData(MonopolyAllocator* pmanager) : manager_(pmanager) {}
            friend class MonopolyAllocator;
            MonopolyAllocator* manager_ = nullptr;
            std::shared_ptr<_ItemType> data_;
            bool available_ = true;
        };
        using MonopolyDataPointer = std::shared_ptr<MonopolyData>;

        // 省略具体实现细节，内部通过 condition_variable 和 mutex 控制查询 (query) 与释放 (release_one)
    };
}

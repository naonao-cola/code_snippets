#include <algorithm>
#include <execution>
#include <type_traits>
#include "MemoryPool.h"
#include "Logger.h"
#include "Utils.h"
#define KB 1024ULL
#define MB (KB*KB)
#define GB (MB*KB)

MemoryPool::MemoryPool(){}

MemoryPool::~MemoryPool()
{
    release();
}


typedef  std::pair<size_t, std::vector<size_t>> info_pair;
typedef  std::map<size_t, std::vector<size_t>> map_pair;

void MemoryPool::Initialize(const json& commonParams)
{
    release();
    blockSizeOffset_ = Utils::GetProperty(commonParams, "block_size_offset", 0) * MB;
    for (int i = 0; i < MAX_PRE_ALLOC_GROUP; i++) {
        std::string blockCntKey = fmt::format("block_count_{}", i);
        std::string blockSizeKey = fmt::format("block_size_{}", i);
        if (commonParams.contains(blockCntKey) && commonParams.contains(blockSizeKey) && commonParams[blockCntKey] > 0){
            int blockCnt = Utils::GetProperty(commonParams, blockCntKey, 0);
            int blockSize = Utils::GetProperty(commonParams, blockSizeKey, 0);
            if (blockCnt > 0 && blockSize > 0) {
                AllocMem(blockCnt, blockSize * MB);
                LOGI("Alloc memory blocks, Count={}, Size={} MB", blockCnt, blockSize);
            }
            //偏移内存先占位，不申请内存。
            std::vector<size_t> temp(blockCnt, 0);
            //若存在则覆盖
            m_memory_info[blockSize * MB]= temp;
            m_memory_info[blockSize * MB + blockSizeOffset_] = std::vector<size_t>{};
        }
    }
    SortBlocks();
}

void MemoryPool::AllocMem(size_t blockCount, size_t blockSize)
{
    while (m_lock.test_and_set(std::memory_order_acquire));
    memoryBlocks_.reserve(blockCount);
    for (size_t i = 0; i < blockCount; ++i) {
        MemoryBlock* block = new MemoryBlock(blockSize);
        if (block->data != nullptr) {
            memoryBlocks_.emplace_back(block);
        }
        else {
            LOGE("Alloc memory fail. size:{}", blockSize);
        }
    }
    m_lock.clear(std::memory_order_release);
}


void MemoryPool::CreateMat(cv::Mat& img,int rows, int cols, int type, bool setFlag)
{
    while (m_lock.test_and_set(std::memory_order_acquire));
    size_t requiredSize = static_cast<size_t>(rows * cols * CV_ELEM_SIZE(type));
    MemoryBlock* memBlock = nullptr;

    map_pair::iterator map_ptr = std::lower_bound(m_memory_info.begin(), m_memory_info.end(), std::make_pair(requiredSize, std::vector<size_t>{}),[&](const info_pair t_lhs, const info_pair t_rhs) { return t_lhs.first <= t_rhs.first ? true : false;});
    if (map_ptr== m_memory_info.end()) throw std::runtime_error("malloc memory is bigger");
    std::vector<size_t>::iterator vec_ptr = std::find(map_ptr->second.begin(), map_ptr->second.end(), 0);

    if (vec_ptr != map_ptr->second.end()) {
        size_t map_index = std::accumulate(m_memory_info.begin(), map_ptr, 0, [](const int& t_lhs, const info_pair t_rhs) {return t_lhs + t_rhs.second.size(); });

        int vec_index = vec_ptr - map_ptr->second.begin();
        memBlock = memoryBlocks_[vec_index + map_index];
        memBlock->inuse = true;
        *vec_ptr = 1;
    }
    else {
        memBlock = new MemoryBlock(requiredSize);
        if (memBlock->data == nullptr) {
            LOGE("Alloc MemoryBlock fail. requiredSize:{}", requiredSize);
            m_lock.clear(std::memory_order_release);
            return ;
        }
        memBlock->inuse = true;
        memoryBlocks_.emplace_back(memBlock);
        map_ptr->second.emplace_back(1);
        SortBlocks();
    }
    img = cv::Mat{ rows, cols, type, (uchar*)memBlock->data};
    if (setFlag) {
        img.setTo(0);
    }
    m_lock.clear(std::memory_order_release);
    return ;
}

void MemoryPool::ReleaseMat(cv::Mat mat)
{
    while (m_lock_release.test_and_set(std::memory_order_acquire));
    MemoryBlock* memBlock = nullptr;
    size_t requiredSize = static_cast<size_t>(mat.rows * mat.cols * CV_ELEM_SIZE(mat.type()));

    map_pair::iterator map_ptr = std::upper_bound(m_memory_info.begin(), m_memory_info.end(), std::make_pair(requiredSize, std::vector<size_t>{}), [&](const info_pair& a, const info_pair& b) { return a.first <= b.first ? true : false; });
    size_t map_index = std::accumulate(m_memory_info.begin(), map_ptr, 0, [](const int& t_lhs, const info_pair t_rhs) {return t_lhs + t_rhs.second.size(); });
    auto ret = std::find_if(&memoryBlocks_[map_index], &memoryBlocks_[map_index + map_ptr->second.size() - 1], [&](const MemoryBlock* a) { return a->data == mat.data ? true : false; });

    int vec_index = ret - (&memoryBlocks_[map_index]);
    int index = ret - &memoryBlocks_[0];
    memBlock = memoryBlocks_[index];
    memBlock->inuse = false;
    map_ptr->second[vec_index] = 0;
    m_lock_release.clear(std::memory_order_release);

}

void MemoryPool::Dump() {
    while (m_lock_release.test_and_set(std::memory_order_acquire));
    std::for_each(memoryBlocks_.begin(), memoryBlocks_.end(), [&](MemoryBlock* t_lhs) { LOGI("[MemoryBlock] size:{}   inuse:{}", t_lhs->size / MB * 1.0, t_lhs->inuse);});
    m_lock_release.clear(std::memory_order_release);
}

void MemoryPool::SortBlocks() {
    std::sort(memoryBlocks_.begin(), memoryBlocks_.end(),[](const MemoryBlock* a, const MemoryBlock* b){return a->size < b->size;});
}

void MemoryPool::release() {
    if (memoryBlocks_.size() >= 0 || m_memory_info.size() >= 0) {
        std::for_each(memoryBlocks_.begin(), memoryBlocks_.end(), [](MemoryBlock*t_lhs) {
            delete[] t_lhs->data;
            t_lhs->data = nullptr;
            delete t_lhs;
            t_lhs = nullptr;
            });
        memoryBlocks_.clear();
        m_memory_info.clear();
    }
}
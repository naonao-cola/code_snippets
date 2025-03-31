#pragma once
#include "../framework/Defines.h"
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <opencv2/core/mat.hpp>
#include <vector>


#define MAX_PRE_ALLOC_GROUP 5

class MemoryBlock {
public:
    MemoryBlock(size_t blocksize)
        : size(blocksize)
    {
        this->data = new uchar[size]();
        this->inuse = false;
    }

public:
    size_t size;
    void* data;
    bool inuse;
};

class MemoryPool {
public:
    DECLARE_SINGLETON(MemoryPool);
    // MemoryPool();
    // ~MemoryPool();

    void Initialize(const json& commonParams);
    void AllocMem(size_t blockCount, size_t blockSize);
    void CreateMat(cv::Mat& img, int rows, int cols, int type, bool setFlag = false);
    void ReleaseMat(cv::Mat mat);
    void Dump();
    std::vector<MemoryBlock*> memoryBlocks_;
    int new_count = 0;

private:
    IMPLEMENT_SINGLETON(MemoryPool);
    void SortBlocks();
    void release();

private:
    // std::mutex mutex_;
    int blockSizeOffset_;

    std::atomic_flag m_lock = ATOMIC_FLAG_INIT;
    std::atomic_flag m_lock_release = ATOMIC_FLAG_INIT;
    std::map<size_t, std::vector<size_t>> m_memory_info;
};

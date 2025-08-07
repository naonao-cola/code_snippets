
/**
 * @FilePath     : /test03/src/utils.h
 * @Description  :
 * @Author       : naonao
 * @Date         : 2025-05-30 12:49:25
 * @Version      : 0.0.1
 * @LastEditors  : naonao
 * @LastEditTime : 2025-07-25 10:10:53
 * @Copyright (c) 2025 by G, All Rights Reserved.
 **/
#pragma once

#ifndef __UTILS_H__
#define __UTILS_H__

#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <stdint.h>
#include <stdio.h>
#include <vector>
#include <string.h>
#include <algorithm>
#include <dirent.h>
#define TICK(x) auto bench_##x = std::chrono::high_resolution_clock::now();
#define TOCK(x)                                                                                                                                  \
    std::cout << #x ": " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - bench_##x).count() \
              << " us " << std::endl;

namespace utils {

float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1, float ymax1);

int nms(int validCount, std::vector<float>& outputLocations, std::vector<int> classIds, std::vector<int>& order,
    int filterId, float threshold, const int max_det = 3500);

void generateRandomData(int count, std::vector<float>& outputLocations,  std::vector<float>& prob_vec,std::vector<int>& classIds);

void LoadImagePath(std::string imgDirPath, std::vector<std::string>& vimgPath);
} // namespace utils

#endif // __UTILS_H__

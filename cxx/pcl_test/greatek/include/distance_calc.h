
#pragma once

#ifndef DISTANCE_CALC_H
#define DISTANCE_CALC_H

// CUDA 版本的点到点最小距离计算
// cloudA 和 cloudB 应该是以 x,y,z 连续存储的 float 数组
// numA 和 numB 是点的数量
float calculateMinDistanceCUDA(const float* cloudA, int numA, const float* cloudB, int numB);

#endif // DISTANCE_CALC_H
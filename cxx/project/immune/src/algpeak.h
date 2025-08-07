/*!
* @file alg.h
* @brief 算法模块接口定义
* @version 版本号
* @date 2022.03.28
* @author 谢丽慧
*
 */
/*
--lua算法返回validflag:
-  0代表成功，
- -1代表试剂卡信息参数错误；
- -2代参数解析出错,
- -3代表C线失效,
- -4代表校准系数超出合法范围，
- -5代表校准两个通道判定错误，
- -6代表校准卡信息参数888通道不合法,
建议：-1.-2，-6表示试剂卡参数解析错误， -3：上报本轮测试失效； -4，-5表示本轮校准失败
*/
#ifndef __ALGPEAK_H

#define __ALGPEAK_H
#include <stdint.h>
#include <stdlib.h>
#include "algcommon.h"


int FlipHorizontal(int* data, int data_len);

int FlipVertical(int* data, int data_len);

int DataFilter(LineResult* line_rst, AlgInput* input, Alg_HandleTypeDef* handle, int model);

int GenerateGaussTemp(float* gausstemp, double* sum, float sigma, int window);

int GaussianFilter(float* dst, int* src, float* gausstemp, float gausstempsum, int data_len, int window);

int ComGaussianFilter(float* dst, int* src, float* gausstemp, float gausstempsum, int data_len, int window);

int FindMaxValue(float* maxValue, int* maxpoint, float* src, int N);

int GetMinValue(int* src, int N);

int GetMaxValue(int* src, int N);

int FindPeak(FindPeakInfo* find_peak_info, float* data, int data_len, int start, int end);

int SearchLRvalley(int* left_point, int* right_point, float* data, int s_left, int s_right, int start, int end);

int SecondFindPeak(FindPeakInfo* find_peak_info, float* data, MergeInfo* mergePeak, int merge_cnt, int start, int end);

int FindLineInfo(LineResult* line_result, AlgInput* input, char* coef);

int GetSignalRegion(SingleLineRst* line_info, float* data, int signal_window, int point, int data_len);

int AlgGetLineResult(AlgInput* input, LineResult* line_rst, Alg_HandleTypeDef* handle, char* coef);

int CalculateSignal(LineResult* line_result, int line_index, int Tmodel, int Cmodel, int decimal);

int CommonProcessLineInfo(LineResult* line_result, int line_index, int start, int end, int signal_window, float coef_value,
                          int Tmodel, int Cmodel, float gate, int flag);
int MedianFilter(float* dst, int* src, int data_len);

#endif  // __ALGPEAK_H
/*!
* @file alg.h
* @brief 算法模块接口定义
* @version 版本号
* @date 2022.03.28
* @author 谢丽慧
*
 */

#ifndef __ALG_H

#define __ALG_H
#include <stdint.h>
#include <stdlib.h>
#include <string>

#include "algcommon.h"

#define ALG_VERSION  "V1.0.00.20241119"

// 获取C算法版本信息
char* AlgGetVersion();

// 算法初始化
int AlgInit(Alg_HandleTypeDef* handle, const std::string& lua_file);

// 结果初始化
int RstInit(AlgResult* algrst);

// 通过试剂卡信息参数初始化
int InitCardParam(AlgInput* input, char* cardTxt);

// 载入数据
int LoadData(AlgInput* input, int* data, int datalen);

// 数据计算
int ImmunoCalculate(char* json_buf, AlgResult* algrst, char* cardinfo, AlgInput input, Alg_HandleTypeDef* handle, char* coef);

// 获取方法学，MEATHOD_IM 0 ，MEATHOD_CG 1
int GetMethod(AlgInput input);

// 获取测试模式
int GetTestMode(AlgInput input);

// 获取高低增益，GAIN_LOW 1， GAIN_HIGH 2
int GetGain(AlgInput input);

#endif  // __ALG_H
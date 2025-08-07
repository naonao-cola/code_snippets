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
#ifndef __ALGJSON_H

#define __ALGJSON_H
#include <stdint.h>
#include <stdlib.h>
#include "cJSON/cJSON.h"
#include "algcommon.h"

// struct lua文件解析浓度所需的信息
typedef struct
{
  char strCline[32];                  /**< C线信息*/
  char strTline[128];                 /**< T线信息*/
  char* strcard_info;                 /**< 试剂卡相关的信息*/
  char* sample_id;                    /**< 区分校准和质控 1为校准*/
  char* nature;                       /**< 定性定量标志*/
}AlgLuaInput;

// 获取线信息
int GetLineParam(char* cardinfo, AlgInput* input);

// 获取方法学，荧光或者胶体金
int GetMethodParam(char* cardinfo, AlgInput* input);

// 获取通道模式和门限
int GetChannelModeAndGate(char* cardinfo, AlgInput* input);

// lua编码
int EncodeLuaInput(char *json_buf, int bufsize, AlgLuaInput *luaipt);

// 转化成lua所需的格式
int ConverLuaInput(AlgLuaInput* luaipt, int cid, int cerr, char* cardinfo, LineResult* line_rst, int testMode, int nature);

// lua 解码
int DecodeLuaOutput(char* opt_text, ChannelResult* channel_rst);

// 转换峰参数
int ConverPeakInfoInput(char* strCline, char* strTsignal, LineResult* line_rst, int testMode);
#endif  // __ALGJSON_H
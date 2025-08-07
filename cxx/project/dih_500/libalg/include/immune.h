
// Created by y on 24-4-20.
//

#ifndef IMMUNE_PROJECT_IMMUNE_H
#define IMMUNE_PROJECT_IMMUNE_H

#include <iostream>
#include <vector>

#define MAX_LINE_CNT 5          /** 峰最大数目 **/
#define MAX_SAMPLE_CNT 400      /** 信号最大长度 **/
#define MAX_CHANNEL_NAME_LEN 16 /** 单通道名字最大长度 **/
#define MAX_CHANNEL_CNT 4       /** 预定义检测项目个数，待定 **/
#define MAX_CARD_INFO 6800      /** 试剂卡信息长度*/

// 滤波参数
#define GAUSSAIN_FILTER_WINDOWN 6     /** 单次高斯滤波窗口参数 **/
#define MEAN_FILTER_WINDOWN 2         /** 均值滤波窗口参数 **/
#define COM_GAUSSAIN_FILTER_WINDOWN 6 /** 联合高斯滤波窗口参数 **/

// 内容开辟确认
#define MAX_INPUT_JSON_BUFFER 3000  /** 输入参数长度，lua模块输入 */
#define MAX_OUTPUT_JSON_BUFFER 2400 /** 输出参数长度，lua模块输出 */
#define MAX_DECODE_CARD_INFO 2800   /** 试剂卡信息长度 */
#define MAX_ENCODE_CARD_INFO 3200   /** 加密的多个机型的试剂卡长度 */
#define MAX_ENCODE_ONECARD_INFO 800 /** 加密的多个机型的试剂卡长度 */

#define MACHINETYPE 0 /** 机型类型 */
/* 函数返回状态 */
typedef enum
{
    ALG_ERR_O = -1, /** 失败 **/
    ALG_OK_O  = 0,  /** 成功 **/
} Alg_State_Out;

/**
 * \struct 单个项目检测结果
 */
typedef struct
{
    int   channel_no;                         /**< 测试项目编号 */
    char  channel_name[MAX_CHANNEL_NAME_LEN]; /**< 测试项目名字 */
    int   validflag;                          /**< 计算有效值 0代表成功，-1代表失败 */
    char  signal[16];                         /**< 项目的信号值,字符串 */
    int   nature_flag;                        /**< 定性定量标志 */
    char  nature[8];                          /**< 定性结果返回 */
    char  concentration[24];                  /**< 项目的浓度字符串 */
    float concentration_value;                /**< 项目浓度值 */
    char  unit[16];                           /**< 浓度单位 */
    float coef;                               /**< 校准系数 */
    char  mode[16];                           /**< 通道模式 */
} SingleChannelRstOut;

/**
 * \struct 多个项目的检测结果
 */
typedef struct
{
    int                 channel_cnt;                         /**< 实际测试项目通道数 */
    SingleChannelRstOut single_channel_rst[MAX_CHANNEL_CNT]; /**< 通道检测结果 */
} ChannelResultOut;

/**
 * \struct 单个检测线结果
 */
typedef struct
{
    int  signal_start; /**< line-信号起始点 */
    int  signal_end;   /**< line-信号结束点 */
    int  left_valley;  /**< line-左侧谷点 */
    int  right_valley; /**< line-右侧谷点 */
    int  max_point;    /**< line-max值对应的点即保留时间最大值对应点 */
    int  max_value;    /**< line-max值 */
    int  base_line;    /**< 每条检测线的基线值 */
    int  model;        /**< 计算模式 */
    int  area;         /**< line-面积值 整型 */
    char signal[16];   /**< 计算信号 字符串，如果是小数，保留四位小数 */
    int  line_id;      /**< 检测线的id */
} SingleLineRstOut;

/**
 * \struct 检测线信号计算结果
 */

typedef struct
{
    int              line_cnt;                      /**< 测试线条数 */
    int              channel_cnt;                   /**< 测试通道数 */
    int              length;                        /**< 实际信号长度（采样点的个数）*/
    float            filter_data[MAX_SAMPLE_CNT];   /**< 滤波后的信号值，MAX_SAMPLE_CNT为采样点的个数 */
    SingleLineRstOut single_line_rst[MAX_LINE_CNT]; /**< 测试线的检测结果 */
    int16_t          input_length;                  /**< 输入信号长度（采样点的个数）*/
    float            input_data[MAX_SAMPLE_CNT];    /**< 输入信号值，MAX_SAMPLE_CNT为采样点的个数 */
} LineResultOut;

/**
 * \struct 算法最终结果
 */

typedef struct
{
    int              cal_flag;    /**< 检测模式，校准1，质控2，常规0 */
    LineResultOut    line_rst;    /**< 检测线的各个结果 */
    ChannelResultOut channel_rst; /**< 通道检测结果 */
} AlgResultOut;

typedef struct
{
    int              cal_flag;    /**< 检测模式，校准1，质控2，常规0 */
    LineResultOut    line_rst;    /**< 检测线的各个结果 */
    ChannelResultOut channel_rst; /**< 通道检测结果 */
} AlgResult;

/*
    * 算法初始化
    * @param lua_file lua文件名
    * @return -1 失败，0 表示成功
    */
int
AlgImmuneInitOut(const std::string& lua_file);

/*
 * 获取算法版本信息
 * @param alg_version     算法处理C部分版本信息
 * @param qr_json_version 二维码解析版本
 * @param version_length  版本长度
 * @return -1 失败，0 表示成功
 */
int AlgImmuneGetVersion(char* alg_version, char* qr_json_version, char* lua_version, char* main_versioon, int version_length);

/*
 * 免疫算法接口
 * @param card_info            加密试剂卡信息
 * @param data_v               采样点数据
 * @param coef                 校准系数
 * @param algrstout            输出结构体
 * @param decoded_card_info    解密的试剂卡信息
 * @return  0表示成功，＜0表示各类失败
 */
int AlgImmuneCalculateOut(
    const std::string& card_info, const std::vector<float>& data_v, char* coef, AlgResultOut* algrstout, char* decoded_card_info);

/*
 * 获取试剂卡信息
 * @param group           机型系列
 * @param mask            型号掩码
 * @param buf             解码后的二维码信息
 * @param size            解码前的二维码长度
 * @param err             错误信息
 * @card_info             解密后的试剂卡信息
 * @return
 */
int GetCardInfo(uint32_t group, uint32_t mask, char* buf, uint32_t size, int* err, char* card_info);

#endif   // IMMUNE_PROJECT_IMMUNE_H

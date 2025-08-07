
#ifndef __ALGCOMMON_H

#define __ALGCOMMON_H
#include <stdint.h>
#include <stdlib.h>

#define Alg_malloc(size) malloc(size)
#define Alg_free(size) free(size)
#define Algo_GetGauss1(_HANDLE_) ((_HANDLE_)->gausstemp1)
#define Algo_GetGauss2(_HANDLE_) ((_HANDLE_)->gausstemp2)

// 最大值设定
#define MAX_LINE_CNT 5 /** 峰最大数目**/
#define MAX_SAMPLE_CNT 400 /** 信号最大长度 **/
#define MAX_CHANNEL_NAME_LEN 16
#define MAX_CHANNEL_CNT 4 /** 预定义检测项目个数，待定 **/

// 寻峰
#define MAX_FIRST_PEAK_CNT 30 /** 预定义第一次寻峰的最大数目**/
#define MAX_SECOND_PEAK_CNT 15 /** 预定义第二次寻峰的最大数目**/
#define WIDTH_LLIMIT 20 /** 最小峰宽 **/
#define HALF_WIDTH_LIMIT 15 /** 半峰宽度 **/
#define MERGE_LIMIT 10 /** 峰合并参数**/

// 滤波参数
#define GAUSSAIN_FILTER_WINDOWN 6 /** 单次高斯滤波窗口参数 **/
#define MEAN_FILTER_WINDOWN 2 /** 均值滤波窗口参数 **/
#define MEDAIN_FILTER_WINDOWN 2 /** 中值滤波窗口参数 **/
#define COM_GAUSSAIN_FILTER_WINDOWN 6 /** 联合高斯滤波窗口参数 **/
#define PI 3.1415

// 方法学
#define MEATHOD_IM 0 /** 荧光免疫 **/
#define MEATHOD_CG 1 /** 胶体金免疫 **/

// 定量定性判定
#define QUANTITATINE 0 /*定量*/
#define QUALITATIVES 1 /*定性夹心*/
#define HQUANTITATINES 2 /*半定量夹心*/
#define QUALITATIVEC 3 /*定性竞争*/
#define HQUANTITATINEC 4 /*半定量竞争*/

// 高低增益
#define GAIN_LOW 1 /** 低增益 **/
#define GAIN_HIGH 2 /** 高增益 **/

// 计算面积
#define MAX_SIGNAL_WIDTH 200 /** 有效信号值最大宽度（计算面积有效期）**/
#define CALIPROID 105 /**  免疫校准ID **/

// 内容开辟确认
#define MAX_INPUT_JSON_BUFFER 3000 /** 输入参数长度，lua模块输入 */
#define MAX_OUTPUT_JSON_BUFFER 2400 /** 输出参数长度，lua模块输出 */

/* 函数返回状态 */
typedef enum {
    TNORMAL = 0, /** 常规测试 **/
    TCAL = 1, /** 校准测试 **/
    TQC = 2, /** 质控测试 **/
} TestMode;

/* 函数返回状态 */
typedef enum {
    ALG_ERR = -1, /** 失败 **/
    ALG_OK = 0, /** 成功 **/
} Alg_State;

/**
 * \struct 寻峰结果结构体
 */
typedef struct
{
    int peak_point; /**< 峰点 */
    int start_point; /**< 左波谷点 */
    int end_point; /**< 右波谷点 */
    float peak_value; /**< 峰值 */
} OnePeakInfo;

/**
 * \struct 寻峰结果结构体
 */
typedef struct
{
    int peak_num; /**< 寻峰最波峰数目 */
    OnePeakInfo one_peak_info[MAX_SECOND_PEAK_CNT]; /**< 寻峰结果 */
} FindPeakInfo;

/**
 * \struct 单个项目检测结果
 */
typedef struct
{
    int channel_no; /**< 测试项目编号 */
    char channel_name[MAX_CHANNEL_NAME_LEN]; /**< 测试项目名字 */
    int validflag; /**< 计算有效值 0代表成功，-1代表失败*/
    char signal[16]; /**< 项目的信号值,字符串*/
    int nature_flag; /**< 定性定量标志*/
    char nature[8]; /**< 定性结果返回 */
    char concentration[24]; /**< 项目的浓度字符串 */
    float concentration_value; /**< 项目浓度值 */
    char unit[16]; /**< 浓度单位 */
    float coef; /**< 校准系数 */
    char mode[16]; /**< 模式 */
} SingleChannelRst;

/**
 * \struct 多个项目的检测结果
 */
typedef struct
{
    int channel_cnt; /**< 实际测试项目通道数 */
    SingleChannelRst single_channel_rst[MAX_CHANNEL_CNT]; /**< 通道检测结果 */
} ChannelResult;

/**
 * \struct 单个检测线结果
 */
typedef struct
{
    int signal_start; /**< line-信号起始点 */
    int signal_end; /**< line-信号结束点 */
    int left_valley; /**< line-左侧谷点 */
    int right_valley; /**< line-右侧谷点 */
    int max_point; /**< line-max值对应的点即保留时间最大值对应点 */
    int max_value; /**< line-max值 */
    int base_line; /**< 每条检测线的基线值 */
    int model; /**< 计算模式 */
    int area; /**< line-面积值 整型 */
    char signal[16]; /**< 计算信号 字符串，如果是小数，保留四位小数 */
    int line_id; /**< 检测线的id */
} SingleLineRst;

/**
 * \struct 检测线信号计算结果
 */

typedef struct
{
    int line_cnt; /**< 测试线条数 */
    int channel_cnt; /**< 测试通道数 */
    int length; /**< 实际信号长度（采样点的个数）*/
    float filter_data[MAX_SAMPLE_CNT]; /**< 滤波后的信号值，MAX_SAMPLE_CNT为采样点的个数 */
    SingleLineRst single_line_rst[MAX_LINE_CNT]; /**< 测试线的检测结果 */
    int16_t input_length; /**< 输入信号长度（采样点的个数）*/
    float input_data[MAX_SAMPLE_CNT]; /**< 输入信号值，MAX_SAMPLE_CNT为采样点的个数 */
} LineResult;

/**
 * \struct 算法最终结果
 */

typedef struct
{
    int cal_flag; /**< 检测模式，校准1，质控2，常规0 */
    LineResult line_rst; /**< 检测线的各个结果 */
    ChannelResult channel_rst; /**< 通道检测结果 */
} AlgResult;

typedef struct
{
    int old_point; /**< 峰位置 */
    int new_point; /**< 合并后的峰位置 */
    int adj_point; /**< 相邻合并的点 */
    int flag; /**< 合并标志 */
} MergeInfo;

/* *************************检测线输入结构体**************************************/

// C检测线的输入结构体
typedef struct
{
    int line_id; /**< 检测线的id */
    int dis; /**< 与质控线的距离*/
    int signal_window; /**< 面积计算宽度 */
    int Tmodel; /**< 计算T模式 */
    int Cmodel; /**< 计算C模式 */
    int decimal; /**< 保留位数*/
    float gate; /**< 门限*/
} OneLinePara;

typedef struct
{
    int cid; /**< c线id */
    int cerr; /**< c线失效值 */
    int cstart; /**< c线起始点 */
    int twindow; /**< T线寻峰宽度 */
    OneLinePara paras[MAX_LINE_CNT]; /**< 线信息 */
} LinePara;

// 算法参数输入结构体

typedef struct
{
    int line_cnt; /**< 线条数目（含C线）*/
    int channel_cnt; /**< 通道数目 */
    int method; /**< 荧光或者胶体金模式 */
    int testmode; /**< 是否未校准*/
    int proid; /**< 项目id*/
    int gain; /**< 通道增益*/
    int nature; /**< 测量性质*/
    int length; /**< 实际信号长度（采样点的个数）*/
    int data[MAX_SAMPLE_CNT]; /**< 待处理的数据对象 */
    LinePara line_para; /**< 测试线先验信息 */
} AlgInput;

typedef struct
{
    void* init_flag;
    float gausstemp[GAUSSAIN_FILTER_WINDOWN * 2 + 1];
    double gausstempsum;
    float comgausstemp[COM_GAUSSAIN_FILTER_WINDOWN * 2 + 1];
    double comgausstempsum;
} Alg_HandleTypeDef;

#endif // __ALGCOMMON_H
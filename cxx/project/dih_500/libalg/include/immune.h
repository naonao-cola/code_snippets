//
// Created by y on 24-4-20.
//

#ifndef IMMUNE_PROJECT_IMMUNE_H
#define IMMUNE_PROJECT_IMMUNE_H

#include <iostream>
#include <vector>

#define MAX_LINE_CNT                5                             /** 峰最大数目**/
#define MAX_SAMPLE_CNT              400                           /** 信号最大长度 **/
#define MAX_CHANNEL_NAME_LEN        16
#define MAX_CHANNEL_CNT             4                             /** 预定义检测项目个数，待定 **/

#define MAX_CARD_INFO                6800                         /** 试剂卡信息长度*/
// 滤波参数
#define GAUSSAIN_FILTER_WINDOWN         6                         /** 单次高斯滤波窗口参数 **/
#define MEAN_FILTER_WINDOWN             2                         /** 均值滤波窗口参数 **/
#define COM_GAUSSAIN_FILTER_WINDOWN     6                         /** 联合高斯滤波窗口参数 **/

// 内容开辟确认
#define MAX_INPUT_JSON_BUFFER        3000                         /** 输入参数长度，lua模块输入 */
#define MAX_OUTPUT_JSON_BUFFER       2400                         /** 输出参数长度，lua模块输出 */
#define MAX_DECODE_CARD_INFO         2800                         /** 试剂卡信息长度 */
#define MAX_ENCODE_CARD_INFO         3200                         /** 加密的多个机型的试剂卡长度 */
#define MAX_ENCODE_ONECARD_INFO      800                          /** 加密的多个机型的试剂卡长度 */

#define MACHINETYPE                   0                          //机型类型
/* 函数返回状态 */
typedef enum
{
  ALG_ERR_O = -1,                               /** 失败 **/
  ALG_OK_O = 0,                                /** 成功 **/
}Alg_State_Out;

/**
 * \struct 单个项目检测结果
 */
typedef struct
{
  int           channel_no;                            /**< 测试项目编号 */
  char          channel_name[MAX_CHANNEL_NAME_LEN];    /**< 测试项目名字 */
  int           validflag;                             /**< 计算有效值 0代表成功，-1代表失败*/
  char          signal[16];                            /**< 项目的信号值,字符串*/
  int           nature_flag;                           /**< 定性定量标志*/
  char          nature[8];                             /**< 定性结果返回 */
  char          concentration[24];		       /**< 项目的浓度字符串 */
  float		concentration_value;		       /**< 项目浓度值 */
  char		unit[16];			       /**< 浓度单位 */
  float		coef;				       /**< 校准系数 */
  char          mode[16];                              /**< 通道模式 */
}SingleChannelRstOut;

/**
 * \struct 多个项目的检测结果
 */
typedef struct
{
  int              channel_cnt;                           /**< 实际测试项目通道数 */
  SingleChannelRstOut  single_channel_rst[MAX_CHANNEL_CNT];   /**< 通道检测结果 */
}ChannelResultOut;

/**
 * \struct 单个检测线结果
 */
typedef struct
{
  int           signal_start;                          /**< line-信号起始点 */
  int           signal_end;                            /**< line-信号结束点 */
  int           left_valley;                           /**< line-左侧谷点 */
  int           right_valley;                          /**< line-右侧谷点 */
  int           max_point;                             /**< line-max值对应的点即保留时间最大值对应点 */
  int           max_value;                             /**< line-max值 */
  int           base_line;                             /**< 每条检测线的基线值 */
  int           model;                                 /**< 计算模式 */
  int           area;				        /**< line-面积值 整型 */
  char          signal[16];                            /**< 计算信号 字符串，如果是小数，保留四位小数 */
  int           line_id;                               /**< 检测线的id */
}SingleLineRstOut;

/**
 * \struct 检测线信号计算结果
 */

typedef struct
{
  int              line_cnt;                               /**< 测试线条数 */
  int              channel_cnt;                            /**< 测试通道数 */
  int              length;                                 /**< 实际信号长度（采样点的个数）*/
  float            filter_data[MAX_SAMPLE_CNT];            /**< 滤波后的信号值，MAX_SAMPLE_CNT为采样点的个数 */
  SingleLineRstOut single_line_rst[MAX_LINE_CNT];          /**< 测试线的检测结果 */
  int16_t          input_length;                          /**< 输入信号长度（采样点的个数）*/
  float            input_data[MAX_SAMPLE_CNT];            /**< 输入信号值，MAX_SAMPLE_CNT为采样点的个数 */
}LineResultOut;

/**
 * \struct 算法最终结果
 */

typedef struct
{
  LineResultOut        line_rst;                             /**< 检测线的各个结果 */
  ChannelResultOut     channel_rst;                          /**< 通道检测结果 */
}AlgResultOut;


// 算法初始化
int AlgImmuneInitOut(const std::string& lua_file);

void AlgImmuneGetVersion(char* alg_version, char* qr_json_version, int version_length);

// 算法运行
int AlgImmuneCalculateOut( const std::string& encodecard, const std::vector<float>&data_v,  char* coef,  AlgResultOut * algrstout, char* decoded_card_info);

// 获取试剂卡信息内容
int GetCardInfo(uint32_t group, uint32_t mask, char* buf, uint32_t size, int* err, char* card_info);

#endif // IMMUNE_PROJECT_IMMUNE_H
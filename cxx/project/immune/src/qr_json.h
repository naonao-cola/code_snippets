/********************************************************************************************************************************
* 二维码JSON转换H文件
* qr_json.h
*
* project: DI-200
*
* version          date            author          note
* 1.0              2024/02/20      unreal          1.初始版本
* 1.1              2024/03/22      unreal          1.适配新版定量/半定量方案，兼容胶体金
* 1.2              2024/03/25      unreal          1.增加协议号
* 1.3              2024/03/26      unreal          1.优化参数空值处理
* 1.4              2024/03/27      unreal          1.总长度计算目标修正为通道计数
* 1.5              2024/03/27      unreal          1.修正数组尺寸限制错误 2.修正负值浮点数转换错误
* 1.6              2024/03/28      unreal          1.修正日期转换错误
* 1.7              2024/07/26      unreal          1.增加三个属性（盒号、测试数、类型（场景属性），解决通道门限没有按照保留位数及逆行约束
* 2.0              2024/10/18      unreal          1.适配评审文档《免疫定量定性二维码编码规则-修订版2410121941》
*
 */
#ifndef __QR_JSON__H
#define __QR_JSON__H


extern "C"
{

#include <stdint.h>
#include "crc16.h"
#define QRJSON_VERSION                      "V1.0.00.20250626"
#define MAX_DECODE_CARD_INFO         2048       /** 单机型试剂卡信息解码之后长度 */
#define MAX_ENCODE_CARD_INFO         3200       /** 加密多个机型的试剂卡信息长度 */


/* 基础参数 */
#define QRJSON_CRC_ENABLE                   1 // CRC校验使能
#define QRJSON_PROTOCOL_NUM                 1 // 协议号

/* 字段分隔符 */
#define QRJSON_PROTOCOL_INTERVAL_SIGN       '&'
#define QRJSON_PARAM_INTERVAL_SIGN          ','

/* 特征文本 */
#define QRJSON_QRCODE_SIGN                  "reagent?"
#define QRJSON_WEBSITE_SIGN                 "http"
#define QRJSON_PROTOCOL_SIGN                "PR="
#define QRJSON_PARAM_SIGN                   "PA="
#define QRJSON_CHECK_SIGN                   "CRC="

/* 信息缓存数量定义 */
#define QRJSON_MAX_BUF_CHL                  4       //通道信息
#define QRJSON_MAX_BUF_LINE                 5       //线信息
#define QRJSON_MAX_BUF_CALCU                4       //定性与定量计算信息
#define QRJSON_MAX_BUF_EQUATION             5       //分段方程信息

/* 字段位数定义 */
#define QRJSON_MAX_CARD_BOX_NUM             4       //盒号
#define QRJSON_MAX_CARD_TEST_NUM            1       //盒装规格
#define QRJSON_MAX_CARD_BATCH_NUM           14      //批号
#define QRJSON_MAX_CARD_BATCH_EN_CODE       6       //批号英文编码
#define QRJSON_MAX_CARD_EXP_DATE            1       //有效期
#define QRJSON_MAX_CARD_SIGN_GAIN           1       //信号增益
#define QRJSON_MAX_CARD_WAIT_TIME           1       //等待时间
#define QRJSON_MAX_CARD_PROJ_ID             4       //项目ID
#define QRJSON_MAX_CARD_METHOD_ID           1       //方法学
#define QRJSON_MAX_LINE_CINDEX              1       //C线位置
#define QRJSON_MAX_CARD_LINE_CNT            1       //T线数量
#define QRJSON_MAX_CARD_CHL_CNT             1       //通道数量

#define QRJSON_MAX_CHL_NAME                 20      //通道名称
#define QRJSON_MAX_CHL_ID                   1       //通道ID
#define QRJSON_MAX_CHL_MODE                 2       //计算模式
#define QRJSON_MAX_CHL_UNIT                 2       //单位
#define QRJSON_MAX_CHL_DECIMAL              1       //结果保存位数
#define QRJSON_MAX_CHL_LIMIT                4       //低/高限显示值

#define QRJSON_MAX_DEV_GROUP                4       //机型系列
#define QRJSON_MAX_DEV_MASK                 8       //型号掩码
#define QRJSON_MAX_CLINE_START              2       //C线起点
#define QRJSON_MAX_CLINE_INVALID            1       //C线失效值
#define QRJSON_MAX_TLINE_WINDOW             1       //T线寻峰窗口

#define QRJSON_MAX_LINE_OFFSET              2       //线偏移量
#define QRJSON_MAX_LINE_WIDTH               1       //线宽度

#define QRJSON_MAX_CALCU_RES_MODE           1       //结果模式
#define QRJSON_MAX_EQUATION_SEGMENT         1       //方程分段数量
#define QRJSON_MAX_EQUATION_TYPE            1       //方程类型
#define QRJSON_MAX_EQUATION_PARAM_NUM       1       //方程参数数量
#define QRJSON_MAX_EQUATION_PARAM_LEN       6       //方程参数值
#define QRJSON_MAX_EQUATION_GATE            4       //方程低/高门限

#define QRJSON_MAX_RESULT_GATE_NUM          4       //定性门限数量
#define QRJSON_MAX_RESULT_GATE_LEN          5       //定性门限位数

#define QRJSON_MAX_CHECK_VAL                4       //CRC校验

#pragma pack(1) //设置字节对齐=1

/* 静态基础信息 */
typedef struct QRJsonStaticBase
{
  char box_number[QRJSON_MAX_CARD_BOX_NUM];
  char test_number[QRJSON_MAX_CARD_TEST_NUM];
  char batch_number[QRJSON_MAX_CARD_BATCH_NUM];
  char exp_date[QRJSON_MAX_CARD_EXP_DATE];
  char sign_gain[QRJSON_MAX_CARD_SIGN_GAIN];
  char wait_time[QRJSON_MAX_CARD_WAIT_TIME];
  char proj_id[QRJSON_MAX_CARD_PROJ_ID];
  char method_id[QRJSON_MAX_CARD_METHOD_ID];
  char cline_idx[QRJSON_MAX_LINE_CINDEX];
  char tline_cnt[QRJSON_MAX_CARD_LINE_CNT];
  char chl_cnt[QRJSON_MAX_CARD_CHL_CNT];
}QRJsonStaticBase_t;

/* 通道信息 */
typedef struct QRJsonChl
{
  char name[QRJSON_MAX_CHL_NAME];
  char id[QRJSON_MAX_CHL_ID];
  char mode[QRJSON_MAX_CHL_MODE];
  char unit[QRJSON_MAX_CHL_UNIT];
  char decimal[QRJSON_MAX_CHL_DECIMAL];
  char low_limit[QRJSON_MAX_CHL_LIMIT];
  char high_limit[QRJSON_MAX_CHL_LIMIT];
}QRJsonChl_t;

/* 基础信息区 */
typedef struct QRJsonBase
{
  QRJsonStaticBase_t *static_area;
  QRJsonChl_t *chl_list[QRJSON_MAX_BUF_CHL];
}QRJsonBase_t;

/* 静态特定信息 */
typedef struct QRJsonStaticSpecial
{
  char group[QRJSON_MAX_DEV_GROUP];
  char mask[QRJSON_MAX_DEV_MASK];
  char cline_start[QRJSON_MAX_CLINE_START];
  char cline_invalid[QRJSON_MAX_CLINE_INVALID];
  char tline_window[QRJSON_MAX_TLINE_WINDOW];
}QRJsonStaticSpecial_t;

/* 线信息 */
typedef struct QRJsonLine
{
  char offset[QRJSON_MAX_LINE_OFFSET];
  char width[QRJSON_MAX_LINE_WIDTH];
}QRJsonLine_t;

/* 分段方程信息 */
typedef struct QRJsonEquation
{
  char *type;
  char *param_cnt;
  char *param;
  char *gate_low;
  char *gate_high;
  char *next;
}QRJsonEquation_t;

/* 结果门限（定性） */
typedef struct QRJsonResultGate
{
  char value[QRJSON_MAX_RESULT_GATE_NUM][QRJSON_MAX_RESULT_GATE_LEN];
}QRJsonResultGate_t;

/* 定性与定量计算信息 */
typedef struct QRJsonCalcu
{
  char *res_mode;
  char *seg_cnt;
  QRJsonEquation_t equation[QRJSON_MAX_BUF_EQUATION];
  QRJsonResultGate_t *gate;
  char *next;
}QRJsonCalcu_t;

/* 特定信息区 */
typedef struct QRJsonSpecial
{
  QRJsonStaticSpecial_t *static_area;
  QRJsonLine_t *line_list[QRJSON_MAX_BUF_LINE];
  QRJsonCalcu_t calcu_list[QRJSON_MAX_BUF_EQUATION];
  QRJsonEquation_t last_equation;
  uint32_t last_chl;
  uint32_t last_seg;
}QRJsonSpecial_t;

/* 试剂卡信息 */
typedef struct QRJsonCard
{
  uint32_t protocal_num;
  QRJsonBase_t base;
  QRJsonSpecial_t special;
  char *website;
  char *check_crc;
  char *boundary;
}QRJsonCard_t;

char* QrJsonGetVersion(void);

int QRJson_SearchCard(QRJsonCard_t *card, uint32_t group, uint32_t mask, char *buf, uint32_t size, int *err);
QRJsonLine_t *QRJson_SearchLine(QRJsonCard_t *card, uint32_t idx, int *err);
QRJsonChl_t *QRJson_SearchChl(QRJsonCard_t *card, uint32_t idx, int *err);
QRJsonResultGate_t *QRJson_SearchResultGate(QRJsonCard_t *card, uint32_t chl_idx, int *err);

int QRJson_GetDevGroup(QRJsonCard_t *card, int *err);
int QRJson_GetDevMask(QRJsonCard_t* card, int* err);

int QRJson_GetBoxNumber(QRJsonCard_t* card, int* err);
int QRJson_GetTestNumber(QRJsonCard_t* card, int* err);
int QRJson_GetBatchNunber(char *buf, uint32_t size, QRJsonCard_t *card, int *err);
int QRJson_GetProDate(char *buf, uint32_t size, QRJsonCard_t *card, int *err);
int QRJson_GetExpDate(char *buf, uint32_t size, QRJsonCard_t *card, int *err);
int QRJson_GetSignGain(QRJsonCard_t *card, int *err);
int QRJson_GetLineCnt(QRJsonCard_t *card, int *err);
int QRJson_GetChlCnt(QRJsonCard_t *card, int *err);
int QRJson_GetWaitTime(QRJsonCard_t *card, int *err);
int QRJson_GetMethodID(QRJsonCard_t *card, int *err);
int QRJson_GetProjID(QRJsonCard_t *card, int *err);
int QRJson_GetProjName(char *buf, uint32_t size, QRJsonCard_t *card, int *err);

int QRJson_GetCLineIndex(QRJsonCard_t *card, int *err);
int QRJson_GetCLineStart(QRJsonCard_t *card, int *err);
int QRJson_GetCLineInvalid(QRJsonCard_t *card, int *err);
int QRJson_GetTLineWindow(QRJsonCard_t *card, int *err);
int QRJson_GetCLineOffset(QRJsonLine_t* line, int* err);
int QRJson_GetTLineOffset(QRJsonLine_t *line, int *err);
int QRJson_GetLineWidth(QRJsonLine_t *line, int *err);

int QRJson_GetChlID(QRJsonChl_t *chl, int *err);
int QRJson_GetChlName(char *buf, uint32_t size, QRJsonChl_t *chl, int *err);
int QRJson_GetChlUnit(char *buf, uint32_t size, QRJsonChl_t *chl, int *err);
uint8_t QRJson_GetChlModeNum(QRJsonChl_t *chl, uint32_t oft, int *err);
int QRJson_GetChlDecimal(QRJsonChl_t *chl, int *err);
int QRJson_GetChlLimit(char *buf, uint32_t size, QRJsonChl_t *chl, uint8_t high_flag, int decimal, int *err);

int QRJson_GetCalcuResultMode(QRJsonCard_t *card, uint32_t chl_idx, int *err);
int QRJson_GetEquationSegment(QRJsonCard_t *card, uint32_t chl_idx, int *err);
int QRJson_GetEquationType(QRJsonCard_t *card, uint32_t chl_idx, uint32_t seg_idx, int *err);
int QRJson_GetEquationParamCnt(QRJsonCard_t *card, uint32_t chl_idx, uint32_t seg_idx, int *err);
int QRJson_GetEquationParamVal(char* buf, uint32_t size, QRJsonCard_t *card, uint32_t chl_idx, uint32_t seg_idx, uint32_t param_idx, int *err);
int QRJson_GetEquationGate(char* buf, uint32_t size, QRJsonCard_t *card, uint32_t chl_idx, uint32_t seg_idx, uint8_t high_flag, int *err);
int QRJson_GetResultGate(char* buf, uint32_t size, QRJsonCard_t *card, uint32_t chl_idx, uint32_t gate_idx, int *err);

int QRJson_EncodeCardInfo(char *buf, uint32_t size, QRJsonCard_t *card, const char **err_string);
int QRJson_EncodeCardFile(char *buf, uint32_t size, QRJsonCard_t *card, const char **err_string);
}

#pragma pack(8) //设置字节对齐=8

#endif /* __QR_JSON__H */





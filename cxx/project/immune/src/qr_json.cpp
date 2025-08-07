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
#include "qr_json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include "cJSON/cJSON.h"
#if(QRJSON_CRC_ENABLE)

#endif /* QRJSON_CRC_ENABLE */

#ifndef NULL
#define NULL            ((void*)0)
#endif /* NULL */


/* 缓存大小定义 */
#define QRJSON_TEMP_BUFSIZE                 32
#define QRJSON_TEXT_BUFSIZE                 128
#define QRJSON_NEGATIVE_SIGN                2

/* JSON字段定义 */
#define CARD_ITEM_PROJ                      "project"
#define CARD_ITEM_CARD                      "reagent_card"
#define CARD_ITEM_INFORMATION               "infomation"

#define CARD_ITEM_DEV_GROUP                 "DEV"
#define CARD_ITEM_DEV_MASK                  "MASK"
#define CARD_ITEM_BOX_NUMBER                "Bnum"
#define CARD_ITEM_TEST_NUMBER               "Tnum"

#define CARD_ITEM_BATCH_NUMBER              "BN"
#define CARD_ITEM_PRODUCTION_DATE           "PD"
#define CARD_ITEM_EXPIRE_DATE               "ED"
#define CARD_ITEM_SIGNAL_GAIN               "Gain"
#define CARD_ITEM_WAIT_TIME                 "WTim"
#define CARD_ITEM_METHOD_ID                 "MID"
#define CARD_ITEM_ALGO_ID                   "AID"
#define CARD_ITEM_PROJ_ID                   "PID"
#define CARD_ITEM_PROJ_NAME                 "PNam"

#define CARD_ITEM_LINE                      "Line"
#define CARD_ITEM_LINE_CINDEX               "LCID"
#define CARD_ITEM_LINE_ID                   "LID"
#define CARD_ITEM_LINE_START                "LSta"
#define CARD_ITEM_LINE_END                  "LEnd"
#define CARD_ITEM_LINE_OFFSET               "LOft"
#define CARD_ITEM_LINE_WIDTH                "LWid"
#define CARD_ITEM_LINE_INVALID              "LErr"
#define CARD_ITEM_TLINE_WINDOW              "TWin"

#define CARD_ITEM_CHL                       "Chl"
#define CARD_ITEM_CHL_ID                    "CID"
#define CARD_ITEM_CHL_NAME                  "CNam"
#define CARD_ITEM_CHL_TMODE                 "TMode"
#define CARD_ITEM_CHL_CMODE                 "CMode"
#define CARD_ITEM_CHL_UNIT                  "Unit"
#define CARD_ITEM_CHL_DECIMAL               "Deci"
#define CARD_ITEM_CHL_HIGH_LIMIT            "HLmt"
#define CARD_ITEM_CHL_LOW_LIMIT             "LLmt"

#define CARD_ITEM_EQUATION_CNT              "ECnt"
#define CARD_ITEM_EQUATION                  "Equa"
#define CARD_ITEM_EQUATION_TYPE             "Type"
#define CARD_ITEM_EQUATION_PARAM_CNT        "PCnt"
#define CARD_ITEM_EQUATION_PARAM_VAL        "Para"
#define CARD_ITEM_EQUATION_GATE             "EGat"

#define CARD_ITEM_RESULT_GATE               "PGat"

/* 字符编码 */
const char qrjson_char_list[] = {
    '1', '2', '3', '4', '5', //5
    '6', '7', '8', '9', '0', //10
    'a', 'b', 'c', 'd', 'e', //15
    'f', 'g', 'h', 'i', 'j', //20
    'k', 'l', 'm', 'n', 'o', //25
    'p', 'q', 'r', 's', 't', //30
    'u', 'v', 'w', 'x', 'y', //35
    'z', 'A', 'B', 'C', 'D', //40
    'E', 'F', 'G', 'H', 'I', //45
    'J', 'K', 'L', 'M', 'N', //50
    'O', 'P', 'Q', 'R', 'S', //55
    'T', 'U', 'V', 'W', 'X', //60
    'Y', 'Z', '~', '!', '@', //65
    '#', '$', '%', '^', '&', //70
    '*', '(', ')', '[', ']', //75
    '{', '}', '<', '>', '_', //80
    '+', '-', '=', '.', ',',  //85
    ':', ';', '?', '/', '\\', //90
    '|', '"', '\''
};

/* 生产日期编号 */
int qrjson_expdate_list[] = {
    3, 6, 9, 12, 15, 18, 24, 30, 36
};

/* 等待时间编号 */
int qrjson_waittime_list[] = {
    3, 5, 10, 15, 20
};

/* C线偏移量编号 */
int qrjson_Cline_offset_list[] = {
    10, 20, 30, 40, 50, 60, 70, 80, 90
};


/* T线偏移量编号 */
int qrjson_Tline_offset_list[] = {
    10, 20, 30, 40, 50, 60, 70, 80, 90,100,
    110, 120, 130, 140, 150, 160, 170, 180, 190, 200,
    210, 220, 230, 240, 250, 260, 270
};

/* 线宽度编号表 */
int qrjson_line_width_list[] = {
    30, 35, 40, 45, 50, 55, 60, 65, 70
};

/* C线失效值编号表 */
int QRJson_line_invalid_list[] = {
    0, 2500, 5000, 7500, 10000, 15000, 20000, 30000, 40000, 50000
};


/* 每盒测试次数*/
int qrjson_testcnt_list[] = {
    99999, 1, 5, 10, 20, 25, 50, 100
};

/* 通道单位编号表 */
const char* qrjson_unit_list[] = {
    " ", //0
    "pg/mL", //1
    "ng/mL", //2
    "ug/mL", //3
    "mg/mL", //4
    "g/mL", //5
    "pmol/mL", //6
    "nmol/mL", //7
    "umol/mL", //8
    "mmol/mL", //9
    "mol/mL", //10
    "pIU/mL", //11
    "nIU/mL", //12
    "uIU/mL", //13
    "mIU/mL", //14
    "IU/mL",  //15
    "%",      //16
    "‰",    //17
    "ng/L",   //18
    "ug/L",   //19
    "mg/L",   //20
    "g/L",    //21
};
#define QRJSON_ERR_FLAG             -1
#define QRJSON_ERR_SET(err, val)    \
if(err){\
   *err = val;\
}\

/**
* 获取试剂卡解析版本信息
* @return 版本字符串
 */
char* QrJsonGetVersion(void)
{
  char* version = QRJSON_VERSION;
  return version;
}

/**
* 转换整形数据（10进制）
* @param data          数据地址
* @param len           数据长度
* @param err           错误标志位
* @return 整形数据
 */
static int QRJson_ConvertInt(char* data, uint32_t len, int* err)
{
  if (data && len && len < QRJSON_TEMP_BUFSIZE)
  {
    char temp[QRJSON_TEMP_BUFSIZE] = "";
    memcpy(temp, data, len);
    return (atoi(temp));
  }
  QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
  return 0;
}

/**
* 转换整形数据（16进制）
* @param data          数据地址
* @param len           数据长度
* @param err           错误标志位
* @return 整形数据
 */
static int QRJson_ConvertHex(char* data, uint32_t len, int* err)
{
  if (data && len && len < QRJSON_TEMP_BUFSIZE)
  {
    char temp[QRJSON_TEMP_BUFSIZE] = "";
    memcpy(temp, data, len);
    int value = 0;
    sscanf(temp, "%x", &value);
    return value;
  }
  QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
  return 0;
}



/**
* 浮点数(字符)小数位数处理
* @param buf           缓存地址
* @param decimal       小数位数
* @param err           错误标志位
* @return 0成功/其他失败
 */
static int QRJson_DecimalProess(char* buf, int decimal, int* err)
{

  char tmp[QRJSON_TEXT_BUFSIZE] = "";
  memset(tmp, 0, QRJSON_TEMP_BUFSIZE);
  memcpy(tmp, buf, strlen(buf));
  if (tmp && decimal >= 0)
  {
    int len = strlen(tmp);
    int pos = -1;
    int fullzeroflag = 0;

    // 先对buf做特殊处理，buf为全零的时候，约束??0
    for (uint32_t idx = 0; idx < len; idx++)
    {
      if (*(tmp + idx) != '0')
      {
        fullzeroflag = 1;
        break;
      }
    }

    if (0 == fullzeroflag)
    {
      *(tmp + 1) = '\0'; // 全零结束符重??
      len = strlen(tmp);
    }

    // 找出小数点的位数
    for (uint32_t idx = 0; idx < len; idx++)
    {
      if (*(tmp + idx) == '.')
      {
        pos = idx;
        break;
      }
    }

    // 没有小数位
    if (pos == -1)
    {
      if (decimal == 0)
      {
        // 保留位数??0，没有小数点，不进行操作
        memcpy(buf, tmp, strlen(tmp) + 1);
        return 0;
      }
      else
      {
        // 没有小数点，保留小数为非零，后面直接补零
        *(tmp + len) = '.';
        for (int j = 1; j <= decimal; j++)
        {
          *(tmp + len + j) = '0';
        }
        *(tmp + len + decimal + 1) = '\0';  // len的位置是结束??
      }
      memcpy(buf, tmp, strlen(tmp) + 1);
      return 0;
    }

    // 有小数点，但是保留位数不0，去掉小数点位置
    if (decimal == 0)
    {
      *(tmp + pos) = '\0';  // 小数点的位置是结束符??
      memcpy(buf, tmp, strlen(tmp) + 1);
      return 0;
    }

    // 求解现有的保留位数
    int actual_decimal = len - (pos + 1);
    int dis_decimal = 0;

    // 实际位数处理
    if (decimal > actual_decimal)  // 不足，末尾增??0
    {
      dis_decimal = decimal - actual_decimal;
      for (int j = 0; j < dis_decimal; j++)
      {
        *(tmp + len + j) = '0';
      }
      *(tmp + len + dis_decimal) = '\0';  // len的位置是结束??
      memcpy(buf, tmp, strlen(tmp) + 1);
      return 0;
    }

    if (actual_decimal > decimal)  // ??,多余
    {
      dis_decimal = actual_decimal - decimal;
      *(tmp + (len - dis_decimal)) = '\0';
      memcpy(buf, tmp, strlen(tmp) + 1);
      return 0;
    }
  }
  QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
  return -1;
}

/**
* 转换正浮点数(字符串)
* @param buf           缓存地址
* @param size          缓存大小
* @param data          数据地址
* @param len           数据长度
* @param err           错误标志位
* @return 0成功/其他失败
 */
static int QRJson_ConvertPositiveFloat(char* buf, uint32_t size, char* data, uint32_t len, int decimal, int* err)
{
  if (buf && size && data && len > 2 && len < QRJSON_TEMP_BUFSIZE)
  {
    char temp[QRJSON_TEMP_BUFSIZE] = "";
    memcpy(temp, &data[0], 1);
    int sign = atoi(temp);
    memset(temp, 0, QRJSON_TEMP_BUFSIZE);
    memcpy(temp, &data[1], 1);
    int power = atoi(temp);
    uint32_t oft = 0;
    for (uint32_t idx = 0; idx < QRJSON_TEMP_BUFSIZE; idx++)
    {
      if (sign == QRJSON_NEGATIVE_SIGN)
      {
        if (oft >= len - 2)
        {
          break;
        }
        if (power >= len - 2)
        {
          if (idx == 0)
          {
            buf[idx] = '0';
            continue;
          }
          else if (idx == 1)
          {
            buf[idx] = '.';
            continue;
          }
          if (power > len - 2)
          {
            buf[idx] = '0';
            power--;
            continue;
          }
        }
        else if (oft + power == len - 2)
        {
          power = 0;
          buf[idx] = '.';
          continue;
        }
        buf[idx] = data[2 + oft];
      }
      else
      {
        if (oft < len - 2)
        {
          buf[idx] = data[2 + oft];
        }
        else if (oft < len - 2 + power)
        {
          buf[idx] = '0';
        }
        else break;
      }
      oft++;
    }

    if (decimal >= 0)
    {
      char tmp[QRJSON_TEMP_BUFSIZE] = "";
      memcpy(tmp, buf, strlen(buf));
      QRJson_DecimalProess(tmp, decimal, err);
      memcpy(buf, tmp, strlen(tmp) + 1);
    }
    return 0;
  }
  QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
  return -1;
}


/**
* 转换浮点数据(字符串)
* @param buf           缓存地址
* @param size          缓存大小
* @param data          数据地址
* @param len           数据长度
* @param err           错误标志位
* @return 0成功/其他失败
 */
static int QRJson_ConvertFloat(char* buf, uint32_t size, char* data, uint32_t len, int* err)
{
  if (len < 1)
  {
    QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
    return -1;
  }
  char temp[QRJSON_TEMP_BUFSIZE] = "";
  if (QRJson_ConvertPositiveFloat(temp, QRJSON_TEMP_BUFSIZE, &data[1], len - 1, -1, err))
  {
    QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
    return -2;
  }
  switch (data[0])
  {
  case '2':
    buf[0] = '-';
    buf++;
  default:
    snprintf(buf, size, "%s", temp);
    break;
  }
  return 0;
}

/**
* 转换字符数据
* @param data          数据地址
* @param len           数据长度
* @param err           错误标志位
* @return 字符数据
 */
static char QRJson_ConvertChar(char* data, uint32_t len, int* err)
{
  int idx = QRJson_ConvertInt(data, len, err);
  if (idx > 0 && idx <= sizeof(qrjson_char_list) / sizeof(char))
  {
    return qrjson_char_list[idx - 1];
  }
  QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
  return 0;
}

/**
* 转换生产日期
* @param data          数据地址
* @param len           数据长度
* @param err           错误标志位
* @return 生产日期编号
 */
static int QRJson_ConvertExpDate(char* data, uint32_t len, int* err)
{
  int idx = QRJson_ConvertInt(data, len, err);
  if (idx > 0 && idx <= sizeof(qrjson_expdate_list) / sizeof(int))
  {
    return qrjson_expdate_list[idx - 1];
  }
  QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
  return 0;
}


/**
* 转换等待时间
* @param data          数据地址
* @param len           数据长度
* @param err           错误标志位
* @return 等待时间(分钟)
 */
static int QRJson_ConvertWaitTime(char* data, uint32_t len, int* err)
{
  int idx = QRJson_ConvertInt(data, len, err);
  if (idx > 0 && idx <= sizeof(qrjson_waittime_list) / sizeof(int))
  {
    return qrjson_waittime_list[idx - 1];
  }
  QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
  return 0;
}

/**
* 转换线起点
* @param data          数据地址
* @param len           数据长度
* @param line_id       线序号
* @param err           错误标志位
* @return 线起点
 */
static int QRJson_ConvertLineStart(char* data, uint32_t len, int* err)
{
  int idx = QRJson_ConvertInt(data, len, err);
  // if(idx > 0 && idx <= sizeof(qrjson_line_start_list)/sizeof(int))
  // {
  //     return qrjson_line_start_list[idx - 1];
  // }
  if (idx > 0 && idx < 36)
  {
    return 10 * idx;
  }
  QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
  return 0;
}

/**
* 转换线相对量
* @param data          数据地址
* @param len           数据长度
* @param line_id       线序号
* @param err           错误标志位
* @return 线终点
 */
static int QRJson_ConvertTLineDistance(char* data, uint32_t len, int* err)
{
  int idx = QRJson_ConvertInt(data, len, err);
  if (idx > 0 && idx <= sizeof(qrjson_Tline_offset_list) / sizeof(int))
  {
    return qrjson_Tline_offset_list[idx - 1];
  }
  QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
  return 0;
}

/**
* 转换线相对量
* @param data          数据地址
* @param len           数据长度
* @param line_id       线序号
* @param err           错误标志位
* @return 线终点
 */
static int QRJson_ConvertCLineDistance(char* data, uint32_t len, int* err)
{
  int idx = QRJson_ConvertInt(data, len, err);
  if (idx > 0 && idx <= sizeof(qrjson_Cline_offset_list) / sizeof(int))
  {
    return qrjson_Cline_offset_list[idx-1];
  }
  QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
  return 0;
}

/**
* 转换线宽度
* @param data          数据地址
* @param len           数据长度
* @param err           错误标志位
* @return 线宽度
 */
static int QRJson_ConvertLineWidth(char* data, uint32_t len, int* err)
{
  int idx = QRJson_ConvertInt(data, len, err);
  if (idx > 0 && idx <= sizeof(qrjson_line_width_list) / sizeof(int))
  {
    return qrjson_line_width_list[idx - 1];
  }
  QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
  return 0;
}

/**
* 转换C线失效值
* @param data          数据地址
* @param len           数据长度
* @param err           错误标志位
* @return C线失效值
 */
static int QRJson_ConvertLineInvalid(char* data, uint32_t len, int* err)
{
  int idx = QRJson_ConvertInt(data, len, err);
  if (idx >= 0 && idx < sizeof(QRJson_line_invalid_list) / sizeof(int))
  {
    return QRJson_line_invalid_list[idx];
  }
  QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
  return 0;
}


/**
* 转换通道单位
* @param data          数据地址
* @param len           数据长度
* @param err           错误标志位
* @return 通道单位
 */
static const char* QRJson_ConvertUnit(char* data, uint32_t len, int* err)
{
  int idx = QRJson_ConvertInt(data, len, err);
  if (idx >= 0 && idx < sizeof(qrjson_unit_list) / sizeof(const char*))
  {
    return qrjson_unit_list[idx];
  }
  QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
  return NULL;
}

/**
* 匹配字符串
* @param buf           缓存地址
* @param size          缓存大小
* @param str           目标字符串
* @param len           字符串长度
* @param reverse       反向标志位（0低地址向高寻找/1高地址向低寻找）
* @param err           错误标志位
* @return 通道单位
 */
static char* QRJson_StrStr(char* buf, uint32_t size, char* str, uint32_t len, uint8_t reverse)
{
  if (buf == NULL || str == NULL || !size || !len)
  {
    return NULL;
  }
  if (reverse)
  {
    for (int idx1 = size - 1; idx1 >= 0; idx1--)
    {
      for (int idx2 = len - 1; idx2 >= 0; idx2--)
      {
        if (buf[idx1 + idx2] != str[idx2])
        {
          break;
        }
        else if (idx2 == 0)
        {
          return buf + idx1;
        }
      }
    }
  }
  else
  {
    for (int idx1 = 0; idx1 < size; idx1++)
    {
      for (int idx2 = 0; idx2 < len; idx2++)
      {
        if (buf[idx1 + idx2] != str[idx2])
        {
          break;
        }
        else if (idx2 + 1 == len)
        {
          return buf + idx1;
        }
      }
    }
  }
  return NULL;
}

/**
* 偏移分段方程
* @param equation      分段方程指针（此指针指向的实例将引用最后一段偏移中的数据地址，可以为NULL）
* @param next          当前地址
* @param boundary      边界地址
* @param seg_oft       偏移段数
* @param frist_sign    首段标志位（非0时第一段偏移将会计算校准方程低门限空间占用）
* @param err           错误标志位
* @return 进行偏移后的地址
 */
static char* QRJson_NextEquation(QRJsonEquation_t* equation, char* next, char* boundary, uint32_t seg_oft, uint8_t frist_sign, int* err)
{
  if (next == NULL)
  {
    QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
    return NULL;
  }
  for (uint32_t idx = 0; idx < seg_oft; idx++)
  {
    char* type = next;
    next += QRJSON_MAX_EQUATION_TYPE;
    char* param_cnt = next;
    next += QRJSON_MAX_EQUATION_PARAM_NUM;
    if (boundary && next > boundary)
    {
      QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
      return NULL;
    }
    char* param = next;
    uint32_t param_num = QRJson_ConvertInt(param_cnt, QRJSON_MAX_EQUATION_PARAM_NUM, err);
    next += param_num * QRJSON_MAX_EQUATION_PARAM_LEN;
    char* gate_low = NULL;
    if (!idx && frist_sign)
    {
      gate_low = next;
      next += QRJSON_MAX_EQUATION_GATE;
    }
    char* gate_high = next;
    next += QRJSON_MAX_EQUATION_GATE;
    if (boundary && next > boundary)
    {
      QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
      return NULL;
    }
    if (idx + 1 == seg_oft && equation)
    {
      equation->type = type;
      equation->param_cnt = param_cnt;
      equation->param = param;
      equation->gate_low = gate_low;
      equation->gate_high = gate_high;
      equation->next = next;
    }
  }
  return next;
}

/**
* 偏移定性与定量计算信息
* @param calcu         定性与定量计算信息指针（该指针指向的实例将引用最后一段偏移中的数据地址，可以为NULL）
* @param next          当前地址
* @param boundary      边界地址
* @param chl_oft       偏移段数
* @param err           错误标志位
* @return 进行偏移后的地址
 */
static char* QRJson_NextCalcu(QRJsonCalcu_t* calcu, char* next, char* boundary, uint32_t chl_oft, int* err)
{
  if (next == NULL)
  {
    return NULL;
  }
  for (uint32_t idx1 = 0; idx1 < chl_oft; idx1++)
  {
    char* res_mode = next;
    if (boundary && next > boundary)
    {
      QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
      return NULL;
    }
    uint32_t res_num = QRJson_ConvertInt(res_mode, QRJSON_MAX_CALCU_RES_MODE, err);
    next += QRJSON_MAX_CALCU_RES_MODE;
    char* seg_cnt = NULL;
    if (res_num != 2)
    {
      seg_cnt = next;
      if (boundary && next > boundary)
      {
        QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
        return NULL;
      }
      uint32_t seg_num = QRJson_ConvertInt(seg_cnt, QRJSON_MAX_EQUATION_SEGMENT, err);
      next += QRJSON_MAX_EQUATION_SEGMENT;
      for (uint32_t idx2 = 0; idx2 < seg_num; idx2++)
      {
        QRJsonEquation_t* equation = NULL;
        if (idx1 + 1 == chl_oft && idx2 < QRJSON_MAX_BUF_EQUATION && calcu)
        {
          equation = &calcu->equation[idx2];
        }
        uint8_t first_sign = 0;
        if (!idx2)
        {
          first_sign = 1;
        }
        next = QRJson_NextEquation(equation, next, boundary, 1, first_sign, err);
        if (next == NULL)
        {
          QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
          return NULL;
        }
      }
    }
    if (res_num != 1)
    {
      if (idx1 + 1 == chl_oft && calcu)
      {
        calcu->gate = (QRJsonResultGate_t*)next;
      }
      next += sizeof(QRJsonResultGate_t);
    }
    if (boundary && next > boundary)
    {
      QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
      return NULL;
    }
    if (idx1 + 1 == chl_oft && calcu)
    {
      calcu->res_mode = res_mode;
      calcu->seg_cnt = seg_cnt;
      calcu->next = next;
    }
  }
  return next;
}

/**
* 检索试剂卡信息
* @param card          试剂卡信息指针（此指针指向的实例将引用检索到的数据地址）
* @param group         机型系列
* @param mask          型号掩码
* @param buf           缓存地址
* @param size          缓存大小
* @param err           错误标志位
* @return 0成功/其他失败
 */
int QRJson_SearchCard(QRJsonCard_t* card, uint32_t group, uint32_t mask, char* buf, uint32_t size, int* err)
{
  if (card == NULL || buf == NULL || !size)
  {
    QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
    return -1;
  }
  for (uint32_t idx = 0; idx < size; idx++)
  {
    /* 试剂卡文本匹配 */
    char* qrcode_sign = QRJson_StrStr(buf + idx, size - idx, QRJSON_QRCODE_SIGN, strlen(QRJSON_QRCODE_SIGN), 0);
    if (qrcode_sign == NULL)
    {
      QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
      return -2;
    }
    /* 协议号检查 */
    char* protocol_sign = QRJson_StrStr(qrcode_sign, (uint32_t)(buf + size - qrcode_sign), QRJSON_PROTOCOL_SIGN, strlen(QRJSON_PROTOCOL_SIGN), 0);
    if (protocol_sign == NULL)
    {
      QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
      return -3;
    }
    char* protocol_num = protocol_sign + strlen(QRJSON_PROTOCOL_SIGN);
    uint32_t protocol_len = 0;
    for (; protocol_len < (uint32_t)(buf + size - protocol_num); protocol_len++)
    {
      if (qrcode_sign[protocol_len] == 0 || qrcode_sign[protocol_len] == QRJSON_PROTOCOL_INTERVAL_SIGN)
      {
        break;
      }
    }
    uint32_t protocol_val = QRJson_ConvertInt(protocol_num, protocol_len, err);
    if (protocol_val != QRJSON_PROTOCOL_NUM)
    {
      idx = (uint32_t)(qrcode_sign - buf);
      continue;
    }
    /* 网址字段文本匹配 */
    char* website_sign = QRJson_StrStr(buf, (uint32_t)(qrcode_sign - buf), QRJSON_WEBSITE_SIGN, strlen(QRJSON_WEBSITE_SIGN), 1);
    if (website_sign == NULL)
    {
      QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
      return -4;
    }
#if(QRJSON_CRC_ENABLE)
    /* CRC校验码检查 */
    char* check_sign = QRJson_StrStr(qrcode_sign, (uint32_t)(buf + size - qrcode_sign), QRJSON_CHECK_SIGN, strlen(QRJSON_CHECK_SIGN), 0);
    if (check_sign == NULL)
    {
      QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
      return -5;
    }
    char* check_val_char = check_sign + strlen(QRJSON_CHECK_SIGN);
    uint16_t check_val_int = QRJson_ConvertHex(check_val_char, QRJSON_MAX_CHECK_VAL, err);
    uint16_t check_val_crc = CRC16_MODBUS((unsigned char*)website_sign, (uint16_t)(check_sign - website_sign - 1));
    if (check_val_int != check_val_crc)
    {
      idx = (uint32_t)(qrcode_sign - buf);
      continue;
    }
    char* boundary = check_sign;
#else /* QRJSON_CRC_ENABLE */
    char* boundary = buf + size;
#endif /* QRJSON_CRC_ENABLE */
    /* 参数字段文本匹配 */
    char* param_sign = QRJson_StrStr(qrcode_sign, (uint32_t)(boundary - qrcode_sign), QRJSON_PARAM_SIGN, strlen(QRJSON_PARAM_SIGN), 0);
    if (param_sign == NULL)
    {
      QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
      return -6;
    }
    QRJsonStaticSpecial_t* static_special = NULL;
    for (uint32_t idx2 = 0; param_sign + idx2 < boundary; idx2++)
    {
      if (param_sign[idx2] == QRJSON_PARAM_INTERVAL_SIGN)
      {
        static_special = (QRJsonStaticSpecial_t*)&param_sign[idx2 + 1];
        if ((long)static_special + sizeof(QRJsonStaticSpecial_t) > (long)boundary)
        {
          QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
          return -7;
        }
        int a_temp =0;
        a_temp = QRJson_ConvertInt(static_special->group, QRJSON_MAX_DEV_GROUP, err);
        /* 机型系列检查 */
        if (group != a_temp)
        {
          static_special = NULL;
          continue;
        }
        int b_temp =0;
        b_temp =QRJson_ConvertHex(static_special->mask, QRJSON_MAX_DEV_MASK, err);
        /* 型号掩码检查 */
        if (!(mask & QRJson_ConvertHex(static_special->mask, QRJSON_MAX_DEV_MASK, err)))
        {
          QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
          return -8;
        }
        break;
      }
    }
    if (static_special == NULL)
    {
      idx = (uint32_t)(qrcode_sign - buf);
      continue;
    }
    /* 建立数据地址引用 */
    memset(card, 0, sizeof(QRJsonCard_t));
    card->protocal_num = protocol_val;
    card->website = website_sign;
    card->check_crc = check_val_char;
    card->boundary = boundary;
    card->base.static_area = (QRJsonStaticBase_t*)(param_sign + strlen(QRJSON_PARAM_SIGN));
    if ((long)card->base.static_area + sizeof(QRJsonStaticBase_t) > (long)boundary)
    {
      QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
      return -9;
    }
    uint32_t chl_num = QRJson_ConvertInt(card->base.static_area->chl_cnt, QRJSON_MAX_CARD_CHL_CNT, err);
    for (uint32_t oft = 0; oft < chl_num; oft++)
    {
      if (oft < QRJSON_MAX_BUF_CHL)
      {
        card->base.chl_list[oft] = (QRJsonChl_t*)((long)card->base.static_area + sizeof(QRJsonStaticBase_t) + oft * sizeof(QRJsonChl_t));
      }
    }
    card->special.static_area = static_special;
    char* next = (char*)((long)card->special.static_area + sizeof(QRJsonStaticSpecial_t));
    uint32_t line_num = QRJson_ConvertInt(card->base.static_area->tline_cnt, QRJSON_MAX_CARD_LINE_CNT, err);
    for (uint32_t oft = 0; oft < line_num + 1; oft++)
    {
      if (oft < QRJSON_MAX_BUF_LINE)
      {
        card->special.line_list[oft] = (QRJsonLine_t*)next;
      }
      next += sizeof(QRJsonLine_t);
    }
    for (uint32_t oft = 0; oft < chl_num; oft++)
    {
      if (oft < QRJSON_MAX_BUF_CALCU)
      {
        next = QRJson_NextCalcu(&(card->special.calcu_list[oft]), next, boundary, 1, err);
      }
      else
      {
        next = QRJson_NextCalcu(NULL, next, boundary, 1, err);
      }
      if (next == NULL)
      {
        QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
        return -10;
      }
    }
    return 0;
  }
  return -11;
}

/**
* 检索线信息
* @param card          试剂卡信息指针
* @param line_idx      线序针
* @param err           错误标志位
* @return 线信息指针/NULL
 */
QRJsonLine_t* QRJson_SearchLine(QRJsonCard_t* card, uint32_t line_idx, int* err)
{
  if (card == NULL || card->base.static_area == NULL || card->special.line_list[0] == NULL)
  {
    QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
    return NULL;
  }
  uint32_t line_num = QRJson_ConvertInt(card->base.static_area->tline_cnt, QRJSON_MAX_CARD_LINE_CNT, err);
  if (line_idx < line_num + 1)
  {
    if (line_idx < QRJSON_MAX_BUF_LINE)
    {
      return card->special.line_list[line_idx];
    }
    return card->special.line_list[0] + line_idx * sizeof(QRJsonLine_t);
  }
  QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
  return NULL;


}

/**
* 检索通道信息
* @param card          试剂卡信息指针
* @param chl_idx       通道序号
* @param err           错误标志位
* @return 通道信息指针/NULL
 */
QRJsonChl_t* QRJson_SearchChl(QRJsonCard_t* card, uint32_t chl_idx, int* err)
{
  if (card == NULL || card->base.static_area == NULL || card->base.chl_list[0] == NULL)
  {
    QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
    return NULL;
  }
  uint32_t chl_num = QRJson_ConvertInt(card->base.static_area->chl_cnt, QRJSON_MAX_CARD_CHL_CNT, err);
  if (chl_idx < chl_num)
  {
    if (chl_idx < QRJSON_MAX_BUF_CHL)
    {
      return card->base.chl_list[chl_idx];
    }
    return card->base.chl_list[0] + chl_idx * sizeof(QRJsonChl_t);
  }
  QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
  return NULL;
}

/**
* 检索定性与定量计算信息
* @param calcu_ins     定性与定量计算信息指针（此指针指向的实例将引用检索到的数据地址）
* @param card          试剂卡信息指针
* @param chl_idx       通道序号
* @param err           错误标志位
* @return 通道信息指针/NULL
 */
static int QRJson_SearchCalcu(QRJsonCalcu_t* calcu_ins, QRJsonCard_t* card, uint32_t chl_idx, int* err)
{
  if (calcu_ins == NULL || card == NULL || card->base.static_area == NULL || card->special.calcu_list[0].res_mode == NULL)
  {
    QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
    return -1;
  }
  uint32_t chl_num = QRJson_ConvertInt(card->base.static_area->chl_cnt, QRJSON_MAX_CARD_CHL_CNT, err);
  if (chl_idx < chl_num)
  {
    if (chl_idx < QRJSON_MAX_BUF_CALCU)
    {
      memcpy(calcu_ins, &(card->special.calcu_list[chl_idx]), sizeof(QRJsonCalcu_t));
      return 0;
    }
    if (NULL != QRJson_NextCalcu(calcu_ins, card->special.calcu_list[QRJSON_MAX_BUF_CALCU - 1].next, card->boundary, chl_idx - QRJSON_MAX_BUF_CALCU + 1, err))
    {
      return 0;
    }
  }
  QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
  return -2;
}

/**
* 检索分段方程
* @param equation_ins  分段方程指针（此指针指向的实例将引用检索到的数据地址）
* @param card          试剂卡信息指针
* @param chl_idx       通道索引
* @param seg_idx       分段索引
* @param err           错误标志位
* @return 校准方程信息指针/NULL
 */
static int QRJson_SearchEquation(QRJsonEquation_t* equation_ins, QRJsonCard_t* card, uint32_t chl_idx, uint32_t seg_idx, int* err)
{
  if (card == NULL)
  {
    QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
    return -1;
  }
  QRJsonCalcu_t calcu_ins = { 0 };
  if (QRJson_SearchCalcu(&calcu_ins, card, chl_idx, err))
  {
    QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
    return -2;
  }
  uint32_t seg_num = QRJson_ConvertInt(calcu_ins.seg_cnt, QRJSON_MAX_EQUATION_SEGMENT, err);
  if (seg_idx < seg_num)
  {

    if (seg_idx < QRJSON_MAX_BUF_EQUATION)
    {
      memcpy(equation_ins, &(calcu_ins.equation[seg_idx]), sizeof(QRJsonEquation_t));
      return 0;
    }
    if (chl_idx == card->special.last_chl && seg_idx == card->special.last_seg)
    {
      memcpy(equation_ins, &(card->special.last_equation), sizeof(QRJsonEquation_t));
      return 0;
    }
    if (NULL != QRJson_NextEquation(equation_ins, calcu_ins.equation[QRJSON_MAX_BUF_EQUATION - 1].next, card->boundary, seg_idx - QRJSON_MAX_BUF_EQUATION + 1, 0, err))
    {
      memcpy(&(card->special.last_equation), equation_ins, sizeof(QRJsonEquation_t));
      card->special.last_chl = chl_idx;
      card->special.last_seg = seg_idx;
      return 0;
    }
  }
  return -3;
}

/**
* 检索结果门限信息（定性）
* @param card          试剂卡信息指针
* @param chl_idx       通道序号
* @param err           错误标志位
* @return 结果门限信息指针/NULL
 */
QRJsonResultGate_t* QRJson_SearchResultGate(QRJsonCard_t* card, uint32_t chl_idx, int* err)
{
  if (card == NULL)
  {
    QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
    return NULL;
  }
  QRJsonCalcu_t calcu_ins = { 0 };
  if (QRJson_SearchCalcu(&calcu_ins, card, chl_idx, err))
  {
    QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
    return NULL;
  }
  return calcu_ins.gate;
}

/**
* 获取机型系列
* @param card          试剂卡信息指针
* @param err           错误标志位
* @return 系列编号
 */
int QRJson_GetDevGroup(QRJsonCard_t* card, int* err)
{
  if (card == NULL || card->special.static_area == NULL)
  {
    QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
    return 0;
  }
  return QRJson_ConvertInt(card->special.static_area->group, QRJSON_MAX_DEV_GROUP, err);
}


/**
* 获取子型号掩码
* @param card          试剂卡信息指针
* @param err           错误标志位
* @return 机型掩码
 */
int QRJson_GetDevMask(QRJsonCard_t* card, int* err)
{
  if (card == NULL || card->special.static_area == NULL)
  {
    QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
    return 0;
  }
  return QRJson_ConvertInt(card->special.static_area->mask, QRJSON_MAX_DEV_MASK, err);
}

/**
* 获取盒号
* @param card          试剂卡信息指针
* @param err           错误标志位
* @return 盒号
 */
int QRJson_GetBoxNumber(QRJsonCard_t* card, int* err)
{
  if (card == NULL || card->base.static_area == NULL)
  {
    QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
    return 0;
  }
  return QRJson_ConvertInt(card->base.static_area->box_number, QRJSON_MAX_CARD_BOX_NUM, err);
}

/**
* 获取盒装规格（测试数）
* @param card          试剂卡信息指针
* @param err           错误标志位
* @return 测试数
 */
int QRJson_GetTestNumber(QRJsonCard_t* card, int* err)
{
  if (card == NULL || card->base.static_area == NULL)
  {
    QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
    return 0;
  }
  int idx = QRJson_ConvertInt(card->base.static_area->test_number, QRJSON_MAX_CARD_TEST_NUM, err);
  if (idx >= 0 && idx < sizeof(qrjson_testcnt_list) / sizeof(int))
  {
    return qrjson_testcnt_list[idx];
  }
  QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
  return 0;
}

/**
* 获取批号
* @param buf           缓存地址
* @param size          缓存大小
* @param card          试剂卡信息指针
* @param err           错误标志位
* @return 0成功/其他失败
 */
int QRJson_GetBatchNunber(char* buf, uint32_t size, QRJsonCard_t* card, int* err)
{
  if (buf == NULL || size <= QRJSON_MAX_CARD_BATCH_NUM - 3 || card == NULL || card->base.static_area == NULL || card->base.static_area->batch_number == NULL)
  {
    QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
    return -1;
  }
  uint32_t cnt = 0;
  uint32_t idx = 0;
  for (; idx < QRJSON_MAX_CARD_BATCH_EN_CODE / 2; idx++)
  {
    char temp = QRJson_ConvertChar(&card->base.static_area->batch_number[idx * 2], 2, err);
    if (temp)
    {
      buf[cnt++] = temp;
    }
  }
  memcpy(&buf[cnt], &card->base.static_area->batch_number[idx * 2], QRJSON_MAX_CARD_BATCH_NUM - idx * 2);
  return 0;
}

/**
* 获取生产日期
* @param buf           缓存地址
* @param size          缓存大小
* @param card          试剂卡信息指针
* @param err           错误标志位
* @return 0成功/其他失败
 */
int QRJson_GetProDate(char* buf, uint32_t size, QRJsonCard_t* card, int* err)
{
  if (buf == NULL || !size || size < 10 || card == NULL || card->base.static_area == NULL || card->base.static_area->batch_number == NULL)
  {
    QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
    return -1;
  }
  buf[0] = '2';
  buf[1] = '0';
  buf[4] = '-';
  buf[7] = '-';
  memcpy(&buf[2], &card->base.static_area->batch_number[QRJSON_MAX_CARD_BATCH_EN_CODE], 2);
  memcpy(&buf[5], &card->base.static_area->batch_number[QRJSON_MAX_CARD_BATCH_EN_CODE + 2], 2);
  memcpy(&buf[8], &card->base.static_area->batch_number[QRJSON_MAX_CARD_BATCH_EN_CODE + 4], 2);
  return 0;
}

/**
* 获取失效日期
* @param buf           缓存地址
* @param size          缓存大小
* @param card          试剂卡信息指针
* @param err           错误标志位
* @return 0成功/其他失败
 */
int QRJson_GetExpDate(char* buf, uint32_t size, QRJsonCard_t* card, int* err)
{
  if (buf == NULL || !size || size < 10 || card == NULL || card->base.static_area == NULL || card->base.static_area->batch_number == NULL || card->base.static_area->exp_date == NULL)
  {
    QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
    return -1;
  }
  int year = QRJson_ConvertInt(&card->base.static_area->batch_number[QRJSON_MAX_CARD_BATCH_EN_CODE], 2, err);
  int month = QRJson_ConvertInt(&card->base.static_area->batch_number[QRJSON_MAX_CARD_BATCH_EN_CODE + 2], 2, err);
  int oft = QRJson_ConvertExpDate(card->base.static_area->exp_date, QRJSON_MAX_CARD_EXP_DATE, err);
  month += oft % 12;
  year += oft / 12 + month / 12;
  month = month % 12;
  if (!month)
  {
    month = 12;
    if (year)
    {
      year--;
    }
  }
  buf[0] = '2';
  buf[1] = '0';
  buf[2] = 0x30 + year / 10;
  buf[3] = 0x30 + year % 10;
  buf[4] = '-';
  buf[5] = 0x30 + month / 10;
  buf[6] = 0x30 + month % 10;
  buf[7] = '-';
  memcpy(&buf[8], &card->base.static_area->batch_number[QRJSON_MAX_CARD_BATCH_EN_CODE + 4], 2);
  return 0;
}

/**
* 获取信号增益
* @param card          试剂卡信息指针
* @param err           错误标志位
* @return 增益编号
 */
int QRJson_GetSignGain(QRJsonCard_t* card, int* err)
{
  if (card == NULL || card->base.static_area == NULL)
  {
    QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
    return 0;
  }
  return QRJson_ConvertInt(card->base.static_area->sign_gain, QRJSON_MAX_CARD_SIGN_GAIN, err);
}

/**
* 获取T线数量
* @param card          试剂卡信息指针
* @param err           错误标志位
* @return T线数量
 */
int QRJson_GetLineCnt(QRJsonCard_t* card, int* err)
{
  if (card == NULL || card->base.static_area == NULL)
  {
    QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
    return 0;
  }
  return QRJson_ConvertInt(card->base.static_area->tline_cnt, QRJSON_MAX_CARD_LINE_CNT, err);
}

/**
* 获取通道数量
* @param card          试剂卡信息指针
* @param err           错误标志位
* @return 通道数量
 */
int QRJson_GetChlCnt(QRJsonCard_t* card, int* err)
{
  if (card == NULL || card->base.static_area == NULL)
  {
    QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
    return 0;
  }
  return QRJson_ConvertInt(card->base.static_area->chl_cnt, QRJSON_MAX_CARD_CHL_CNT, err);
}

/**
* 获取等待时间
* @param card          试剂卡信息指针
* @param err           错误标志位
* @return 等待时间(分钟)
 */
int QRJson_GetWaitTime(QRJsonCard_t* card, int* err)
{
  if (card == NULL || card->base.static_area == NULL)
  {
    QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
    return 0;
  }
  return QRJson_ConvertWaitTime(card->base.static_area->wait_time, QRJSON_MAX_CARD_WAIT_TIME, err);
}

/**
* 获取方法学ID
* @param card          试剂卡信息指针
* @param err           错误标志位
* @return 方法学ID
 */
int QRJson_GetMethodID(QRJsonCard_t* card, int* err)
{
  if (card == NULL || card->base.static_area == NULL)
  {
    QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
    return 0;
  }
  return QRJson_ConvertInt(card->base.static_area->method_id, QRJSON_MAX_CARD_METHOD_ID, err);
}

/**
* 获取项目ID
* @param card          试剂卡信息指针
* @param err           错误标志位
* @return 项目ID
 */
int QRJson_GetProjID(QRJsonCard_t* card, int* err)
{
  if (card == NULL || card->base.static_area == NULL)
  {
    QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
    return 0;
  }
  return QRJson_ConvertInt(card->base.static_area->proj_id, QRJSON_MAX_CARD_PROJ_ID, err);
}

/**
* 获取项目名称
* @param buf           缓存地址
* @param size          缓存大小
* @param card          试剂卡信息指针
* @param err           错误标志位
* @return 0成功/其他失败
 */
int QRJson_GetProjName(char* buf, uint32_t size, QRJsonCard_t* card, int* err)
{
  if (buf == NULL || !size || card == NULL || card->base.chl_list[0] == NULL)
  {
    QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
    return -1;
  }
  char text[QRJSON_TEXT_BUFSIZE];
  uint32_t oft = 0;
  for (uint32_t idx = 0;; idx++)
  {
    QRJsonChl_t* chl = QRJson_SearchChl(card, idx, NULL);
    if (chl == NULL)
    {
      break;
    }
    memset(text, 0, QRJSON_TEXT_BUFSIZE);
    if (QRJson_GetChlName(text, QRJSON_TEXT_BUFSIZE, chl, err))
    {
      QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
      return -2;
    }
    if (!idx)
    {
      oft += snprintf(buf + oft, size - oft, "%s", text);
    }
    else
    {
      oft += snprintf(buf + oft, size - oft, "/%s", text);
    }
  }
  return 0;
}

/**
* 获取C线位置
* @param card          试剂卡信息指针
* @param err           错误标志位
* @return C线位置索引
 */
int QRJson_GetCLineIndex(QRJsonCard_t* card, int* err)
{
  if (card == NULL || card->base.static_area == NULL)
  {
    QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
    return 0;
  }
  return QRJson_ConvertInt(card->base.static_area->cline_idx, QRJSON_MAX_LINE_CINDEX, err);
}

/**
* 获取C线起点
* @param card          剂卡信息指针
* @param err           错误标志位
* @return C线起点
 */
int QRJson_GetCLineStart(QRJsonCard_t* card, int* err)
{
  if (card == NULL || card->special.static_area == NULL)
  {
    QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
    return 0;
  }
  return QRJson_ConvertLineStart(card->special.static_area->cline_start, QRJSON_MAX_CLINE_START, err);
}

/**
* 获取C线失效值
* @param card          试剂卡信息指针
* @param err           错误标志位
* @return C线失效值
 */
int QRJson_GetCLineInvalid(QRJsonCard_t* card, int* err)
{
  if (card == NULL || card->special.static_area == NULL)
  {
    QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
    return 0;
  }
  return QRJson_ConvertLineInvalid(card->special.static_area->cline_invalid, QRJSON_MAX_CLINE_INVALID, err);
}

/**
* 获取T线寻峰半窗口
* @param card          试剂卡信息指针
* @param err           错误标志位
* @return T线寻峰半窗口
 */
int QRJson_GetTLineWindow(QRJsonCard_t* card, int* err)
{
  if (card == NULL || card->special.static_area == NULL)
  {
    QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
    return 0;
  }
  return 2 * QRJson_ConvertInt(card->special.static_area->tline_window, QRJSON_MAX_TLINE_WINDOW, err);
}

/**
* 获取C线偏移量
* @param line          线信息指针
* @param err           错误标志位
* @return 线终点
 */
int QRJson_GetCLineOffset(QRJsonLine_t* line, int* err)
{
  if (line == NULL)
  {
    QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
    return 0;
  }
  return QRJson_ConvertCLineDistance(line->offset, QRJSON_MAX_LINE_OFFSET, err);
}

/**
* 获取T线偏移量
* @param line          线信息指针
* @param err           错误标志位
* @return 线终点
 */
int QRJson_GetTLineOffset(QRJsonLine_t* line, int* err)
{
  if (line == NULL)
  {
    QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
    return 0;
  }
  return QRJson_ConvertTLineDistance(line->offset, QRJSON_MAX_LINE_OFFSET, err);
}

/**
* 获取线宽度
* @param line          线信息指针
* @param err           错误标志位
* @return 线宽度
 */
int QRJson_GetLineWidth(QRJsonLine_t* line, int* err)
{
  if (line == NULL)
  {
    QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
    return 0;
  }
  return QRJson_ConvertLineWidth(line->width, QRJSON_MAX_LINE_WIDTH, err);
}

/**
* 获取通道ID
* @param chl           通道信息指针
* @param err           错误标志位
* @return 通道ID
 */
int QRJson_GetChlID(QRJsonChl_t* chl, int* err)
{
  if (chl == NULL)
  {
    QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
    return 0;
  }
  int val = QRJson_ConvertInt(chl->id, QRJSON_MAX_CHL_ID, err);
  if (!val)
  {
    return 888;
  }
  return val;
}

/**
* 获取通道名称
* @param buf           缓存地址
* @param size          缓存大小
* @param chl           通道信息指针
* @param err           错误标志位
* @return 0成功/其他失败
 */
int QRJson_GetChlName(char* buf, uint32_t size, QRJsonChl_t* chl, int* err)
{
  if (buf == NULL || !size || chl == NULL)
  {
    return -1;
  }
  for (uint32_t idx = 0; idx < size && idx * 2 + 1 < QRJSON_MAX_CHL_NAME; idx++)
  {
    buf[idx] = QRJson_ConvertChar(&chl->name[idx * 2], 2, err);
  }
  return 0;
}

/**
* 获取通道单位
* @param buf           缓存地址
* @param size          缓存大小
* @param chl           通道信息指针
* @param err           错误标志位
* @return 0成功/其他失败
 */
int QRJson_GetChlUnit(char* buf, uint32_t size, QRJsonChl_t* chl, int* err)
{
  if (buf == NULL || !size || chl == NULL)
  {
    return -1;
  }
  const char* text = QRJson_ConvertUnit(chl->unit, QRJSON_MAX_CHL_UNIT, err);
  if (text == NULL)
  {
    return -2;
  }
  snprintf(buf, size, "%s", text);
  return 0;
}

/**
* 获取通道计算模式编号
* @param chl           通道信息指针
* @param oft           偏移字节数
* @param err           错误标志位
* @return 计算模式编号
 */
uint8_t QRJson_GetChlModeNum(QRJsonChl_t* chl, uint32_t oft, int* err)
{
  if (chl == NULL || oft > 1)
  {
    QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
    return -1;
  }
  char temp[QRJSON_TEMP_BUFSIZE] = "";
  memcpy(temp, &chl->mode[oft], 1);
  return atoi(temp);
}

/**
* 获取保留小数位数
* @param chl           通道信息指针
* @param err           错误标志位
* @return 保留小数位数
 */
int QRJson_GetChlDecimal(QRJsonChl_t* chl, int* err)
{
  if (chl == NULL)
  {
    QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
    return 0;
  }
  return QRJson_ConvertInt(chl->decimal, QRJSON_MAX_CHL_DECIMAL, err);
}

/**
* 获取通道高/低限显示值
* @param buf           缓存地址
* @param size          缓存大小
* @param chl           通道信息指针
* @param high_flag     高限标志位(0低限/1高限)
* @param decimal       小数位数
* @param err           错误标志位
* @return 0成功/其他失败
 */
int QRJson_GetChlLimit(char* buf, uint32_t size, QRJsonChl_t* chl, uint8_t high_flag, int decimal, int* err)
{
  if (buf == NULL || !size || chl == NULL)
  {
    QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
    return -1;
  }
  if (high_flag)
  {
    return QRJson_ConvertPositiveFloat(buf, size, chl->high_limit, QRJSON_MAX_CHL_LIMIT, decimal, err);
  }
  return QRJson_ConvertPositiveFloat(buf, size, chl->low_limit, QRJSON_MAX_CHL_LIMIT, decimal, err);
}

/**
* 获取计算结果模式
* @param card          试剂卡信息指针
* @param chl_idx       通道索引
* @param err           错误标志位
* @return 计算结果模式
 */
int QRJson_GetCalcuResultMode(QRJsonCard_t* card, uint32_t chl_idx, int* err)
{
  if (card == NULL)
  {
    QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
    return 0;
  }
  QRJsonCalcu_t calcu_ins = { 0 };
  if (QRJson_SearchCalcu(&calcu_ins, card, chl_idx, err))
  {
    QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
    return 0;
  }
  return QRJson_ConvertInt(calcu_ins.res_mode, QRJSON_MAX_CALCU_RES_MODE, err);
}

/**
* 获取校准方程分段数量
* @param card          试剂卡信息指针
* @param chl_idx       通道索引
* @param err           错误标志位
* @return 校准方程分段数量
 */
int QRJson_GetEquationSegment(QRJsonCard_t* card, uint32_t chl_idx, int* err)
{
  if (card == NULL)
  {
    QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
    return 0;
  }
  QRJsonCalcu_t calcu_ins = { 0 };
  if (QRJson_SearchCalcu(&calcu_ins, card, chl_idx, err))
  {
    QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
    return 0;
  }
  return QRJson_ConvertInt(calcu_ins.seg_cnt, QRJSON_MAX_EQUATION_SEGMENT, err);
}

/**
* 获取校准方程类型
* @param card          试剂卡信息指针
* @param chl_idx       通道索引
* @param seg_idx       分段索引
* @param err           错误标志位
* @return 校准方程类型
 */
int QRJson_GetEquationType(QRJsonCard_t* card, uint32_t chl_idx, uint32_t seg_idx, int* err)
{
  if (card == NULL)
  {
    QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
    return 0;
  }
  QRJsonEquation_t equation_ins = { 0 };
  if (QRJson_SearchEquation(&equation_ins, card, chl_idx, seg_idx, err))
  {
    QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
    return 0;
  }
  return QRJson_ConvertInt(equation_ins.type, QRJSON_MAX_EQUATION_TYPE, err);
}

/**
* 获取校准方程参数数量
* @param card          试剂卡信息指针
* @param chl_idx       通道索引
* @param seg_idx       分段索引
* @param err           错误标志位
* @return 校准方程参数数量
 */
int QRJson_GetEquationParamCnt(QRJsonCard_t* card, uint32_t chl_idx, uint32_t seg_idx, int* err)
{
  if (card == NULL)
  {
    QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
    return 0;
  }
  QRJsonEquation_t equation_ins = { 0 };
  if (QRJson_SearchEquation(&equation_ins, card, chl_idx, seg_idx, err))
  {
    QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
    return 0;
  }
  return QRJson_ConvertInt(equation_ins.param_cnt, QRJSON_MAX_EQUATION_TYPE, err);
}

/**
* 获取校准方程参数值
* @param buf           缓存地址
* @param size          缓存大小
* @param card          试剂卡信息指针
* @param chl_idx       通道索引
* @param seg_idx       分段索引
* @param param_idx     参数索引
* @param err           错误标志位
* @return  0成功/其他失败
 */
int QRJson_GetEquationParamVal(char* buf, uint32_t size, QRJsonCard_t* card, uint32_t chl_idx, uint32_t seg_idx, uint32_t param_idx, int* err)
{
  if (card == NULL)
  {
    QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
    return 0;
  }
  QRJsonEquation_t equation_ins = { 0 };
  if (QRJson_SearchEquation(&equation_ins, card, chl_idx, seg_idx, err))
  {
    QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
    return 0;
  }
  uint32_t param_num = QRJson_ConvertInt(equation_ins.param_cnt, QRJSON_MAX_EQUATION_PARAM_NUM, err);
  if (param_idx < param_num)
  {
    return QRJson_ConvertFloat(buf, size, equation_ins.param + param_idx * QRJSON_MAX_EQUATION_PARAM_LEN, QRJSON_MAX_EQUATION_PARAM_LEN, err);
  }
  QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
  return 0;
}

/**
* 获取通道方程高/低门限
* @param buf           缓存地址
* @param size          缓存大小
* @param card          试剂卡信息指针
* @param chl_idx       通道索引
* @param seg_idx       分段索引
* @param high_flag     高门限标志位
* @param err           错误标志位
* @return  0成功/其他失败
 */
int QRJson_GetEquationGate(char* buf, uint32_t size, QRJsonCard_t* card, uint32_t chl_idx, uint32_t seg_idx, uint8_t high_flag, int* err)
{
  if (buf == NULL || !size || card == NULL)
  {
    QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
    return -1;
  }
  QRJsonEquation_t equation = { 0 };
  uint32_t oft = seg_idx;
  if (!high_flag && seg_idx > 0)
  {
    oft--;
  }
  if (QRJson_SearchEquation(&equation, card, chl_idx, oft, err))
  {
    QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
    return -1;
  }
  if (!high_flag && !seg_idx)
  {
    return QRJson_ConvertPositiveFloat(buf, size, equation.gate_low, QRJSON_MAX_EQUATION_GATE, -1, err);
  }
  return QRJson_ConvertPositiveFloat(buf, size, equation.gate_high, QRJSON_MAX_EQUATION_GATE, -1, err);
}

/**
* 获取结果门限（定性）
* @param buf           缓存地址
* @param size          缓存大小
* @param card          试剂卡信息指针
* @param chl_idx       通道索引
* @param gate_idx      门限索引
* @param err           错误标志位
* @return  0成功/其他失败
 */
int QRJson_GetResultGate(char* buf, uint32_t size, QRJsonCard_t* card, uint32_t chl_idx, uint32_t gate_idx, int* err)
{
  if (buf == NULL || !size || card == NULL || gate_idx >= QRJSON_MAX_RESULT_GATE_NUM)
  {
    QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
    return -1;
  }
  QRJsonResultGate_t* gate = QRJson_SearchResultGate(card, chl_idx, err);
  if (gate == NULL)
  {
    QRJSON_ERR_SET(err, QRJSON_ERR_FLAG)
    return -1;
  }
  return QRJson_ConvertPositiveFloat(buf, size, &gate->value[gate_idx][0], QRJSON_MAX_RESULT_GATE_LEN, -1, err);
}

#define QRJSON_ADD_OBJECT(obj, name, item, err_flag, err_string)             \
do{\
if(err_flag){\
   QRJSON_ERR_SET(err_string, "item data err:" name"")\
}\
if(cJSON_False == cJSON_AddItemToObject(obj, name, item)){\
   QRJSON_ERR_SET(err_string, "item add fail:" name"")\
}\
err_flag = 0;\
}while(0);


#define QRJSON_ADD_ARRAY(array, item, err_flag, err_string)             \
do{\
if(err_flag){\
   QRJSON_ERR_SET(err_string, "array data err")\
}\
if(cJSON_False == cJSON_AddItemToArray(array, item)){\
   QRJSON_ERR_SET(err_string, "array add fail")\
}\
}while(0);

/**
* 编码试剂卡信息
* @param buf           缓存地址
* @param size          缓存大小
* @param card          试剂卡信息指针
* @param err_string    错误字符串指针的地址
* @return  0成功/其他失败
 */
int QRJson_EncodeCardInfo(char* buf, uint32_t size, QRJsonCard_t* card, const char** err_string)
{
  if (buf == NULL || !size || card == NULL)
  {
    QRJSON_ERR_SET(err_string, "encode param err")
    return -1;
  }
  cJSON* obj, * item1, * item2;
  double value = 0;
  char text[QRJSON_TEXT_BUFSIZE] = "";
  obj = cJSON_CreateObject();
  err_string = NULL;
  int err_flag = 0;
  int decimal = 0;

  // 机型系列
  value = (double)QRJson_GetDevGroup(card, &err_flag);
  QRJSON_ADD_OBJECT(obj, CARD_ITEM_DEV_GROUP, cJSON_CreateNumber(value), err_flag, err_string);
  // 型号掩码
  value = (double)QRJson_GetDevMask(card, &err_flag);
  QRJSON_ADD_OBJECT(obj, CARD_ITEM_DEV_MASK, cJSON_CreateNumber(value), err_flag, err_string);
  // 盒号
  value = (double)QRJson_GetBoxNumber(card, &err_flag);
  QRJSON_ADD_OBJECT(obj, CARD_ITEM_BOX_NUMBER, cJSON_CreateNumber(value), err_flag, err_string);
  // 盒装规格（测试数）
  value = (double)QRJson_GetTestNumber(card, &err_flag);
  QRJSON_ADD_OBJECT(obj, CARD_ITEM_TEST_NUMBER, cJSON_CreateNumber(value), err_flag, err_string);
  // 批号
  memset(text, 0, QRJSON_TEXT_BUFSIZE);
  QRJson_GetBatchNunber(text, QRJSON_TEXT_BUFSIZE, card, &err_flag);
  QRJSON_ADD_OBJECT(obj, CARD_ITEM_BATCH_NUMBER, cJSON_CreateString(text), err_flag, err_string);
  // 生产日期
  memset(text, 0, QRJSON_TEXT_BUFSIZE);
  QRJson_GetProDate(text, QRJSON_TEXT_BUFSIZE, card, &err_flag);
  QRJSON_ADD_OBJECT(obj, CARD_ITEM_PRODUCTION_DATE, cJSON_CreateString(text), err_flag, err_string);
  // 失效日期
  memset(text, 0, QRJSON_TEXT_BUFSIZE);
  QRJson_GetExpDate(text, QRJSON_TEXT_BUFSIZE, card, &err_flag);
  QRJSON_ADD_OBJECT(obj, CARD_ITEM_EXPIRE_DATE, cJSON_CreateString(text), err_flag, err_string);
  // 信号增益
  value = (double)QRJson_GetSignGain(card, &err_flag);
  QRJSON_ADD_OBJECT(obj, CARD_ITEM_SIGNAL_GAIN, cJSON_CreateNumber(value), err_flag, err_string);
  // 等待时间
  value = (double)QRJson_GetWaitTime(card, &err_flag);
  QRJSON_ADD_OBJECT(obj, CARD_ITEM_WAIT_TIME, cJSON_CreateNumber(value), err_flag, err_string);
  // 方法学
  value = (double)QRJson_GetMethodID(card, &err_flag);
  QRJSON_ADD_OBJECT(obj, CARD_ITEM_METHOD_ID, cJSON_CreateNumber(value), err_flag, err_string);
  // 项目ID
  value = (double)QRJson_GetProjID(card, &err_flag);
  QRJSON_ADD_OBJECT(obj, CARD_ITEM_PROJ_ID, cJSON_CreateNumber(value), err_flag, err_string);
  // 算法ID
  value = 1;
  QRJSON_ADD_OBJECT(obj, CARD_ITEM_ALGO_ID, cJSON_CreateNumber(value), err_flag, err_string);
  // 项目名称
  memset(text, 0, QRJSON_TEXT_BUFSIZE);
  QRJson_GetProjName(text, QRJSON_TEXT_BUFSIZE, card, &err_flag);
  QRJSON_ADD_OBJECT(obj, CARD_ITEM_PROJ_NAME, cJSON_CreateString(text), err_flag, err_string);
  // C线位置索引
  value = (double)QRJson_GetCLineIndex(card, &err_flag);
  QRJSON_ADD_OBJECT(obj, CARD_ITEM_LINE_CINDEX, cJSON_CreateNumber(value), err_flag, err_string);
  int cline_idx = value;
  // C线起始位置
  value = (double)QRJson_GetCLineStart(card, &err_flag);
  QRJSON_ADD_OBJECT(obj, CARD_ITEM_LINE_START, cJSON_CreateNumber(value), err_flag, err_string);
  // C线失效值
  value = (double)QRJson_GetCLineInvalid(card, &err_flag);
  QRJSON_ADD_OBJECT(obj, CARD_ITEM_LINE_INVALID, cJSON_CreateNumber(value), err_flag, err_string);
  // T线寻峰宽度
  value = (double)QRJson_GetTLineWindow(card, &err_flag);
  QRJSON_ADD_OBJECT(obj, CARD_ITEM_TLINE_WINDOW, cJSON_CreateNumber(value), err_flag, err_string);

  cJSON* array_line = cJSON_AddArrayToObject(obj, CARD_ITEM_LINE);
  for (uint32_t idx = 0; ; idx++)
  {
    QRJsonLine_t* line = QRJson_SearchLine(card, idx, &err_flag);
    if (line == NULL)
    {
      break;
    }
    item1 = cJSON_CreateObject();
    QRJSON_ADD_ARRAY(array_line, item1, err_flag, err_string);
    // 线编号
    QRJSON_ADD_OBJECT(item1, CARD_ITEM_LINE_ID, cJSON_CreateNumber(idx + 1), err_flag, err_string);
    // 线偏移量
    if (idx + 1 == cline_idx)
    {
      value = (double)QRJson_GetCLineOffset(line, &err_flag);
      QRJSON_ADD_OBJECT(item1, CARD_ITEM_LINE_OFFSET, cJSON_CreateNumber(value), err_flag, err_string);
    }
    else
    {
      value = (double)QRJson_GetTLineOffset(line, &err_flag);
      QRJSON_ADD_OBJECT(item1, CARD_ITEM_LINE_OFFSET, cJSON_CreateNumber(value), err_flag, err_string);
    }
    // 线宽度
    value = (double)QRJson_GetLineWidth(line, &err_flag);
    QRJSON_ADD_OBJECT(item1, CARD_ITEM_LINE_WIDTH, cJSON_CreateNumber(value), err_flag, err_string);
  }

  cJSON* array_project = cJSON_AddArrayToObject(obj, CARD_ITEM_CHL);
  for (uint32_t idx = 0;; idx++)
  {
    QRJsonChl_t* chl = QRJson_SearchChl(card, idx, NULL);
    if (chl == NULL)
    {
      break;
    }
    item1 = cJSON_CreateObject();
    QRJSON_ADD_ARRAY(array_project, item1, err_flag, err_string);
    // 通道ID
    value = (double)QRJson_GetChlID(chl, &err_flag);
    QRJSON_ADD_OBJECT(item1, CARD_ITEM_CHL_ID, cJSON_CreateNumber(value), err_flag, err_string);
    // 小数位数
    value = (double)QRJson_GetChlDecimal(chl, &err_flag);
    QRJSON_ADD_OBJECT(item1, CARD_ITEM_CHL_DECIMAL, cJSON_CreateNumber(value), err_flag, err_string);
    decimal = value;
    // 通道名称
    memset(text, 0, QRJSON_TEXT_BUFSIZE);
    QRJson_GetChlName(text, QRJSON_TEXT_BUFSIZE, chl, &err_flag);
    QRJSON_ADD_OBJECT(item1, CARD_ITEM_CHL_NAME, cJSON_CreateString(text), err_flag, err_string);
    // 单位
    memset(text, 0, QRJSON_TEXT_BUFSIZE);
    QRJson_GetChlUnit(text, QRJSON_TEXT_BUFSIZE, chl, &err_flag);
    QRJSON_ADD_OBJECT(item1, CARD_ITEM_CHL_UNIT, cJSON_CreateString(text), err_flag, err_string);

    // 计算模式前框
    value = (double)QRJson_GetChlModeNum(chl, 0, &err_flag);
    QRJSON_ADD_OBJECT(item1, CARD_ITEM_CHL_TMODE, cJSON_CreateNumber(value), err_flag, err_string);

    // 计算模式后框
    value = (double)QRJson_GetChlModeNum(chl, 1, &err_flag);
    QRJSON_ADD_OBJECT(item1, CARD_ITEM_CHL_CMODE, cJSON_CreateNumber(value), err_flag, err_string);

    cJSON* array_high_limit = cJSON_AddArrayToObject(item1, CARD_ITEM_CHL_HIGH_LIMIT);
    cJSON* array_low_limit = cJSON_AddArrayToObject(item1, CARD_ITEM_CHL_LOW_LIMIT);
    cJSON* array_equa = cJSON_AddArrayToObject(item1, CARD_ITEM_EQUATION);
    // 分段数量
    uint32_t equa_cnt = QRJson_GetEquationSegment(card, idx, &err_flag);
    QRJSON_ADD_OBJECT(item1, CARD_ITEM_EQUATION_CNT, cJSON_CreateNumber(equa_cnt), err_flag, err_string);
    for (uint32_t oft = 0; oft < equa_cnt; oft++)
    {
      item2 = cJSON_CreateObject();
      QRJSON_ADD_ARRAY(array_equa, item2, err_flag, err_string);
      // 类型
      value = (double)QRJson_GetEquationType(card, idx, oft, &err_flag);
      QRJSON_ADD_OBJECT(item2, CARD_ITEM_EQUATION_TYPE, cJSON_CreateNumber(value), err_flag, err_string);

      cJSON* array_gate = cJSON_AddArrayToObject(item2, CARD_ITEM_EQUATION_GATE);
      // 分段低门限
      memset(text, 0, QRJSON_TEXT_BUFSIZE);
      QRJson_GetEquationGate(text, QRJSON_TEXT_BUFSIZE, card, idx, oft, 0, &err_flag);
      QRJSON_ADD_ARRAY(array_gate, cJSON_CreateString(text), err_flag, err_string);
      // 分段高门限
      memset(text, 0, QRJSON_TEXT_BUFSIZE);
      QRJson_GetEquationGate(text, QRJSON_TEXT_BUFSIZE, card, idx, oft, 1, &err_flag);
      QRJSON_ADD_ARRAY(array_gate, cJSON_CreateString(text), err_flag, err_string);
      // 参数数量
      value = (double)QRJson_GetEquationParamCnt(card, idx, oft, &err_flag);
      QRJSON_ADD_OBJECT(item2, CARD_ITEM_EQUATION_PARAM_CNT, cJSON_CreateNumber(value), err_flag, err_string);
      uint32_t param_cnt = value;

      cJSON* array_param = cJSON_AddArrayToObject(item2, CARD_ITEM_EQUATION_PARAM_VAL);
      for (uint32_t cnt = 0; cnt < param_cnt; cnt++)
      {
        // 参数值
        memset(text, 0, QRJSON_TEXT_BUFSIZE);
        QRJson_GetEquationParamVal(text, QRJSON_TEXT_BUFSIZE, card, idx, oft, cnt, &err_flag);
        QRJSON_ADD_ARRAY(array_param, cJSON_CreateString(text), err_flag, err_string);
      }
    }

    // 通道低门限
    memset(text, 0, QRJSON_TEXT_BUFSIZE);
    QRJson_GetEquationGate(text, QRJSON_TEXT_BUFSIZE, card, idx, 0, 0, &err_flag);
    QRJSON_ADD_ARRAY(array_low_limit, cJSON_CreateString(text), err_flag, err_string);
    // 低门限显示值
    memset(text, 0, QRJSON_TEXT_BUFSIZE);
    QRJson_GetChlLimit(text, QRJSON_TEXT_BUFSIZE, chl, 0, decimal, &err_flag);
    QRJSON_ADD_ARRAY(array_low_limit, cJSON_CreateString(text), err_flag, err_string);
    // 通道高门限
    memset(text, 0, QRJSON_TEXT_BUFSIZE);
    QRJson_GetEquationGate(text, QRJSON_TEXT_BUFSIZE, card, idx, equa_cnt - 1, 1, &err_flag);
    QRJSON_ADD_ARRAY(array_high_limit, cJSON_CreateString(text), err_flag, err_string);
    // 高门限显示值
    memset(text, 0, QRJSON_TEXT_BUFSIZE);
    QRJson_GetChlLimit(text, QRJSON_TEXT_BUFSIZE, chl, 1, decimal, &err_flag);
    QRJSON_ADD_ARRAY(array_high_limit, cJSON_CreateString(text), err_flag, err_string);

    // 结果门限（定性）
    cJSON* array_gate = cJSON_AddArrayToObject(item1, CARD_ITEM_RESULT_GATE);
    for (uint32_t oft = 0; oft < QRJSON_MAX_RESULT_GATE_NUM; oft++)
    {
      memset(text, 0, QRJSON_TEXT_BUFSIZE);
      QRJson_GetResultGate(text, QRJSON_TEXT_BUFSIZE, card, idx, oft, &err_flag);
      QRJSON_ADD_ARRAY(array_gate, cJSON_CreateString(text), err_flag, err_string);
    }
  }

  int ret = 0;
  if (!cJSON_PrintPreallocated(obj, buf, size, 0))
  {
    QRJSON_ERR_SET(err_string, "encode err")
    ret = -2;
  }
  cJSON_Delete(obj);
  return ret;
__ENCODE_ERR:
  cJSON_Delete(obj);
  return -1;
}

/**
* 编码试剂卡信息更新文件
* @param buf           缓存地址
* @param size          缓存大小
* @param card          试剂卡信息指针
* @param err_string    错误字符串指针的地址
* @return 0成功/其他失败
 */
int QRJson_EncodeCardFile(char* buf, uint32_t size, QRJsonCard_t* card, const char** err_string)
{
  if (buf == NULL || !size || card == NULL)
  {
    QRJSON_ERR_SET(err_string, "encode param err")
    return -1;
  }
  memset(buf, 0, size);
  if (QRJson_EncodeCardInfo(buf, size, card, err_string))
  {
    return -2;
  }
  cJSON* obj, * item1;
  double value;
  char text[QRJSON_TEXT_BUFSIZE] = "";
  obj = cJSON_CreateObject();
  int err_flag = 0;

  cJSON* array_card = cJSON_AddArrayToObject(obj, CARD_ITEM_CARD);
  item1 = cJSON_CreateObject();
  QRJSON_ADD_ARRAY(array_card, item1, err_flag, err_string);
  QRJSON_ADD_OBJECT(item1, CARD_ITEM_INFORMATION, cJSON_CreateString(buf), err_flag, err_string);

  cJSON* array_project = cJSON_AddArrayToObject(obj, CARD_ITEM_PROJ);
  for (uint32_t idx = 0;; idx++)
  {
    QRJsonChl_t* chl = QRJson_SearchChl(card, idx, &err_flag);
    if (chl == NULL)
    {
      break;
    }
    item1 = cJSON_CreateObject();
    QRJSON_ADD_ARRAY(array_project, item1, err_flag, err_string);

    value = (double)QRJson_GetProjID(card, &err_flag);
    QRJSON_ADD_OBJECT(item1, CARD_ITEM_PROJ_ID, cJSON_CreateNumber(value), err_flag, err_string);

    memset(text, 0, QRJSON_TEXT_BUFSIZE);
    QRJson_GetProjName(text, QRJSON_TEXT_BUFSIZE, card, &err_flag);
    QRJSON_ADD_OBJECT(item1, CARD_ITEM_PROJ_NAME, cJSON_CreateString(text), err_flag, err_string);

    value = (double)QRJson_GetChlID(chl, &err_flag);
    QRJSON_ADD_OBJECT(item1, CARD_ITEM_CHL_ID, cJSON_CreateNumber(value), err_flag, err_string);

    memset(text, 0, QRJSON_TEXT_BUFSIZE);
    QRJson_GetChlName(text, QRJSON_TEXT_BUFSIZE, chl, &err_flag);
    QRJSON_ADD_OBJECT(item1, CARD_ITEM_CHL_NAME, cJSON_CreateString(text), err_flag, err_string);
  }
  memset(buf, 0, size);
  int ret = 0;
  if (!cJSON_PrintPreallocated(obj, buf, size, 0))
  {
    QRJSON_ERR_SET(err_string, "encode err")
    ret = -2;
  }
  cJSON_Delete(obj);
  return ret;
__ENCODE_ERR:
  cJSON_Delete(obj);
  return -1;
}




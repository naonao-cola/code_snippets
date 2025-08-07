#include <stdio.h>
#include <math.h>
#include <string.h>
#include "lua_port.h"
#include "algjson.h"


#define gcvt(i, buf)		sprintf(buf, "%.4f", i)  /*将float类型数据i转换为字符串，保留4位小数，结果保存在buf中*/
static char tempbuf[4] = { 0 };

///
/// @brief
///    将各个参数转为luacjson格式，供lua调用
///
/// @param[in]  json_buf          json文本格式
/// @param[in]  bufsize           文本长度
/// @param[in]  luaipt            luajson转换相关的输入参数
///
/// @return
///
/// @par History:
/// @xielihui，2022年4月13日，新建函数
///
///
int EncodeLuaInput(char *json_buf, int bufsize, AlgLuaInput *luaipt)
{
  if(json_buf == NULL || luaipt == NULL)
  {
    return ALG_ERR;
  }

  // 创建一个JSON 对象，保存结构体数组，
  cJSON *root = cJSON_CreateObject();

  if (!root)
  {
#ifdef ALGO_DEBUG_ENABLE
    printf("root is unvalid when cJSON_CreateObject.\n");
#endif
    return ALG_ERR;
  }

  // 将signal对象挂载到根上
  cJSON_AddItemToObject(root, "cinfo", cJSON_CreateString(luaipt->strCline));
  cJSON_AddItemToObject(root,"signal",cJSON_CreateString(luaipt->strTline));
  cJSON_AddItemToObject(root,"cardinfo",cJSON_CreateString(luaipt->strcard_info));
  cJSON_AddItemToObject(root, "sample_id", cJSON_CreateString(luaipt->sample_id));
  cJSON_AddItemToObject(root, "nature", cJSON_CreateString(luaipt->nature));

  // ***** 解析
#ifdef ALGO_DEBUG_ENABLE
  char*out = cJSON_Print(root);
  printf("%s\n", out);
#endif

  cJSON_PrintPreallocated(root, json_buf, bufsize, 0);
  cJSON_Delete(root);

  return ALG_OK;
}

///
/// @brief
///    将luacjson格式解码为算法结构体的检测结果
///
/// @param[in]  opt_text          json文本格式
/// @param[in]  channel_rst       各个通道的检测结果
///
/// @return
///
/// @par History:
/// @xielihui，2022年4月13日，新建函数
///
int DecodeLuaOutput(char *opt_text, ChannelResult* channel_rst)
{
  if(NULL == opt_text || NULL == channel_rst)
  {
    return ALG_ERR;
  }

  cJSON *root = NULL, *ret = NULL, *arr = NULL, *json = NULL;//定义4个cJOSN结构体指针
  uint8_t size = 0;
  root = cJSON_Parse(opt_text);
  arr = cJSON_GetObjectItem(root,"output");//在root结构体中查找"output"
  size = cJSON_GetArraySize(arr);

  channel_rst->channel_cnt = size;

  for(uint8_t i = 0; i < size; i++)
  {
    json = cJSON_GetArrayItem(arr, i);//获取数组第i个成员
    ret= cJSON_GetObjectItem(json, "channel_id");//在josn结构体中查找"channel_id"
    if(ret)
    {
      channel_rst->single_channel_rst[i].channel_no = ret->valueint;
    }
    ret= cJSON_GetObjectItem(json, "channel_name");//在josn结构体中查找"project_name"
    if(ret)
    {
      snprintf(channel_rst->single_channel_rst[i].channel_name, MAX_CHANNEL_NAME_LEN, "%s", ret->valuestring);
    }
    ret = cJSON_GetObjectItem(json, "signal");//在josn结构体中查找"signal"
    if(ret)
    {
      strcpy(channel_rst->single_channel_rst[i].signal, ret->valuestring);
    }
    ret = cJSON_GetObjectItem(json, "concentration");//在josn结构体中查找"concentration"
    if(ret)
    {
      strcpy(channel_rst->single_channel_rst[i].concentration, ret->valuestring);
    }
    ret = cJSON_GetObjectItem(json, "validflag");//在josn结构体中查找"validflag"
    if(ret)
    {
      channel_rst->single_channel_rst[i].validflag = ret->valueint;
    }
    ret = cJSON_GetObjectItem(json, "concentration_value");//在json结构体中查找"concentration_value"
    if(ret)
    {
      channel_rst->single_channel_rst[i].concentration_value = ret->valuedouble;
    }

    ret = cJSON_GetObjectItem(json, "unit");//在json结构体中查找"unit"
    if (ret)
    {
      strcpy(channel_rst->single_channel_rst[i].unit, ret->valuestring);
    }

    ret = cJSON_GetObjectItem(json, "coef");//在json结构体中查找"coef"
    if (ret)
    {
      channel_rst->single_channel_rst[i].coef = ret->valuedouble;
    }

    ret = cJSON_GetObjectItem(json, "nature");//在json结构体中查找"nature"
    if (ret)
    {
      strcpy(channel_rst->single_channel_rst[i].nature, ret->valuestring);
    }

    ret = cJSON_GetObjectItem(json, "natureflag");//在json结构体中查找"nature_flag"
    if (ret)
    {
      channel_rst->single_channel_rst[i].nature_flag = ret->valueint;
    }
    ret = cJSON_GetObjectItem(json, "mode");//在json结构体中查找"mode"
    if (ret)
    {
      strcpy(channel_rst->single_channel_rst[i].mode, ret->valuestring);
    }

  }
  cJSON_Delete(root);
  return 0;
}

///
/// @brief
///    将各个检测线的信息转换为字符串
///
/// @param[in]  signal_str         信号值组合的字符串
/// @param[in]  max_value_str      信号值组合的字符串
/// @param[in]  line_rst           各个测试线的检测结果
///
/// @return
///
/// @par History:
/// @xielihui，2022年4月13日，新建函数
/// @xielihui，2022年7月13日，modify
int ConverPeakInfoInput(char* strCline, char* strTsignal, int cid, int cerr, LineResult* line_rst, int testMode)
{
  if(NULL == strCline || NULL == strTsignal || NULL == line_rst)
  {
    return ALG_ERR;
  }

  // 检测线条数为0的时候,报错
  if(0 == line_rst->line_cnt)
  {
    return ALG_ERR;
  }

  // 组合各条线的信号值，将信号值组合成string,方便下一步写入json
  char line_signal_str[128] ={0};
  char signal_temp[128]= {0};
  double signal = 0.0;

  // C线面积赋值:计算值，失效值
  sprintf(line_signal_str, "%d", line_rst->single_line_rst[cid-1].area);
  memset(signal_temp, 0x00, 128);
  sprintf(signal_temp, "%d", cerr);
  sprintf(strCline, "%s;%s", line_signal_str, signal_temp);


  int firstflag = 0;
  // 信号存入lua
  if (TCAL == testMode)
  {
    // 校准模式下，送入面积以及对应的信号
    memset(line_signal_str, 0x00, 128);
    firstflag = 0;
    for (int i = 0; i < line_rst->channel_cnt; i++)
    {
      if (0 == firstflag)
      {
        strcpy(signal_temp, line_rst->single_line_rst[i].signal);
        firstflag = 1;
      }
      else
      {
        sprintf(signal_temp, "%s;%s", line_signal_str, line_rst->single_line_rst[i].signal);
      }
      strcpy(line_signal_str, signal_temp);
    }
    strcpy(strTsignal, line_signal_str);
  }
  else
  {
    // 常规测试和质控模式下，仅送信号
    memset(line_signal_str, 0x00, 128);
    firstflag = 0;
    for (uint8_t i = 0; i < line_rst->line_cnt; i++)
    {
      memset(signal_temp, 0x00, 128);
      if (i != (cid - 1))
      {
        if (0 == firstflag)
        {
          strcpy(signal_temp, line_rst->single_line_rst[i].signal);
          firstflag = 1;
        }
        else
        {
          sprintf(signal_temp, "%s;%s", line_signal_str, line_rst->single_line_rst[i].signal);
        }
        strcpy(line_signal_str, signal_temp);
      }
      else
      {
        continue;
      }
    }
    strcpy(strTsignal, line_signal_str);
  }
  return ALG_OK;
}


///
/// @brief
///    组织信号与试剂卡信息
///
/// @param[in]  luaipt          参数结构体
/// @param[in]  cardinfo        试剂卡信息指针
/// @param[in]  line_rst        测试线检测结果信息
/// @param[in]  sample_id       表征校准或非校准模式
///
/// @return
///
/// @par History:
/// @xielihui，2022年4月13日，新建函数
///
int ConverLuaInput(AlgLuaInput* luaipt, int cid, int cerr, char* cardinfo, LineResult* line_rst, int testMode, int nature)
{
  if(NULL == luaipt || NULL == cardinfo || NULL == line_rst)
  {
    return ALG_ERR;
  }

  if (ALG_ERR == ConverPeakInfoInput(luaipt->strCline, luaipt->strTline, cid, cerr, line_rst, testMode))
  {
    return ALG_ERR;
  }
  luaipt->strcard_info = cardinfo;

  // 默认常规测量
  luaipt->sample_id = "0";

  if (TCAL == testMode)
  {
    luaipt->sample_id = "1";
  }
  else if (TQC == testMode)
  {
    luaipt->sample_id = "2";
  }

  memset(tempbuf, 0x00, 4);
  sprintf(tempbuf, "%d",nature);
  luaipt->nature = tempbuf;

  return ALG_OK;
}

// 获取校准参数
int GetLineParam(char* cardinfo, AlgInput* input)
{
  if (NULL == cardinfo || NULL == input)
  {
    return ALG_ERR;
  }
  cJSON* root = NULL; // 定义4个cJOSN结构体指针
  cJSON* jsonline = NULL;
  cJSON* json = NULL;
  cJSON* jsonlineinfo = NULL;

  root = cJSON_Parse(cardinfo);
  if (NULL == root)
  {
    return ALG_ERR;
  }

  jsonline = cJSON_GetObjectItem(root, "LCID");
  input->line_para.cid = jsonline->valueint;

  jsonline = cJSON_GetObjectItem(root, "LErr");
  input->line_para.cerr = jsonline->valueint;

  jsonline = cJSON_GetObjectItem(root, "LSta");
  input->line_para.cstart = jsonline->valueint;


  jsonline = cJSON_GetObjectItem(root, "TWin");
  input->line_para.twindow= jsonline->valueint;

  // 获取线信息
  jsonline = cJSON_GetObjectItem(root, "Line");
  int cnt = cJSON_GetArraySize(jsonline);
  input->line_cnt = cnt;
  if (cnt <= 0)
  {
    return ALG_ERR;
  }

  // 解析线信息的成员
  for (uint8_t i = 0; i < input->line_cnt; i++)
  {
    json = cJSON_GetArrayItem(jsonline, i);
    if (NULL == json)
    {
      return ALG_ERR;
    }

    jsonlineinfo = cJSON_GetObjectItem(json, "LID");
    if (NULL == jsonlineinfo)
    {
      return ALG_ERR;
    }
    input->line_para.paras[i].line_id = jsonlineinfo->valueint;

    jsonlineinfo = cJSON_GetObjectItem(json, "LOft");
    if (NULL == jsonlineinfo)
    {
      return ALG_ERR;
    }
    input->line_para.paras[i].dis = jsonlineinfo->valueint;

    jsonlineinfo = cJSON_GetObjectItem(json, "LWid");
    if (NULL == jsonlineinfo)
    {
      return ALG_ERR;
    }
    input->line_para.paras[i].signal_window = jsonlineinfo->valueint;
  }
  cJSON_Delete(root);
  return ALG_OK;
}

// 获取方法学，荧光或者胶体金
int GetMethodParam(char* cardinfo, AlgInput* input)
{
  if (NULL == cardinfo || NULL == input)
  {
    return ALG_ERR;
  }

  cJSON* root = NULL; // 定义4个cJOSN结构体指针
  cJSON* json = NULL;

  root = cJSON_Parse(cardinfo);
  if (NULL == root)
  {
    return ALG_ERR;
  }

  // 获取线信息
  json = cJSON_GetObjectItem(root, "MID");
  int nature = json->valueint;
  switch (nature)
  {
  case 0:
  case 1:
  case 2:
  case 3:
  case 4:
  {
    input->method = MEATHOD_IM;
    break;
  }
  case 5:
  case 6:
  case 7:
  case 8:
  case 9:
  {
    input->method = MEATHOD_CG;
    break;
  }
  default:
  {
    input->method = MEATHOD_IM;
    return ALG_ERR;
  }
  }

  if ((0 == nature) || (9 == nature))
  {
    input->nature = QUANTITATINE;  // 定量
  }
  else if ((4 == nature) || (6 == nature))
  {
    input->nature = QUALITATIVES;  // 定性夹心
  }
  else if ((2 == nature) || (8== nature))
  {
    input->nature = HQUANTITATINES; // 半定量夹心
  }
  else if ((3 == nature) || (5 == nature))
  {
    input->nature = QUALITATIVEC;  // 定性竞争
  }
  else
  {
    input->nature = HQUANTITATINEC; // 半定量竞争
  }


  cJSON_Delete(root);

  return ALG_OK;
}


// 获取通道数和门限，进而判定校准或者常规测量
int GetChannelModeAndGate(char* cardinfo, AlgInput * input)
{
  if (NULL == cardinfo || NULL == input)
  {
    return ALG_ERR;
  }


  cJSON* root = NULL; // 定义4个cJOSN结构体指针
  cJSON* channel = NULL;
  cJSON* onechannel = NULL;
  cJSON* json = NULL;
  cJSON* gain = NULL;

  root = cJSON_Parse(cardinfo);
  if (NULL == root)
  {
    return ALG_ERR;
  }

  input->proid = cJSON_GetObjectItem(root, "PID")->valueint;
  input->gain = cJSON_GetObjectItem(root, "Gain")->valueint;

  channel = cJSON_GetObjectItem(root, "Chl");
  input->channel_cnt = cJSON_GetArraySize(channel);

  // 根据通道数和线条数的关系确认测试模式：校准、质控还是常规
  if (input->channel_cnt == input->line_cnt && input->proid == CALIPROID)
  {
    input->testmode = TCAL;    // 通道数等于线条数，说明是校准
  }
  else if (input->channel_cnt < input->line_cnt && input->proid == CALIPROID)
  {
    input->testmode = TQC;      // 通道数小于线条数，说明是质控
  }
  else
  {
    input->testmode = TNORMAL;  // 常规测试
  }


  char info[16];
  int cid = input->line_para.cid - 1;
  int index = 0;

  if (TCAL == input->testmode)
  {
    // 校准流程
    for (uint8_t i = 0; i < (input->channel_cnt); i++)
    {
      onechannel = cJSON_GetArrayItem(channel, i);
      if (NULL == onechannel)
      {
        return ALG_ERR;
      }

      // 获取信号模式和保留位数
      input->line_para.paras[i].Tmodel = cJSON_GetObjectItem(onechannel, "TMode")->valueint;
      input->line_para.paras[i].Cmodel = cJSON_GetObjectItem(onechannel, "CMode")->valueint;
      input->line_para.paras[i].decimal = cJSON_GetObjectItem(onechannel, "Deci")->valueint;

      // 获取低门限
      json = cJSON_GetArrayItem(cJSON_GetObjectItem(onechannel, "LLmt"), 0);
      memset(info, 0xff, 16);
      snprintf(info, 16, "%s", json->valuestring);
      input->line_para.paras[i].gate = atof(info);
    }
  }
  else
  {
    // 常规流程和质控流程
    for (uint8_t i = 0; i < (input->line_cnt); i++)
    {
      if (cid == i)
      {
        input->line_para.paras[i].Tmodel = 0;
        input->line_para.paras[i].Cmodel = 0;
        input->line_para.paras[i].gate = 0.0;
        continue;
      }
      onechannel = cJSON_GetArrayItem(channel, index);
      if (NULL == onechannel)
      {
        return ALG_ERR;
      }

      // 获取信号模式和保留位数
      input->line_para.paras[i].Tmodel = cJSON_GetObjectItem(onechannel, "TMode")->valueint;
      input->line_para.paras[i].Cmodel = cJSON_GetObjectItem(onechannel, "CMode")->valueint;
      input->line_para.paras[i].decimal = cJSON_GetObjectItem(onechannel, "Deci")->valueint;

      // 获取低门限
      json = cJSON_GetArrayItem(cJSON_GetObjectItem(onechannel, "LLmt"), 0);
      memset(info, 0xff, 16);
      snprintf(info, 16, "%s", json->valuestring);
      input->line_para.paras[i].gate = atof(info);
      index = index + 1;
    }
  }

  cJSON_Delete(root);
  return ALG_OK;
}
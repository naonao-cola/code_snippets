#include <stdio.h>
#include <math.h>
#include <string.h>
#include "lua_port.h"
//#include "algjson.h"
#include "algpeak.h"


///
/// @brief
///     产生高斯模板
///
/// @param[in]  gausstemp       指向装载高斯滤波参数的地址
/// @param[in]  sum             高斯加权和
/// @param[in]  sigma           高斯sigma
///
/// @return
///
/// @par History:
/// @xielihui，2022年3月29日，新建函数
///
int GenerateGaussTemp(float* gausstemp, double* sum, float sigma, int window)
{
  if (NULL == gausstemp || NULL == sum)
  {
    return ALG_ERR;
  }

  double gausstempsum = 0.0;
  for (int i = 0; i < (window * 2 + 1); i++)
  {
    gausstemp[i] = exp(-(i + 1 - window) * ((i + 1 - window)) / (2 * sigma * sigma)) / (sigma * sqrt(2 * PI));
  }
  // 计算高斯参数的总和
  for (int i = 0; i < (window * 2 + 1); i++)
  {
    gausstempsum = gausstempsum + gausstemp[i];
  }
  *sum = gausstempsum;
  return ALG_OK;
}


/// 高斯滤波
/// @param[in]  input           测试线输入参数
/// @param[in]  result          测试线检测结果
/// @param[in]  gausstemp       高斯模板
/// @param[in]  gausstempsum    高斯模板系数之和
///
/// @return
///
/// @par History:
/// @xielihui，2023年1月18日，新建函数
///
int GaussianFilter(float* dst, int* src, float* gausstemp, float gausstempsum, int data_len, int window)
{
  if(NULL == src || NULL == gausstemp)
  {
    return ALG_ERR;
  }

  // 数据长度
  if(data_len <= window *2)
  {
#ifdef ALGO_DEBUG_ENABLE
    printf("data len is unvalid when excute to filter.\n");
#endif
    return ALG_ERR;
  }

  // 高斯滤波
  double sum = 0.0;
  int index = 0;
  for(int i = window; i < (data_len - window); i++)
  {
    sum = 0.0;
    index = 0;

    for (int j = (i- window); j <=(i+ window);j++)
    {
      sum = sum + *(src + j) * gausstemp[index];
      index++;
    }
    *(dst + i) = sum / gausstempsum;
  }
  return ALG_OK;
}


/// 中值滤波
/// @param[in]  input           测试线输入参数
/// @param[in]  result          测试线检测结果
///
/// @return
///
/// @par History:
/// @xielihui，2023年1月10日，新建函数
///
int MedianFilter(float* dst, int* src, int data_len)
{
  if (NULL == dst || NULL == src || data_len <= 0)
  {
    return ALG_ERR;
  }

  // 数据长度
  if (data_len <= MEDAIN_FILTER_WINDOWN * 2)
  {
#ifdef ALGO_DEBUG_ENABLE
    printf("data len is unvalid when excute to filter.\n");
#endif
    return ALG_ERR;
  }

  // 中值滤波
  uint16_t len = 2 * MEDAIN_FILTER_WINDOWN + 1;
  float* buff = NULL;
  uint32_t isize = (2 * MEDAIN_FILTER_WINDOWN + 1) * sizeof(float);
  buff = (float*)Alg_malloc(isize);
  float medain_value = 0.0;

  uint16_t m = 0;
  uint16_t n = 0;
  uint16_t tmp = 0;
  uint16_t index = 0;

  for (uint16_t i = MEDAIN_FILTER_WINDOWN; i < (data_len - MEDAIN_FILTER_WINDOWN); i++)
  {
    memset(buff, 0x00, isize);
    index = 0;
    for (uint16_t j = (i - MEDAIN_FILTER_WINDOWN); j <= (i + MEDAIN_FILTER_WINDOWN); j++)
    {
      // 数据存入
      buff[index] = *(src + j);
      index++;
    }
    // 冒泡排序
    m = 0;
    n = 0;
    tmp = 0;
    medain_value = 0;
    for (m = 0; m < len - 1; m++)
    {
      for (n = 0; n < (len - m - 1); n++)
      {
        if (buff[n] > buff[n + 1])
        {
          tmp = buff[n];
          buff[n] = buff[n + 1];
          buff[n + 1] = tmp;
        }
      }
    }
    medain_value = buff[MEDAIN_FILTER_WINDOWN];
    *(dst+i) = medain_value;
  }

  // 释放内存
  if (NULL != buff)
  {
    free(buff);
    buff = NULL;
  }
  return ALG_OK;
}


/// 高斯滤波
/// @param[in]  input           测试线输入参数
/// @param[in]  result          测试线检测结果
/// @param[in]  gausstemp       高斯模板
/// @param[in]  gausstempsum    高斯模板系数之和
///
/// @return
///
/// @par History:
/// @xielihui，2023年1月18日，新建函数
///
int ComGaussianFilter(float* dst, int* src, float* gausstemp, float gausstempsum, int data_len, int window)
{
  if (NULL == src || NULL == gausstemp || NULL == dst)
  {
    return ALG_ERR;
  }

  // 数据长度
  if (data_len <= MEAN_FILTER_WINDOWN * 2 || data_len <= window * 2)
  {
#ifdef ALGO_DEBUG_ENABLE
    printf("data len is unvalid when excute to filter.\n");
#endif
    return ALG_ERR;
  }


  // step1:中值滤波
  if(ALG_ERR == MedianFilter(dst, src, data_len))
  {
#ifdef ALGO_DEBUG_ENABLE
    printf("GaussianFilter failed when first process.\n");
#endif
    return ALG_ERR;
  }


  // 高斯滤波
  int index = 0;
  float* buff = NULL;
  double sum = 0.0;

  int isize = MAX_SAMPLE_CNT * sizeof(float);
  buff = (float*)Alg_malloc(isize);
  memset(buff, 0, isize);

  for (int i = window; i < (data_len - window); i++)
  {
    sum = 0.0;
    index = 0;

    for (int j = (i - window); j <= (i + window); j++)
    {
      sum = sum + *(dst + j) * gausstemp[index];
      index++;
    }
    *(buff + i) = sum / gausstempsum;
  }

  for (int i = window; i < (data_len - window); i++)
  {
    *(dst + i) = *(buff + i);
  }

  // 释放内存
  if (NULL != buff)
  {
    Alg_free(buff);
    buff = NULL;
  }

  return ALG_OK;
}


///
/// @brief
///     数据处理
///
/// @param[in]  input           算法输入
/// @param[in]  line_rst        检测线处理结果
/// @param[in]  handle          算法句柄
/// @param[in]  model           测量模式
///
/// @return  返回函数处理的成功与否的状态
///
/// @par History:
/// @xielihui，2023年1月18日，新建函数
///
int DataFilter(LineResult* line_rst, AlgInput* input, Alg_HandleTypeDef* handle, int model)
{
  // 参数检查
  if (NULL == input || NULL == line_rst || NULL == handle)
  {
    return ALG_ERR;
  }

  // 数据长度
  int data_len = input->length;

  int window = 0;

  // 高斯滤波
  window = COM_GAUSSAIN_FILTER_WINDOWN;
  if (ALG_ERR == ComGaussianFilter(line_rst->filter_data, input->data, handle->comgausstemp, handle->comgausstempsum, data_len, COM_GAUSSAIN_FILTER_WINDOWN))
  {
#ifdef ALGO_DEBUG_ENABLE
    printf("GaussianFilter failed when second process.\n");
#endif
    return ALG_ERR;
  }

  // 头补齐
  for (int i = 0; i < window; i++)
  {
    line_rst->filter_data[i] = line_rst->filter_data[window];
  }

  // 尾补齐
  for (int i = (data_len - window); i < data_len; i++)
  {
    line_rst->filter_data[i] = line_rst->filter_data[data_len - window - 1];
  }

  return ALG_OK;
}

///
/// @brief
///     水平翻转
///
/// @param[in]  data         指向待处理数据内存指针
/// @param[in]  data_len     待处理数据长度
///
/// @return  返回函数处理的成功与否的状态
///
/// @par History:
/// @xielihui，2024年1月18日，新建函数
///
int FlipHorizontal(int* data, int data_len)
{
  // 参数检查
  if (NULL == data)
  {
    return ALG_ERR;
  }
  // 原始参数信号逆转
  int* buff = NULL;
  int isize = MAX_SAMPLE_CNT * sizeof(int);
  buff = (int*)Alg_malloc(isize);
  int j = 0;
  for (int i = data_len - 1; i >= 0; i--)
  {
    *(buff + j) = *(data + i);
    j++;
  }

  for (uint16_t i = 0; i < data_len; i++)
  {
    *(data + i) = *(buff + i);
  }

  // 释放内存
  if (NULL != buff)
  {
    Alg_free(buff);
    buff = NULL;
  }
  return ALG_OK;
}

///
/// @brief
///     寻找整型数组中最大的值
///
/// @param[in]  data         指向待处理数据内存指针
/// @param[in]  data_len     待处理数据长度
///
/// @return  返回最大值
///
/// @par History:
/// @xielihui，2024年1月18日，新建函数
///
int GetMaxValue(int* data, int data_len)
{
  if (NULL == data)
  {
    return ALG_ERR;
  }

  int max = *data;
  for (int i = 0; i < data_len; i++, data++)
  {
    if (max < *data)
    {
      max = *data;
    }
  }
  return max;
}

///
/// @brief
///     寻找整型数组中最小的值
///
/// @param[in]  data         指向待处理数据内存指针
/// @param[in]  data_len     待处理数据长度
///
/// @return  返回最小值
///
/// @par History:
/// @xielihui，2024年1月18日，新建函数
///
int GetMinValue(int* data, int data_len)
{
  if (NULL == data)
  {
    return ALG_ERR;
  }

  int min = *data;
  for (int i = 0; i < data_len; i++, data++)
  {
    if (min > *data)
    {
      min = *data;
    }
  }
  return min;
}

///
/// @brief
///     寻找一段区间最大的值以及对应的点
/// @param[in]  maxValue     存储最大值的地址
/// @param[in]  maxpoint     存储最大值对应的采样点的地址
/// @param[in]  data_len     数据区间长度
///
/// @return  返回函数处理的成功与否的状态
///
/// @par History:
/// @xielihui，2024年1月18日，新建函数
int FindMaxValue(float* maxValue, int* maxpoint, float* src, int data_len)
{
  if (NULL == maxValue || NULL == maxpoint || NULL == src)
  {
    return ALG_ERR;
  }

  for (int i = 0; i < data_len; i++, src++)
  {
    if (*maxValue < *src)
    {
      *maxpoint = i;
      *maxValue = *src;
    }
  }
  return ALG_OK;
}


///
/// @brief
///     垂直翻转
///
/// @param[in]  data         指向待处理数据内存指针
/// @param[in]  data_len     待处理数据长度
///
/// @return  返回函数处理的成功与否的状态
///
/// @par History:
/// @xielihui，2024年1月18日，新建函数
///
int FlipVertical(int* data, int data_len)
{
  // 参数检查
  if (NULL == data)
  {
    return ALG_ERR;
  }
  // 寻找序列中最大的点
  int maxValue = GetMaxValue(data, data_len);

  for (uint16_t i = 0; i < data_len; i++)
  {
    *(data + i) = 2 * maxValue - *(data + i);
  }
  int minValue = GetMinValue(data, data_len);

  for (uint16_t i = 0; i < data_len; i++)
  {
    *(data + i) = *(data + i) - minValue;
  }
  return ALG_OK;
}

///
/// @brief
///     数据处理,获取每一条线的信息
///
/// @param[in]  input           算法输入
/// @param[in]  line_rst        检测线处理结果
/// @param[in]  handle          算法句柄
/// @param[in]  coef            校准系数
///
/// @return  返回函数处理的成功与否的状态
///
/// @par History:
/// @xielihui，2023年1月18日，新建函数
///
int AlgGetLineResult(AlgInput* input, LineResult* line_rst, Alg_HandleTypeDef* handle, char* coef)
{

#ifdef ALGO_DEBUG_ENABLE
  printf("AlgGetLineResult is in progress.\n");
#endif
  // 参数检查
  if (NULL == input || NULL == line_rst || NULL == handle)
  {
    return ALG_ERR;
  }

  if (input->method == MEATHOD_CG)
  {
    // 胶体金垂直反转

    if (ALG_ERR == FlipVertical(input->data, input->length))
    {
      return ALG_ERR;
    }
  }

  // 滤波
  if (ALG_ERR == DataFilter(line_rst, input, handle, input->method))
  {
#ifdef ALGO_DEBUG_ENABLE
    printf("DataFilter failed.\n");
#endif
    return ALG_ERR;
  }
  // 寻峰处理及其计算面积
  if (ALG_ERR == FindLineInfo(line_rst, input, coef))
  {
#ifdef ALGO_DEBUG_ENABLE
    printf("FindLineInfo failed.\n");
#endif
    return ALG_ERR;
  }
  return ALG_OK;
}


///
/// @brief
///     合并峰
///
/// @param[in]  merge_info   指向峰合并后的结果的内存指针
/// @param[in]  merge_cnt    峰合并的数目
/// @param[in]  firstpeak    指向第一次寻峰结果内存指针
/// @param[in]  peak_cnt     第一次寻峰结果的数目
/// @return ALG_OK 表示成功 ALG_ERR 表示失败
///
/// @par History:
/// @xielihui，2024年1月18日，新建函数
///
int MergePeak(MergeInfo* merge_info, int* merge_cnt, int* firstpeak, int peak_cnt)
{
#ifdef ALGO_DEBUG_ENABLE
  printf("MergePeak is in progress.\n");
#endif
  if(NULL == merge_cnt || NULL == merge_info || NULL == firstpeak || peak_cnt <=1)
  {
    return ALG_ERR;
  }

  int temp;
  int isize = MAX_FIRST_PEAK_CNT * sizeof(int);
  int* delta_array = (int*)Alg_malloc(isize);
  int i, j;

  // 计算两峰之间的距离（峰点与峰点的距离）
  for (i = 0; i < peak_cnt - 1; i++)
  {
    temp = *(firstpeak + i + 1) - *(firstpeak + i);
    *(delta_array + i) = temp;
  }
  *(delta_array + peak_cnt - 1) = MERGE_LIMIT + 10;

  // 遍历合并峰,合并峰的结果存在merge_info中 */
  int start_peak = 0;
  int old_start = 0;

  int mergecnt = 0;
  while (start_peak < peak_cnt)
  {
    old_start = start_peak;

    for (j = start_peak; j < peak_cnt; j++)
    {
      if (*(delta_array + j) > MERGE_LIMIT)
        break;
      else
        continue;
    }

    if (old_start == j)
    {
      if (mergecnt < MAX_SECOND_PEAK_CNT - 1)
      {
        merge_info[mergecnt].old_point = *(firstpeak + j);
        merge_info[mergecnt].new_point = *(firstpeak + j);  // 合并峰的位置
        merge_info[mergecnt].adj_point = 0;                // 合并峰结束的位置
        merge_info[mergecnt].flag = 0;                     // 峰合并标志
        mergecnt = mergecnt + 1;
      }
    }
    else
    {
      if (mergecnt < MAX_SECOND_PEAK_CNT - 1)
      {
        merge_info[mergecnt].old_point = *(firstpeak + old_start);
        merge_info[mergecnt].new_point = round((*(firstpeak + old_start) + *(firstpeak + j)) / 2);  // 合并峰的位置
        merge_info[mergecnt].adj_point = *(firstpeak + j);                // 合并峰结束的位置
        merge_info[mergecnt].flag = 1;                     // 峰合并标志
        mergecnt = mergecnt + 1;
      }
    }
    start_peak = j + 1;
  }
  *merge_cnt = mergecnt;

  // 释放内存
  if (NULL != delta_array)
  {
    Alg_free(delta_array);
    delta_array = NULL;
  }

  return ALG_OK;
}



///
/// @brief
///     寻找左右谷点
///
/// @param[in]  left_point   左侧峰点
/// @param[in]  right_point  右侧峰点
/// @param[in]  s_left       左侧寻峰起点
/// @param[in]  s_right      右侧寻峰起点
/// @param[in]  start        起点
/// @param[in]  end          结束点
/// @return ALG_OK 表示成功 ALG_ERR 表示失败
///
/// @par History:
/// @xielihui，2024年1月18日，新建函数
///
int SearchLRvalley(int* left_point, int* right_point, float* data, int s_left, int s_right, int start, int end)
{
#ifdef ALGO_DEBUG_ENABLE
  printf("SearchLRvalley is in progress.\n");
#endif
  if (NULL == left_point || NULL == right_point || NULL == data)
  {
    return ALG_ERR;
  }

  int left_cnt = 0;
  int right_cnt = 0;
  int j = 0;

  // 寻找左波谷点
  for (j = s_left; j >= start; j--)
  {
    if (j >= 1)
    {
      // 后面一个总大于等于前面的值
      if (*(data+j) >= *(data + j - 1))
      {
        left_cnt += 1;
      }
      else
      {
        break;
      }
    }
    else
    {
      // 搜索到边界，跳出
      break;
    }

  }

  if (left_cnt == 0)
  {
    *left_point = s_left;
  }
  else
  {
    *left_point = j;
  }


  // 寻找右波谷点
  for (j = s_right; j <= end; j++)
  {
    //
    if (*(data + j) >= *(data + j + 1))
    {
      right_cnt += 1;
    }
    else
    {
      break;
    }
  }

  if (right_cnt == 0)
  {
    *right_point = s_right;
  }
  else
  {

    *right_point = j;
  }
  return 	ALG_OK;
}

///
/// @brief
///     二次寻峰
///
/// @param[in]  find_peak_info   指向寻峰结果的内存地址
/// @param[in]  mergePeak        合并峰
/// @param[in]  merge_cnt        合并峰的个数
/// @param[in]  start           起点
/// @param[in]  end              结束点
/// @return ALG_OK 表示成功 ALG_ERR 表示失败
///
/// @par History:
/// @xielihui，2024年1月18日，新建函数
///
int SecondFindPeak(FindPeakInfo* find_peak_info, float *data, MergeInfo * mergePeak, int merge_cnt, int start, int end)
{
  if (NULL == find_peak_info || NULL == data || NULL == mergePeak)
  {
    return ALG_ERR;
  }

  int left_locate = 0;
  int right_locate = 0;
  int left_point = 0;
  int right_point = 0;
  int point = 0;
  int peak_cnt = 0;
  int maxpoint = start;
  float maxValue = 0.0;
  int datacnt = end - start + 1;

  for (int i = 0; i < merge_cnt; i++)
  {
    point = mergePeak[i].new_point;

    if (1 == mergePeak[i].flag)
    {
      // 有合并的峰
      left_locate = mergePeak[i].old_point;
      right_locate = mergePeak[i].adj_point;
    }
    else if (0 == mergePeak[i].flag)
    {
      // 没有合并的峰
      left_locate = mergePeak[i].old_point;
      right_locate = mergePeak[i].old_point;
    }
    SearchLRvalley(&left_point, &right_point, data, left_locate, right_locate, start, end);

    int dis_left = abs(point - left_point);
    int dis_right = abs(point - right_point);
    FindMaxValue(&maxValue, &maxpoint, data+start, datacnt);

    if (dis_left + dis_right <= WIDTH_LLIMIT)
    {
      continue;
    }

    find_peak_info->one_peak_info[peak_cnt].start_point = left_point;
    find_peak_info->one_peak_info[peak_cnt].peak_point = point;
    find_peak_info->one_peak_info[peak_cnt].end_point = right_point;
    find_peak_info->one_peak_info[peak_cnt].peak_value = *(data + point);
    peak_cnt = peak_cnt + 1;
  }
  find_peak_info->peak_num = peak_cnt;
  return ALG_OK;
}


///
/// @brief
///     峰寻找函数
///
/// @param[in]  find_peak_info      保存峰合并后的峰检测结果的结构体指针
/// @param[in]  data                指向存储吸光度数据的内存指针
/// @param[in]  data_len            数据长度
/// @param[in]  satrt               区间起点
/// @param[in]  end                 区间结束点
///
/// @return ALG_OK 表示成功，ALG_ERR表示成功
/// @par History:
/// @xielihui，2024年1月18日，新建函数
///

int FindPeak(FindPeakInfo* find_peak_info, float *data, int data_len, int start, int end)
{
#ifdef ALGO_DEBUG_ENABLE
  printf("FindPeak is in progress.\n");
#endif

  if(NULL == data || NULL == find_peak_info)
  {
    return ALG_ERR;
  }

  // 定义第一次寻峰的个数
  uint16_t first_peak_cnt = 0;


  // 初次寻峰
  float pre_value = 0.0;
  float curent_value = 0.0;
  float next_value = 0.0;
  int peak_cnt = 0;
  int firstpeak[MAX_FIRST_PEAK_CNT];

  // 第一次寻峰，寻找可能的点
  int i;
  int special_points = 0;
  float special_values = 0.0;
  int special_flag = 0;

  // 区间异常判定，只能是1~248
  if (start < 1)
  {
    start = 1;
  }

  if (end > (data_len - 2))
  {
    end = data_len - 2;
  }


  if (start >= end)
  {
    return ALG_ERR;
  }

  for (i = (start+1); i <= (end-1); i++)
  {
    pre_value = *(data + i - 1);
    curent_value = *(data + i);
    next_value = *(data + i + 1);

    if (curent_value > pre_value && curent_value > next_value)
    {
      if (peak_cnt < MAX_FIRST_PEAK_CNT)
      {
        firstpeak[peak_cnt] = i;
        peak_cnt = peak_cnt + 1;
      }
      else
      {
        printf("Firstpeak cnt is larger than MAX_FIRST_PEAK_CNT.\n");
        break;
      }
    }
    else
    {
      if (curent_value == next_value)
      {
        // 中间与右侧相等，进行不同的策略
        if (special_flag == 0)
        {
          // 记录第一次出现的点
          special_points = i - 1;              // 记录前一个点
          special_values = *(data + (i - 1)); // 记录前一个点的值
        }
        special_flag = special_flag + 1;
      }
      else
      {
        // 存在连续的情况后中断
        if (special_flag > 0)
        {
          if (special_values < curent_value && next_value < curent_value)
          {
            // 是一个峰，峰点为当前峰
            if (peak_cnt < MAX_FIRST_PEAK_CNT)
            {
              firstpeak[peak_cnt] = i;
              peak_cnt = peak_cnt + 1;
            }
            else
            {
              printf("Firstpeak cnt is larger than MAX_FIRST_PEAK_CNT.\n");
              break;
            }
          }
          special_flag = 0;
        }
      }
    }

  }

  // 没有峰，直接返回
  if (0 == peak_cnt)
  {
    printf("No peak when firtpeak detection.\n");
    find_peak_info->peak_num = 0;
    return ALG_OK;
  }

  // 只有一个峰
  if (1 == peak_cnt)
  {
    // 对其进行左右波谷点的查找，查找之后，按照半峰和全峰的逻辑筛选
    int point = firstpeak[0];
    int left_point;
    int right_point;
    SearchLRvalley(&left_point, &right_point, data, point, point, 2, data_len-2);
    int dis_left = abs(point - left_point);
    int dis_right = abs(point - right_point);
    if (dis_left + dis_right <= WIDTH_LLIMIT)
    {
      // 不符合宽度，不符合一个峰
      peak_cnt = 0;
      find_peak_info->peak_num = 0;
      return ALG_OK;
    }

    // 如果符合宽度，则认为是一个峰
    find_peak_info->peak_num = 1;
    find_peak_info->one_peak_info[0].start_point = left_point;
    find_peak_info->one_peak_info[0].end_point = right_point;
    find_peak_info->one_peak_info[0].peak_point = point;
    find_peak_info->one_peak_info[0].peak_value = *(data+ point);
    return ALG_OK;
  }

  // 多个峰, 两个以及两个以上的峰需要进行合并操作

  if (peak_cnt > 1)
  {
    MergeInfo merge_info[MAX_SECOND_PEAK_CNT];
    int merge_cnt = 0;
    MergePeak(merge_info, &merge_cnt, firstpeak, peak_cnt);

    SecondFindPeak(find_peak_info, data, merge_info, merge_cnt, start, end);
  }

  return ALG_OK;
}

// 按照模式计算信号
int CalculateSignal(LineResult* line_result, int line_index, int Tmodel, int Cmodel, int decimal)
{
  if (NULL == line_result)
  {
    return ALG_ERR;
  }

  double signal = 0.0;
  double TSignal = 0.0;
  double CSignal = 0.0;
  switch (Tmodel)
  {
  case 0:
  {
    TSignal = 0;  // 如果是0，失效
    break;
  }
  case 1:
  {
    TSignal = line_result->single_line_rst[0].area;
    break;
  }
  case 2:
  {
    TSignal = line_result->single_line_rst[1].area; //2
    break;
  }
  case 3:
  {
    TSignal = line_result->single_line_rst[2].area; // 3
    break;
  }
  case 4:
  {
    TSignal = line_result->single_line_rst[3].area; // 4
    break;
  }
  case 5:
  {
    TSignal = line_result->single_line_rst[4].area; // 5
    break;
  }

  case 6:
  {
    TSignal = line_result->single_line_rst[0].area + line_result->single_line_rst[1].area; // 1+2
    break;
  }
  case 7:
  {
    TSignal = line_result->single_line_rst[0].area + line_result->single_line_rst[1].area + line_result->single_line_rst[2].area; // 1+2+3
    break;
  }
  case 8:
  {
    TSignal = line_result->single_line_rst[0].area + line_result->single_line_rst[1].area + line_result->single_line_rst[2].area + line_result->single_line_rst[3].area; // 1+2+3+4
    break;
  }
  default:
  {
    TSignal = 0;
    break;
  }
  }


  switch (Cmodel)
  {
  case 0:
  {
    CSignal = 1;  // 如果是0，使用T信号
    break;
  }
  case 1:
  {
    CSignal = line_result->single_line_rst[0].area;
    break;
  }
  case 2:
  {
    CSignal = line_result->single_line_rst[1].area; //2
    break;
  }
  case 3:
  {
    CSignal = line_result->single_line_rst[2].area; // 3
    break;
  }
  case 4:
  {
    CSignal = line_result->single_line_rst[3].area; // 4
    break;
  }
  case 5:
  {
    CSignal = line_result->single_line_rst[4].area; // 5
    break;
  }
  case 6:
  {
    CSignal = line_result->single_line_rst[0].area + line_result->single_line_rst[1].area; // 1+2
    break;
  }
  case 7:
  {
    CSignal = line_result->single_line_rst[0].area + line_result->single_line_rst[1].area + line_result->single_line_rst[2].area; // 1+2+3
    break;
  }
  case 8:
  {
    CSignal = line_result->single_line_rst[0].area + line_result->single_line_rst[1].area + line_result->single_line_rst[2].area + line_result->single_line_rst[3].area; // 1+2+3+4
    break;
  }
  default:
  {
    CSignal = 1;
    break;
  }
  }
  if (0 == CSignal)
  {
    signal = 0;
  }
  else
  {
    signal = TSignal / CSignal;
  }

  memset(line_result->single_line_rst[line_index].signal, 0x00, 16);
  double fsignal = floor(signal * pow(10, decimal)) / pow(10, decimal);

  if (0 == decimal)
  {
    sprintf(line_result->single_line_rst[line_index].signal, "%.0f", signal);
  }
  else
  {
    sprintf(line_result->single_line_rst[line_index].signal, "%.4f", fsignal);
  }

  return ALG_OK;
}


///
/// @brief
///     计算面积
///
/// @param[in]  line_info       指向保存检测线结果的内存地址
/// @param[in]  data            指向保存原始数据的内存地址
/// @param[in]  coeff           校准系数
///
/// @return ALG_OK 表示成功，ALG_ERR表示成功
/// @par History:
/// @xielihui，2024年1月18日，新建函数
///
int CalculateArea(SingleLineRst *line_info, float* data, float coeff)
{
  uint32_t isize = MAX_SIGNAL_WIDTH * sizeof(float);
  float* base_line_array = (float*)Alg_malloc(isize);
  if (NULL == data || NULL == line_info || NULL == base_line_array)
  {
    return ALG_ERR;
  }

  // 计算基线，求斜率
  uint16_t cnt = line_info->signal_end - line_info->signal_start + 1;
  if (cnt > MAX_SIGNAL_WIDTH)
  {
#ifdef ALGO_DEBUG_ENABLE
    printf("signal width is larger than MAX_SIGNAL_WIDTH when %dth testline.\n", line_index + 1);
#endif
    return ALG_ERR;
  }

  // 求解基线
  float k = 0.0;
  float b = 0.0;
  float signal_start = *(data + line_info->signal_start);
  float signal_end = *(data + line_info->signal_end);

  double area = 0.0;
  double temp_area = 0.0;
  if (line_info->signal_start != line_info->signal_end)
  {
    k = (signal_start - signal_end) / ((int32_t)(line_info->signal_start) - (int32_t)(line_info->signal_end));
    b = *(data + line_info->signal_start) - k * (line_info->signal_start + 1);

    for (uint16_t i = 0; i < cnt; i++)
    {
      *(base_line_array + i) = k * (line_info->signal_start + i + 1) + b;
    }


    // 计算面积
    for (uint16_t i = line_info->signal_start; i < line_info->signal_end; i++)
    {
      temp_area = (*(data + i) + *(data + i + 1)) / 2.0 - *(base_line_array + i - line_info->signal_start);
      if (temp_area > 0)
      {
        area = area + temp_area;
      }
    }
  }
  else
  {
#ifdef ALGO_DEBUG_ENABLE
    printf("The denominator is 0 when %dth testline.\n", line_index + 1);
#endif
    area = 0.0;
  }

  line_info->area = area * coeff;

  // 释放内存
  if (NULL != base_line_array)
  {
    Alg_free(base_line_array);
    base_line_array = NULL;
  }
  return ALG_OK;
}

///
/// @brief
///     各个通道检测线信息查找
///
/// @param[in]  line_info     指向保存检测线结果的内存地址
/// @param[in]  data          指向数据的内存地址
/// @param[in]  signal_window 信号窗口
/// @param[in]  data_len      数据长度
///
/// @return ALG_OK 表示成功，ALG_ERR表示成功
/// @par History:
/// @xielihui，2024年1月18日，新建函数

int GetSignalRegion(SingleLineRst* line_info, float *data, int signal_window, int point, int data_len)
{
  // 计算信号区间
  if (NULL == line_info)
  {
    return ALG_ERR;
  }

  int signal_start;
  int signal_end;


  if (point < 0)
  {
    point = 0;
  }

  if (point > (data_len - 1))
  {
    point = data_len - 1;
  }


  if (point - signal_window < 0)
  {
    signal_start = 0;
  }
  else if (point - signal_window > (data_len - 1))
  {
    signal_start = data_len - 1;
  }
  else
  {
    signal_start = point - signal_window;
  }

  if (point + signal_window > (data_len - 1))
  {
    signal_end = data_len - 1;
  }
  else
  {
    signal_end = point + signal_window;
  }



  // 信号线赋值
  line_info->max_point = point;
  line_info->signal_start = signal_start;
  line_info->signal_end = signal_end;
  line_info->left_valley = signal_start;
  line_info->right_valley = signal_end;
  line_info->max_value = *(data + point);
  line_info->base_line = (*(data + signal_start) + *(data + signal_end)) / 2;

  return ALG_OK;
}

///
/// @brief
///     各个通道检测线信息查找
///
/// @param[in]  input           算法输入
/// @param[in]  result          算法处理结果
/// @param[in]  coef            校准系数
///
/// @return  返回函数处理的成功与否的状态
///
/// @par History:
/// @xielihui，2024年05月03日，新建函数
///
int CommonProcessLineInfo(LineResult* line_result, int line_index, int start, int end, int signal_window, float coef_value,
                          int Tmodel, int Cmodel, float gate, int flag)
{
  // 计算信号区间
  if (NULL == line_result)
  {
    return ALG_ERR;
  }
  int point = 0;
  int datacnt;
  float maxValue = 0.0;
  int maxpoint = 0;
  int prepos = 0;


  float* data = &line_result->filter_data[0];
  int data_len = line_result->length;

  SingleLineRst* line_info = NULL;
  line_info = &line_result->single_line_rst[line_index];


  // 默认多个峰，找出区间里最大的峰
  maxValue = 0.0;
  maxpoint = 0;
  datacnt = end - start + 1;
  FindMaxValue(&maxValue, &maxpoint, data + start, datacnt);
  point = maxpoint + start;

  GetSignalRegion(line_info, data, signal_window, point, data_len);
  // 计算面积
  CalculateArea(line_info, data, coef_value);

  if(0 == flag) // T峰处理
  {
    prepos = (start + end) / 2; // 预设点
    CalculateSignal(line_result, line_index, Tmodel, Cmodel, 4);
    float signal = atof(line_result->single_line_rst[line_index].signal);
    if (signal < gate)
    {
      point = prepos;
      GetSignalRegion(line_info, data, signal_window, point, data_len);

      // 计算面积
      CalculateArea(line_info, data, coef_value);

      // 计算信号
      CalculateSignal(line_result, line_index, Tmodel, Cmodel, 4);
    }
  }
  // 信号点+1，特殊处理，不能删除，解决计数起点不一致问题
  line_info->max_point = line_info->max_point + 1;
  line_info->signal_start = line_info->signal_start + 1;
  line_info->signal_end = line_info->signal_end + 1;
  line_info->left_valley = line_info->left_valley + 1;
  line_info->right_valley = line_info->right_valley + 1;
  return ALG_OK;
}

///
/// @brief
///     各个通道检测线信息查找
///
/// @param[in]  input           算法输入
/// @param[in]  result          算法处理结果
/// @param[in]  coef            校准系数
///
/// @return  返回函数处理的成功与否的状态
///
/// @par History:
/// @xielihui，2022年3月28日，新建函数
///
int FindLineInfo(LineResult* line_result, AlgInput* input, char* coef)
{
#ifdef ALGO_DEBUG_ENABLE
  printf("FindLineInfo is in progress.\n");
#endif

  if(NULL == input || NULL == line_result || NULL == coef)
  {
    return ALG_ERR;
  }
  int line_cnt = input->line_cnt;

  if (line_cnt <= 0)
  {
    return ALG_ERR;
  }

  // 校准系数映射
  uint16_t data_len = input->length;
  line_result->line_cnt = line_cnt;
  line_result->channel_cnt = input->channel_cnt;
  line_result->length = data_len;

  float coef_value = strtof(coef, NULL);

  int prepos = 0; // 预设点
  int start, end;
  int signal_window;
  int line_index;
  int Cindex;
  int dis_window = input->line_para.twindow;;
  float gate = 0.0;
  int Tmodel = 0;
  int Cmodel = 0;

  line_index = input->line_para.cid - 1; // 获取C线id
  Cindex = line_index;
  start = input->line_para.cstart - 1;
  end = start + input->line_para.paras[Cindex].dis;
  signal_window = ceil(input->line_para.paras[Cindex].signal_window / 2);
  gate = input->line_para.paras[line_index].gate;
  CommonProcessLineInfo(line_result, line_index, start, end, signal_window,coef_value, Tmodel, Cmodel, gate, 1); // C线处理逻辑

  // 再处理C前的T
  for (line_index = 0; line_index < Cindex; line_index++)
  {
    signal_window = ceil(input->line_para.paras[line_index].signal_window / 2);
    Tmodel = input->line_para.paras[line_index].Tmodel;
    Cmodel = input->line_para.paras[line_index].Cmodel;
    prepos = line_result->single_line_rst[Cindex].max_point - 1 - input->line_para.paras[line_index].dis;
    start = prepos - dis_window;
    end = prepos + dis_window;
    gate = input->line_para.paras[line_index].gate;
    CommonProcessLineInfo(line_result, line_index, start, end, signal_window, coef_value, Tmodel, Cmodel, gate, 0); // T线处理逻辑
  }
  // 再处理C后的T
  for (line_index = Cindex + 1; line_index < input->line_cnt; line_index++)
  {
    // 其它寻峰
    signal_window = ceil(input->line_para.paras[line_index].signal_window / 2);
    prepos = line_result->single_line_rst[Cindex].max_point - 1 + input->line_para.paras[line_index].dis;
    start = prepos - dis_window;
    end = prepos + dis_window;
    gate = input->line_para.paras[line_index].gate;
    Tmodel = input->line_para.paras[line_index].Tmodel;
    Cmodel = input->line_para.paras[line_index].Cmodel;
    CommonProcessLineInfo(line_result, line_index, start, end, signal_window, coef_value, Tmodel, Cmodel, gate, 0); // T线处理逻辑
  }

  // 校准特殊处理流程
  if (TCAL == input->testmode)
  {
    // 将第一条线的信号，按照模式刷新，并按照位数进行保留
    line_index = 0;
    Tmodel = input->line_para.paras[line_index].Tmodel;
    Cmodel = input->line_para.paras[line_index].Cmodel;
    CalculateSignal(line_result, line_index, Tmodel, Cmodel, input->line_para.paras[line_index].decimal);
  }
  return ALG_OK;
}
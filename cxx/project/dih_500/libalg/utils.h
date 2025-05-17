//
// Created by y on 23-8-11.
//

#ifndef RKNN_ALG_DEMO_UTILS_H
#define RKNN_ALG_DEMO_UTILS_H
#include <numeric>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <string>
#include <unistd.h>
// 体积通用设置
#define VOLUME_SIZE 300
#define VOLUME_MAX_AREA 30000

// rbc设置
#define VOLUME_RBC_KERNEL_MEDIUM 17
#define VOLUME_RBC_KERNEL_BLUR 23
#define VOLUME_RBC_DOWN_RATIO 100

// plt设置
#define VOLUME_PLT_DOWN_RATIO 100
#define VOLUME_PLT_SAMPLE_WIDTH 20
#define VOLUME_PLT_DILATE_RATIO 10000
#define VOLUME_PLT_KERNEL_BLUR 15

// rdw设置
#define RDW_SD_PERCENTAGE 0.2

// 其他
#define ADD_DENOMINATOR 1e-5   // 为避免除0，对分母进行增加

enum VOLUME_FUNC_TYPE
{
    VOLUME_TYPE_AREA = 0,
    VOLUME_TYPE_INCLINE,
};

void DrawBox(cv::Mat& img, const int& left, const int& top, const int& right, const int& bottom, const std::string& object_name, const float& prob);
void MatToData(const cv::Mat& srcImg, unsigned char* data);

// 将cunsigned char*型转为 cv::mat类型
bool DataToMat(void* data, const int& nH, const int& nW, const int& nFlag, cv::Mat& outImg);   // nH,nW为BYTE*类型图像的高和宽,nFlag为通道数
void LoadImagePath(std::string imgDirPath, std::vector<std::string>& vimgPath);
long get_memory_usage();
void SaveImage(const std::string& save_path, const cv::Mat& img);
void ReverseGroupIdx(uint32_t src_idx, int& dist_idx);


/*!
 * 根据细胞体积计算用于画线的数据
 * @param vol_v 细胞体积
 * @param result 结果
 */
void GetRbcVolumeLine(const std::vector<float>& vol_v, std::vector<float>& result);
void GetPltVolumeLine(const std::vector<float>& vol_v, std::vector<float>& result);
// 等待
void Delay(const int& time);   // time*1000为秒数


void GetRbcVolume(const std::vector<float>& vol_v,
                  const float&              fusion_rate,
                  const int&                incline_cell_nums,
                  const int&                incline_pixels,
                  const VOLUME_FUNC_TYPE&   volume_func_type,
                  float&                    value);
void GetPltVolume(const std::vector<float>& vol_v, const float& fusion_rate, float& value, std::vector<float>& all_values);
void GetRdwCv(const std::vector<float>& vol_v, const float& mcv, const float& fusion_rate, float& value, std::vector<float>& all_values);
void GetRdwSd(const std::vector<float>& vol_v, const float& fusion_rate, float& value, std::vector<float>& all_values);
/*!
 * hgb结果计算函数
 * @param r0 第0个灯照射空白区域的值
 * @param r1 第1个灯照射空白区域的值
 * @param m0 第0个灯照射样本的值
 * @param m1 第1个灯照射言本的值
 * @param hgb_value 测量结果
 */
void GetHgb(const float& r0, const float& r1, const float& m0, const float& m1, float& hgb_value);
/*!
 * 将超过max_thr的值设为max_thr,
 * @param src       输入
 * @param max_thr   阈值
 * @param dst[out]  输出
 */
void MatClip(const cv::Mat& src, const int& max_thr, cv::Mat& dst);

/*!
 * 中值滤波
 * @param src            输入
 * @param dst[out]       输出
 * @param half_wind_size 半窗口大小
 */
void MedianFilter(const std::vector<float>& src, std::vector<float>& dst, int half_wind_size);


/*!
 * 计算标准差
 * @param numbers_v         输入
 * @param std_value[out]    输出
 */
void CalculateStd(const std::vector<float>& numbers_v, float& std_value);

/*!
 * 查找指定范围内的最小值的索引序号
 * @tparam ForwardIterator 迭代器
 * @param first_iterator 起始迭代器
 * @param last_iterator  结束迭代器
 * @return 索引序号
 */
template<class ForwardIterator>
int PseudoArgmin(const ForwardIterator& first_iterator, const ForwardIterator& last_iterator);


template<class ForwardIterator>
int PseudoArgMax(const ForwardIterator& first_iterator, const ForwardIterator& last_iterator);


template<class ForwardIterator>
float PseudoMean(const ForwardIterator& first_iterator, const ForwardIterator& last_iterator)
{
    float sum = std::accumulate(first_iterator, last_iterator, 0.f);
    return sum / ((int)std::distance(first_iterator, last_iterator) + ADD_DENOMINATOR);
}
/*!
 * 计算数组与目标值差值最小的索引
 * @param data
 * @param target_value
 * @param min_idx
 */
void FindMinDifference(const std::vector<float>& data, const float& target_value, int& min_idx);
/*!
 * NMS
 * @param src
 * @param dst
 * @return
 */
void RemoveDuplicateData(const std::vector<float>& src, const float& iou_thr, float& threshed_cell_nums);

/*!
 * 搜索第一个峰
 * @param list      输入数据
 * @param index     第一个峰的序号
 * @param value     第一个峰的值
 * @param peak_nums 峰总共个数
 * @return
 */
int PeekSeekFirst(std::vector<float>& list, uint32_t* index, float* value, int& peak_nums);

class TimeMonitor
{
public:
    TimeMonitor()  = default;
    ~TimeMonitor() = default;
    int Time();
};

void ReplaceAllDistinct(std::string& str, const std::string& old_value, const std::string& new_value);
#endif   // RKNN_ALG_DEMO_UTILS_H

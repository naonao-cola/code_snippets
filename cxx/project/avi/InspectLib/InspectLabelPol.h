/*********************************************************************
 * @file   InspectLabelPol.h
 * @brief  标签、PolMark定位填充算法
 * 
 * @author 
 * @date   2024.7
 *********************************************************************/
#pragma once
#include<opencv2\opencv.hpp>
#include "Define.h"

/**
 * PolMark类型.
 */
enum PolMarkType
{
	E_POL_NUM = 0,	// polmark 数字
	E_POL_SIGN      // polmark 符号
};

/**
 * PolMark模版匹配结果.
 */
struct stPolMatchInfo
{
	double score = 0;
	std::string name;
	cv::Point loc;
	cv::Mat templateImg;
};


class InspectLabelPol
{
public:
	InspectLabelPol();
	~InspectLabelPol();

	/**
	 * 定位标签和PolMark.
	 * 
	 * @param src 输入图像(已转正）
	 * @param params 输入参数结构体，包含定位参数和polmark模版
	 * @param labelMarkInfo 输出结果，包含标签和PolMark信息
	 * @return 成功返回0，否则返回-1
	 */
	long DoFindLabelMark(cv::Mat src, STRU_LabelMarkParams& params, STRU_LabelMarkInfo& labelMarkInfo);

	/**
	 * 填充标签和PolMark区域.
	 * 
	 * @param image 输入图像(已转正）
	 * @param params 输入参数结构体，包含定位参数和polmark模版
	 * @param labelMarkInfo 通过DoFindLabelMark得到的标签和Polmark信息
	 * @return 成功返回0，否则返回错误码
	 */
	long DoFillLabelMark(cv::Mat& image, STRU_LabelMarkParams& params, const STRU_LabelMarkInfo& labelMarkInfo);

private:
	long PolMarkRec(const cv::Mat& srcImg, std::map<std::string, cv::Mat>& templates, stPolMatchInfo& polMatchInfo, PolMarkType polType);
	long DoFillLabel(cv::Mat& dstImg, const STRU_LabelMarkInfo& labelMarkInfo);
	long DoFillPolArea(cv::Mat& dstImg, STRU_LabelMarkParams& params, const STRU_LabelMarkInfo& labelMarkInfo, PolMarkType polType);

	std::vector<cv::Rect> GetPolRefRect(cv::Mat dstImg, cv::Rect& maskRect, const STRU_LabelMarkInfo& labelMarkInfo, PolMarkType polType);
	long PartialFill(cv::Mat& dstImg, cv::Rect& maskRect, std::vector<cv::Rect>& refRect);
	cv::Mat CalcProjection(const cv::Mat& srcImg, int mode);
	std::vector<int> CalcLighterPos(const cv::Mat projImg, int mode);
	double CalcMeanGrayX(const cv::Mat& src, const std::vector<int>& pos, int count, int posY);
	double CalcMeanGrayY(const cv::Mat& src, const std::vector<int>& pos, int count, int posX);
};


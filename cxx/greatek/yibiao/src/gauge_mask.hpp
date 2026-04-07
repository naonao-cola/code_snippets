/**
 * @FilePath     : /yibiao/src/gauge_mask.hpp
 * @Description  :
 * @Author       : weiwei.wang
 * @Date         : 2026-03-31 10:45:21
 * @Version      : 0.0.1
 * @LastEditors  : weiwei.wang
 * @LastEditTime : 2026-03-31 10:45:36
 * @Copyright (c) 2026 by G, All Rights Reserved.
 **/
#ifndef GAUGE_MASK_HPP
#define GAUGE_MASK_HPP

#include <opencv2/opencv.hpp>
#include <vector>

bool GaugeMasksFromImage(const cv::Mat& src,
                         cv::Mat&       scale_mask_full,
                         cv::Mat&       pointer_mask_full,
                         int            det_class,
                         int            scale_class,
                         int            pointer_class,
                         float          det_thresh,
                         float          seg_thresh);


bool GaugeMasksFromImage(
    const cv::Mat& src, cv::Mat& scale_mask_full, cv::Mat& pointer_mask_full, int scale_class, int pointer_class, float det_thresh, float seg_thresh);

bool NormalizeMasks512(const cv::Mat& scale_mask_full, const cv::Mat& pointer_mask_full, cv::Mat& scale_norm, cv::Mat& pointer_norm);

void BuildLineImageFromMask(const cv::Mat& mask_norm, unsigned char label_value, std::vector<unsigned char>& output, cv::Mat& debug_mat);

void ConvertLineTo1D(const std::vector<unsigned char>& line_output, unsigned char label_value, std::vector<unsigned int>& counts);

void MeanFiltration1D(const std::vector<unsigned int>& input, std::vector<unsigned int>& output);

struct GaugeReadResult
{
    int   scale_num        = 0;
    float scales           = 0.0f;
    float ratio            = 0.0f;
    int   pointer_location = -1;
};

bool ReadMeterFrom1D(const std::vector<unsigned int>& scale_mean, const std::vector<unsigned int>& pointer, GaugeReadResult& result);

struct GaugePipelineDebug
{
    cv::Mat                    scale_mask_full;
    cv::Mat                    pointer_mask_full;
    cv::Mat                    scale_norm;
    cv::Mat                    pointer_norm;
    cv::Mat                    scale_line_mat;
    cv::Mat                    pointer_line_mat;
    int                        line_shift = 0;
    std::vector<unsigned char> scale_line;
    std::vector<unsigned char> pointer_line;
    std::vector<unsigned int>  scale_1d;
    std::vector<unsigned int>  pointer_1d;
    std::vector<unsigned int>  scale_mean_1d;
};

bool ReadGaugePipeline(const cv::Mat&      src,
                       int                 det_class,
                       int                 scale_class,
                       int                 pointer_class,
                       float               det_thresh,
                       float               seg_thresh,
                       GaugeReadResult&    result,
                       GaugePipelineDebug* debug);

bool ReadGaugePipeline(const cv::Mat&      src,
                       int                 det_class,
                       int                 scale_class,
                       int                 pointer_class,
                       float               det_thresh,
                       float               seg_thresh,
                       bool                right_to_left,
                       GaugeReadResult&    result,
                       GaugePipelineDebug* debug);

#endif

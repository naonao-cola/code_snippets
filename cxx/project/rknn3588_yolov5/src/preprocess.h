/**
 * @FilePath     : /test03/src/preprocess.h
 * @Description  :
 * @Author       : naonao
 * @Date         : 2025-07-24 14:25:15
 * @Version      : 0.0.1
 * @LastEditors  : naonao
 * @LastEditTime : 2025-07-24 14:25:19
 * @Copyright (c) 2025 by G, All Rights Reserved.
 **/
#ifndef _RKNN_YOLOV5_DEMO_PREPROCESS_H_
#define _RKNN_YOLOV5_DEMO_PREPROCESS_H_

#include <stdio.h>
#include <iostream>
#include "im2d.h"
#include "rga.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "postprocess.h"

void letterbox(const cv::Mat &image, cv::Mat &padded_image, BOX_RECT &pads, const float scale, const cv::Size &target_size, const cv::Scalar &pad_color = cv::Scalar(128, 128, 128));

int resize_rga(rga_buffer_t &src, rga_buffer_t &dst, const cv::Mat &image, cv::Mat &resized_image, const cv::Size &target_size);

#endif //_RKNN_YOLOV5_DEMO_PREPROCESS_H_
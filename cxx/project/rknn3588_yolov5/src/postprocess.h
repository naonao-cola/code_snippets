#ifndef _RKNN_YOLOV5_DEMO_POSTPROCESS_H_
#define _RKNN_YOLOV5_DEMO_POSTPROCESS_H_

#include <stdint.h>
#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#define OBJ_NAME_MAX_SIZE 30
#define OBJ_NUMB_MAX_SIZE 100000
#define OBJ_CLASS_NUM 1
#define NMS_THRESH 0.35
#define BOX_THRESH 0.5
#define PROP_BOX_SIZE (5 + OBJ_CLASS_NUM)

typedef struct _BOX_RECT {
    int left;
    int right;
    int top;
    int bottom;
} BOX_RECT;

typedef struct __detect_result_t {
    char name[OBJ_NAME_MAX_SIZE];
    BOX_RECT box;
    float prop;
} detect_result_t;

typedef struct _detect_result_group_t {
    int id;
    int count;
    detect_result_t results[OBJ_NUMB_MAX_SIZE];
} detect_result_group_t;

int post_process(int8_t* input0, int8_t* input1, int8_t* input2, int model_in_h, int model_in_w,
    float conf_threshold, float nms_threshold, BOX_RECT pads, float scale_w, float scale_h,
    std::vector<int32_t>& qnt_zps, std::vector<float>& qnt_scales,
    detect_result_group_t* group);

int post_process__unquantization(float* input0, float* input1, float* input2, int model_in_h, int model_in_w,
    float conf_threshold, float nms_threshold, BOX_RECT pads, float scale_w, float scale_h,
    std::vector<int32_t>& qnt_zps, std::vector<float>& qnt_scales,
    detect_result_group_t* group);

void deinitPostProcess();

void RecoverBox(const cv::Size& origin_size, const cv::Size& processed_size,
    const float& src_x1, const float& src_y1, const float& src_x2,
    const float& src_y2, int& x1, int& y1, int& x2, int& y2,
    bool letterbox); 

#endif //_RKNN_YOLOV5_DEMO_POSTPROCESS_H_

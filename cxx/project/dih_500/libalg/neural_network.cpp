#include "neural_network.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <set>
#include <vector>
#include <sstream>


#include "rga.h"
#include "im2d.h"
#include "RgaUtils.h"
#include "rknn_api.h"
#include "utils.h"
#include "replace_std_string.h"
#include "imgprocess.h"
//#include "DihLog.h"
#include "algLog.h"
#include "timecnt.h"

#define USE_OPENCL 0
#if USE_OPENCL
#include "NmsCl.h"
#endif



# define NET_USE_TIMECNT 1
    /* 神经网络模型 */
    typedef struct NNetModel {
    NNetGroup_e group_id;
    NNetModID_e mod_id;
    rknn_context rknnCtx;
    int modelWidth;
    int modelHeight;
    int modelChannel;
    int netNumInput;
    int netNumOutput;
    std::vector<std::string> labels;
    std::vector<rknn_tensor_attr> output_attrs;
    uint8_t multi_label_flag;
    ResizeType resize_type;
    float nms_thr;
    float conf_thr;
    NNetTypeID_e net_type_id;
    std::vector<float> anchors;
    std::vector<float> float_params_v;
    std::vector<float> reserved_float_params;
    std::vector<std::string> reserved_string_params;
    float anchor0[6];
    float anchor1[6];
    float anchor2[6];
} NNetModel_t;
#define NNET_GROUP_ID(mod)                    ((mod)->group_id)                  // 分组ID
#define NNET_MOD_ID(mod)                      ((mod)->mod_id)                    // 模型ID
#define NNET_MOD_RKNN_CTX(mod)                ((mod)->rknnCtx)                   // RKNN上下文
#define NNET_MOD_WIDTH(mod)                   ((mod)->modelWidth)                // 模型输入宽度
#define NNET_MOD_HEIGHT(mod)                  ((mod)->modelHeight)               // 模型输入高度
#define NNET_MOD_CHANNEL(mod)                 ((mod)->modelChannel)              // 模型输入维度
#define NNET_MOD_NET_NUM_IN(mod)              ((mod)->netNumInput)               // 网络输入个数
#define NNET_MOD_NET_NUM_OUT(mod)             ((mod)->netNumOutput)              // 网络输出个数
#define NNET_MOD_LABELS(mod)                  ((mod)->labels)                    // 网络分类标签
#define NNET_MOD_RKNN_ATTRS(mod)              ((mod)->output_attrs)              // 网络输出属性
#define NNET_MOD_MULTI_LABEL_FLAG(mod)        ((mod)->multi_label_flag)          // 多标签标志
#define NNET_MOD_RESIZE_TYPE(mod)             ((mod)->resize_type)               // 模型resize方式
#define NNET_MOD_NMS_THR(mod)                 ((mod)->nms_thr)                   // 模型nms
#define NNET_MOD_CONF_THR(mod)                ((mod)->conf_thr)                  // 模型conf
#define NNET_MOD_NET_TYPE_ID(mod)             ((mod)->net_type_id)               // 模型网络类型
#define NNET_MOD_FLOAT_PARAMS_V(mod)          ((mod)->float_params_v)            // 模型特定参数,如某个模型需要特殊操作,则通过vector内的参数进行指定
#define NNET_MOD_RESERVED_FLOAT_PARAMS(mod)   ((mod)->reserved_float_params)
#define NNET_MOD_RESERVED_STRING_PARAMS(mod)  ((mod)->reserved_string_params)
/* 神经网络上下文 */
typedef struct NNetCtx {
    std::vector<NNetModel_t> mod_list;                //模型列表
} NNetCtx_t;
#define NNET_CTX_MOD_LIST(ctx)                ((ctx)->mod_list)

// 通用anchor
float anchor0[6] = {10, 13, 16, 30, 33, 23};
float anchor1[6] = {30, 61, 62, 45, 59, 119};
float anchor2[6] = {116, 90, 156, 198, 373, 326};

TimeMonitor tm;
#if USE_OPENCL
ALG_CL::NmsCl nms_cl;
#endif


// 输出节点属性信息
void dump_tensor_attr(rknn_tensor_attr *attr) {
    printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
           "zp=%d, scale=%f\n",
           attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
           attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
           get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

inline static int clamp(float val, int min, int max) {
    return val > min ? (val < max ? val : max) : min;
}

static float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1,
                              float ymax1) {
    float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
    float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
    float i = w * h;
    float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) + (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;
    return u <= 0.f ? 0.f : (i / u);
}

static int nms(int validCount, std::vector<float> &outputLocations, std::vector<int> classIds, std::vector<int> &order,
               int filterId, float threshold, const int max_det=3500) {
    int valid_obj_nums = 0;

// #pragma omp parallel for collapse(2)
//     for (int i = 0; i < validCount; ++i) {
//         if (order[i] == -1 || classIds[i] != filterId) {
//             continue;
//         }
//         int n = order[i];
//         for (int j = i + 1; j < validCount; ++j) {
//             int m = order[j];
//             if (m == -1 || classIds[i] != filterId) {
//                 continue;
//             }
//             float xmin0 = outputLocations[n * 4 + 0];
//             float ymin0 = outputLocations[n * 4 + 1];
//             float xmax0 = outputLocations[n * 4 + 0] + outputLocations[n * 4 + 2];
//             float ymax0 = outputLocations[n * 4 + 1] + outputLocations[n * 4 + 3];

//             float xmin1 = outputLocations[m * 4 + 0];
//             float ymin1 = outputLocations[m * 4 + 1];
//             float xmax1 = outputLocations[m * 4 + 0] + outputLocations[m * 4 + 2];
//             float ymax1 = outputLocations[m * 4 + 1] + outputLocations[m * 4 + 3];

//             float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

//             if (iou > threshold) {
// #pragma omp critical
//                 {
//                   order[j] = -1;
//                 }
//             }

//         }
//     }
//     return 0;

    // 预计算所有边界框的坐标，避免重复计算
    std::vector<float> xmin(validCount);
    std::vector<float> ymin(validCount);
    std::vector<float> xmax(validCount);
    std::vector<float> ymax(validCount);

    // 标记数组，用于快速判断是否已被抑制
    std::vector<bool> suppressed(validCount, false);

// 预计算所有边界框的坐标
#pragma omp parallel for
    for (int i = 0; i < validCount; ++i) {
        if (order[i] == -1 || classIds[i] != filterId) {
            order[i] = -1;   // 确保无效检测被正确标记
            continue;
        }
        int n   = order[i];
        xmin[i] = outputLocations[n * 4 + 0];
        ymin[i] = outputLocations[n * 4 + 1];
        xmax[i] = xmin[i] + outputLocations[n * 4 + 2];
        ymax[i] = ymin[i] + outputLocations[n * 4 + 3];
    }

    // 优化的NMS主循环
    for (int i = 0; i < validCount; ++i) {
        if (order[i] == -1)
            continue;
        int n = order[i];
        // 检查是否已被前面的循环抑制
        if (suppressed[i]) {
            order[i] = -1;
            continue;
        }
        // 只处理当前类别
        if (classIds[i] != filterId) {
            order[i] = -1;
            continue;
        }
        // 并行处理后续边界框
#pragma omp parallel for
        for (int j = i + 1; j < validCount; ++j) {
            if (order[j] == -1 || suppressed[j] || classIds[j] != filterId)
                continue;
            float overlap = CalculateOverlap(xmin[i], ymin[i], xmax[i], ymax[i], xmin[j], ymin[j], xmax[j], ymax[j]);
            if (overlap > threshold) {
                // 使用原子操作代替临界区，减少竞争
                suppressed[j] = true;
            }
        }
    }

    // 应用抑制标记
    for (int i = 0; i < validCount; ++i) {
        if (suppressed[i]) {
            order[i] = -1;
        }
    }

    return 0;
}

static int quick_sort_indice_inverse(std::vector<float> &input, int left, int right, std::vector<int> &indices) {
    float key;
    int key_index;
    int low = left;
    int high = right;
    if (left < right) {
        key_index = indices[left];
        key = input[left];
        while (low < high) {
            while (low < high && input[high] <= key) {
                high--;
            }
            input[low] = input[high];
            indices[low] = indices[high];
            while (low < high && input[low] >= key) {
                low++;
            }
            input[high] = input[low];
            indices[high] = indices[low];
        }
        input[low] = key;
        indices[low] = key_index;
        quick_sort_indice_inverse(input, left, low - 1, indices);
        quick_sort_indice_inverse(input, low + 1, right, indices);
    }
    return low;
}

static float sigmoid(float x) {
    return 1.0 / (1.0 + expf(-x));
}

static float unsigmoid(float y) {
    return -1.0 * logf((1.0 / y) - 1.0);
}

inline static int32_t __clip(float val, float min, float max) {
    float f = val <= min ? min : (val >= max ? max : val);
    return f;
}

static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale) {
    float dst_val = (f32 / scale) + zp;
    int8_t res = (int8_t) __clip(dst_val, -128, 127);
    return res;
}

static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) { return ((float) qnt - (float) zp) * scale; }

static int process(int8_t *input, float *anchor, int grid_h, int grid_w, int height, int width, int stride,
                   std::vector<float> &boxes, std::vector<float> &objProbs, std::vector<int> &classId, float threshold,
                   int32_t zp, float scale, int obj_class_num) {
    ALGLogInfo << "process 量化的模型 后处理 节点开始 "<<__FUNCTION__ <<" "<<__LINE__;
    int prop_box_size = obj_class_num + 5;   // 种类+5
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    float thres = unsigmoid(threshold);
    int8_t thres_i8 = qnt_f32_to_affine(thres, zp, scale);
//    int8_t thres_i8   = qnt_f32_to_affine(threshold, zp, scale);
    for (int a = 0; a < 3; a++) {
        for (int i = 0; i < grid_h; i++) {
            for (int j = 0; j < grid_w; j++) {
                int8_t box_confidence = input[(prop_box_size * a + 4) * grid_len + i * grid_w + j];
                if (box_confidence >= thres_i8) {
                    int offset = (prop_box_size * a) * grid_len + i * grid_w + j;
                    int8_t *in_ptr = input + offset;
                    float box_x = sigmoid(deqnt_affine_to_f32(*in_ptr, zp, scale)) * 2.0 - 0.5;
                    float box_y = sigmoid(deqnt_affine_to_f32(in_ptr[grid_len], zp, scale)) * 2.0 - 0.5;
                    float box_w = sigmoid(deqnt_affine_to_f32(in_ptr[2 * grid_len], zp, scale)) * 2.0;
                    float box_h = sigmoid(deqnt_affine_to_f32(in_ptr[3 * grid_len], zp, scale)) * 2.0;
                    box_x = (box_x + j) * (float) stride;
                    box_y = (box_y + i) * (float) stride;
                    box_w = box_w * box_w * (float) anchor[a * 2];
                    box_h = box_h * box_h * (float) anchor[a * 2 + 1];
                    box_x -= (box_w / 2.0);
                    box_y -= (box_h / 2.0);

                    int8_t maxClassProbs = in_ptr[5 * grid_len];
                    int maxClassId = 0;
                    for (int k = 1; k < obj_class_num; ++k) {
                        int8_t prob = in_ptr[(5 + k) * grid_len];
                        if (prob > maxClassProbs) {
                            maxClassId = k;
                            maxClassProbs = prob;
                        }
                    }
                    if (maxClassProbs > thres_i8) {
                        objProbs.push_back(sigmoid(deqnt_affine_to_f32(maxClassProbs, zp, scale)) *
                                           sigmoid(deqnt_affine_to_f32(box_confidence, zp, scale)));
                        classId.push_back(maxClassId);
                        validCount++;
                        boxes.push_back(box_x);
                        boxes.push_back(box_y);
                        boxes.push_back(box_w);
                        boxes.push_back(box_h);
                    }
                }
            }
        }
    }
    ALGLogInfo << "process 量化的模型 后处理 节点开始 "<<__FUNCTION__ <<" "<<__LINE__;
    return validCount;
}


static int process_unquantization(float *input, float *anchor, int grid_h, int grid_w, int height, int width, int stride,
                   std::vector<float> &boxes, std::vector<float> &objProbs, std::vector<int> &classId, float threshold,
                   int32_t zp, float scale, int obj_class_num) {

  ALGLogInfo << "process_unquantization 未量化的模型 后处理 节点开始 "<<__FUNCTION__ <<" "<<__LINE__;

  int prop_box_size = obj_class_num + 5;   // 种类+5
  int validCount = 0;
  int grid_len = grid_h * grid_w;
  float thres = unsigmoid(threshold);
  //    int8_t thres_i8   = qnt_f32_to_affine(threshold, zp, scale);
  for (int a = 0; a < 3; a++) {
    for (int i = 0; i < grid_h; i++) {
      for (int j = 0; j < grid_w; j++) {
        float box_confidence = input[(prop_box_size * a + 4) * grid_len + i * grid_w + j];
        if (box_confidence >= thres) {
          int offset = (prop_box_size * a) * grid_len + i * grid_w + j;
          float *in_ptr = input + offset;
          float box_x = sigmoid(*in_ptr) * 2.0 - 0.5;
          float box_y = sigmoid(in_ptr[grid_len]) * 2.0 - 0.5;
          float box_w = sigmoid(in_ptr[2 * grid_len]) * 2.0;
          float box_h = sigmoid(in_ptr[3 * grid_len]) * 2.0;
          box_x = (box_x + j) * (float) stride;
          box_y = (box_y + i) * (float) stride;
          box_w = box_w * box_w * (float) anchor[a * 2];
          box_h = box_h * box_h * (float) anchor[a * 2 + 1];
          box_x -= (box_w / 2.0);
          box_y -= (box_h / 2.0);

          float maxClassProbs = in_ptr[5 * grid_len];
          int maxClassId = 0;
          for (int k = 1; k < obj_class_num; ++k) {
            float prob = in_ptr[(5 + k) * grid_len];
            if (prob > maxClassProbs) {
              maxClassId = k;
              maxClassProbs = prob;
            }
          }
          if (maxClassProbs > thres) {
            objProbs.push_back(sigmoid(maxClassProbs) *
                               sigmoid(box_confidence));
            classId.push_back(maxClassId);
            validCount++;
            boxes.push_back(box_x);
            boxes.push_back(box_y);
            boxes.push_back(box_w);
            boxes.push_back(box_h);
          }
        }
      }
    }
  }
   ALGLogInfo << "process_unquantization 未量化的模型 后处理 节点结束 "<<__FUNCTION__ <<" "<<__LINE__;
  return validCount;
}





#if USE_OPENCL

static int
yolo_post_process_unquantization(float *input0, float *input1, float *input2, int model_in_h, int model_in_w, float conf_threshold,
                                 float nms_threshold, float scale_w, float scale_h, std::vector<int32_t> &qnt_zps,
                                 std::vector<float> &qnt_scales, std::vector<std::string> &labels, std::list<NNetResult_t> &opt,
                                 const int &img_height_ori, const int &img_width_ori, const ResizeType &resize_type,
                                 float* target_anchor0, float* target_anchor1, float* target_anchor2) {

  std::vector<float> filterBoxes;
  std::vector<float> objProbs;
  std::vector<int> classId;
  int obj_class_num = labels.size(); // 识别的种类

  // stride 8
  int stride0 = 8;
  int grid_h0 = model_in_h / stride0;
  int grid_w0 = model_in_w / stride0;
  int validCount0 = 0;

  ALGLogInfo << "yolo_post_process_unquantization,quick_sort_indice_invers 开始处理未量化的模型 "<<__FUNCTION__ <<"  "<<__LINE__;
  validCount0 = process_unquantization(input0, target_anchor0, grid_h0, grid_w0, model_in_h, model_in_w, stride0, filterBoxes,
                                       objProbs,
                                       classId, conf_threshold, qnt_zps[0], qnt_scales[0], obj_class_num);

  // stride 16
  int stride1 = 16;
  int grid_h1 = model_in_h / stride1;
  int grid_w1 = model_in_w / stride1;
  int validCount1 = 0;
  validCount1 = process_unquantization(input1, target_anchor1, grid_h1, grid_w1, model_in_h, model_in_w, stride1, filterBoxes,
                                       objProbs,
                                       classId, conf_threshold, qnt_zps[1], qnt_scales[1], obj_class_num);

  // stride 32
  int stride2 = 32;
  int grid_h2 = model_in_h / stride2;
  int grid_w2 = model_in_w / stride2;
  int validCount2 = 0;
  validCount2 = process_unquantization(input2, target_anchor2, grid_h2, grid_w2, model_in_h, model_in_w, stride2, filterBoxes,
                                       objProbs,
                                       classId, conf_threshold, qnt_zps[2], qnt_scales[2], obj_class_num);

  int validCount = validCount0 + validCount1 + validCount2;
  // no object detect
  if (validCount <= 0) {
    return 0;
  }
  std::vector<int> indexArray;
  for (int i = 0; i < validCount; ++i) {
    indexArray.push_back(i);
  }

  quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);

  ALGLogInfo << "yolo_post_process_unquantization,quick_sort_indice_inverse 快速排序之后"<<__FUNCTION__ <<" "<<__LINE__;

  std::set<int> class_set(std::begin(classId), std::end(classId));
  #if(NET_USE_TIMECNT)
    TimeCnt_Start("nms");
  #endif
  float box_sorted[indexArray.size()*4];
  float box_expand_cls[indexArray.size()*4];
  const int max_wh = std::max(img_height_ori, img_width_ori);

  ALGLogInfo << "yolo_post_process_unquantization,quick_sort_indice_inverse 进入 openmp 多线程处理"<<__FUNCTION__ <<" "<<__LINE__;

#pragma omp parallel for
  for(int i =0;i<indexArray.size();i++){
    int target_position = i*4;
    int ori_position = indexArray[i];
    memcpy(box_sorted+target_position,&filterBoxes[0]+4*ori_position,sizeof(float)*4);
    box_sorted[target_position+2] += box_sorted[target_position];//x1,y1,w,h->x1,y1, x2,y2
    box_sorted[target_position+3] += box_sorted[target_position+1];

    // 根据box类别对nms中的box进行偏移, 与yolo逻辑一致
    int box_cls = classId[i]+1;
    int expand_ratio = box_cls * max_wh;
    box_expand_cls[target_position] = box_sorted[target_position]*expand_ratio;
    box_expand_cls[target_position+1] = box_sorted[target_position+1]*expand_ratio;
    box_expand_cls[target_position+2] = box_sorted[target_position+2]*expand_ratio;
    box_expand_cls[target_position+3] = box_sorted[target_position+3]*expand_ratio;
  }

  ALGLogInfo << "yolo_post_process_unquantization,quick_sort_indice_inverse  opencl 之前 "<<__FUNCTION__ <<" "<<__LINE__;

  std::vector<int> keep_box;
  // opencl nms
  if(nms_cl.Forward(box_expand_cls, validCount, nms_threshold,keep_box)){
    ALGLogError<<"Failed to run nms cl";
    return -2;
  }
  ALGLogInfo << "yolo_post_process_unquantization,quick_sort_indice_inverse  opencl 之后 "<<__FUNCTION__ <<" "<<__LINE__;

  #if(NET_USE_TIMECNT)
    TimeCnt_End("nms");
  #endif
  ALGLogInfo << "yolo_post_process_unquantization,quick_sort_indice_inverse  生成结果 "<<__FUNCTION__ <<" "<<__LINE__;

  NNetResult_t item;
  for (int i = 0; i < validCount; ++i) {
    if (keep_box[i] == -1) {
      continue;
    }
    int n =i;

    float x1 = box_sorted[n * 4 + 0];
    float y1 = box_sorted[n * 4 + 1];
    float x2 = box_sorted[n * 4 + 2];
    float y2 = box_sorted[n * 4 + 3];
    int id = 0;
    float obj_conf = objProbs[n];

    if (obj_conf < conf_threshold) {
      continue;
    }

    item.box.left = (int) (x1);
    item.box.top = (int) (y1);
    item.box.right = (int) (x2);
    item.box.bottom = (int) (y2);
    item.box.prop = obj_conf;
    item.box.name = labels.at(id).c_str();

    RecoverBox(cv::Size(img_width_ori, img_height_ori),
               cv::Size(model_in_w, model_in_h),
               x1, y1,x2,y2,
               item.box.left, item.box.top, item.box.right, item.box.bottom,
               resize_type==LETTERBOX);

    item.write_rect_box = true;
    opt.push_back(item);
  }
  ALGLogInfo << "yolo_post_process_unquantization,quick_sort_indice_inverse  记录结果之后"<<__FUNCTION__ <<" "<<__LINE__;
  return 0;
}
#else
static int
yolo_post_process_unquantization(float *input0, float *input1, float *input2, int model_in_h, int model_in_w, float conf_threshold,
                  float nms_threshold, float scale_w, float scale_h, std::vector<int32_t> &qnt_zps,
                  std::vector<float> &qnt_scales, std::vector<std::string> &labels, std::list<NNetResult_t> &opt,
                  const int &img_height_ori, const int &img_width_ori, const ResizeType &resize_type,
                                 float* target_anchor0, float* target_anchor1, float* target_anchor2) {

std::vector<float> filterBoxes;
std::vector<float> objProbs;
std::vector<int> classId;
int obj_class_num = labels.size(); // 识别的种类

// stride 8
int stride0 = 8;
int grid_h0 = model_in_h / stride0;
int grid_w0 = model_in_w / stride0;
int validCount0 = 0;

ALGLogInfo << "yolo_post_process_unquantization,quick_sort_indice_invers 开始处理未量化的模型 "<<__FUNCTION__ <<"  "<<__LINE__;

validCount0 = process_unquantization(input0, target_anchor0, grid_h0, grid_w0, model_in_h, model_in_w, stride0, filterBoxes,
                                     objProbs,
                                     classId, conf_threshold, qnt_zps[0], qnt_scales[0], obj_class_num);

// stride 16
int stride1 = 16;
int grid_h1 = model_in_h / stride1;
int grid_w1 = model_in_w / stride1;
int validCount1 = 0;
validCount1 = process_unquantization(input1, target_anchor1, grid_h1, grid_w1, model_in_h, model_in_w, stride1, filterBoxes,
                                     objProbs,
                                     classId, conf_threshold, qnt_zps[1], qnt_scales[1], obj_class_num);

// stride 32
int stride2 = 32;
int grid_h2 = model_in_h / stride2;
int grid_w2 = model_in_w / stride2;
int validCount2 = 0;
validCount2 = process_unquantization(input2, target_anchor2, grid_h2, grid_w2, model_in_h, model_in_w, stride2, filterBoxes,
                                     objProbs,
                                     classId, conf_threshold, qnt_zps[2], qnt_scales[2], obj_class_num);

int validCount = validCount0 + validCount1 + validCount2;
// no object detect
if (validCount <= 0) {
  return 0;
}
std::vector<int> indexArray;
for (int i = 0; i < validCount; ++i) {
  indexArray.push_back(i);
}
//    std::cout<<"valid nums "<<validCount<<std::endl;
quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);

ALGLogInfo << "yolo_post_process_unquantization,quick_sort_indice_inverse 快速排序之后"<<__FUNCTION__ <<" "<<__LINE__;

std::set<int> class_set(std::begin(classId), std::end(classId));

#if (NET_USE_TIMECNT)
  TimeCnt_Start("nms");
#endif

  ALGLogInfo << "yolo_post_process_unquantization,quick_sort_indice_inverse nms开始,validCount: " << validCount
             << " filterBoxes: " << filterBoxes.size() << " classId: " << classId.size() << " indexArray: " << indexArray.size()
             << " nms_threshold: " << nms_threshold;

  for (auto c : class_set) {
      nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold);
}
ALGLogInfo << "yolo_post_process_unquantization,quick_sort_indice_inverse nms 之后" << __FUNCTION__ << " " << __LINE__;


#if (NET_USE_TIMECNT)
TimeCnt_End("nms");
#endif

// box valid detect target
NNetResult_t item;
for (int i = 0; i < validCount; ++i) {
  if (indexArray[i] == -1) {
    continue;
  }

  int n = indexArray[i];
  float x1 = filterBoxes[n * 4 + 0];
  float y1 = filterBoxes[n * 4 + 1];
  float x2 = x1 + filterBoxes[n * 4 + 2];
  float y2 = y1 + filterBoxes[n * 4 + 3];
  int id = classId[n];
  float obj_conf = objProbs[i];

  if (obj_conf < conf_threshold) {
    continue;
  }


  //    item.box.left = (int)(clamp(x1, 0, model_in_w) / scale_w);
  //    item.box.top = (int)(clamp(y1, 0, model_in_h) / scale_h);
  //    item.box.right = (int)(clamp(x2, 0, model_in_w) / scale_w);
  //    item.box.bottom = (int)(clamp(y2, 0, model_in_h) / scale_h);
  item.box.left = (int) (x1);
  item.box.top = (int) (y1);
  item.box.right = (int) (x2);
  item.box.bottom = (int) (y2);
  item.box.prop = obj_conf;
  item.box.name    = labels.at(id).c_str();

  RecoverBox(cv::Size(img_width_ori, img_height_ori),
             cv::Size(model_in_w, model_in_h),
             x1, y1,x2,y2,
             item.box.left, item.box.top, item.box.right, item.box.bottom,
             resize_type==LETTERBOX);

  item.write_rect_box = true;
  opt.push_back(item);
}

return 0;
}
#endif

static int
yolo_post_process(int8_t *input0, int8_t *input1, int8_t *input2, int model_in_h, int model_in_w, float conf_threshold,
             float nms_threshold, float scale_w, float scale_h, std::vector<int32_t> &qnt_zps,
             std::vector<float> &qnt_scales, std::vector<std::string> &labels, std::list<NNetResult_t> &opt,
             const int &img_height_ori, const int &img_width_ori, const ResizeType &resize_type,
                  float* target_anchor0, float* target_anchor1, float* target_anchor2){
    std::vector<float> filterBoxes;
    std::vector<float> objProbs;
    std::vector<int> classId;
    int obj_class_num = labels.size(); // 识别的种类

    ALGLogInfo << "yolo_post_process 开始处理量化的模型 "<<__FUNCTION__ <<"  "<<__LINE__;

    // stride 8
    int stride0 = 8;
    int grid_h0 = model_in_h / stride0;
    int grid_w0 = model_in_w / stride0;
    int validCount0 = 0;

    validCount0 = process(input0, target_anchor0, grid_h0, grid_w0, model_in_h, model_in_w, stride0, filterBoxes,
                          objProbs,
                          classId, conf_threshold, qnt_zps[0], qnt_scales[0], obj_class_num);

    // stride 16
    int stride1 = 16;
    int grid_h1 = model_in_h / stride1;
    int grid_w1 = model_in_w / stride1;
    int validCount1 = 0;
    validCount1 = process(input1, target_anchor1, grid_h1, grid_w1, model_in_h, model_in_w, stride1, filterBoxes,
                          objProbs,
                          classId, conf_threshold, qnt_zps[1], qnt_scales[1], obj_class_num);

    // stride 32
    int stride2 = 32;
    int grid_h2 = model_in_h / stride2;
    int grid_w2 = model_in_w / stride2;
    int validCount2 = 0;
    validCount2 = process(input2, target_anchor2, grid_h2, grid_w2, model_in_h, model_in_w, stride2, filterBoxes,
                          objProbs,
                          classId, conf_threshold, qnt_zps[2], qnt_scales[2], obj_class_num);

    int validCount = validCount0 + validCount1 + validCount2;
    // no object detect
    if (validCount <= 0) {
        return 0;
    }

    std::vector<int> indexArray;
    for (int i = 0; i < validCount; ++i) {
        indexArray.push_back(i);
    }
    ALGLogInfo << "yolo_post_process,quick_sort_indice_inverse  nms 之前 "<< __FUNCTION__ << " " << __LINE__;
    quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);

    std::set<int> class_set(std::begin(classId), std::end(classId));

    for (auto c: class_set) {
        nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold);
    }

    ALGLogInfo << "yolo_post_process,quick_sort_indice_inverse  nms 之后 "<< __FUNCTION__ << " " << __LINE__;
    /* box valid detect target */
    NNetResult_t item;
    for (int i = 0; i < validCount; ++i) {
        if (indexArray[i] == -1) {
            continue;
        }

        int n = indexArray[i];
        float x1 = filterBoxes[n * 4 + 0];
        float y1 = filterBoxes[n * 4 + 1];
        float x2 = x1 + filterBoxes[n * 4 + 2];
        float y2 = y1 + filterBoxes[n * 4 + 3];
        int id = classId[n];
        float obj_conf = objProbs[i];

        if (obj_conf < conf_threshold) {
            continue;
        }


//    item.box.left = (int)(clamp(x1, 0, model_in_w) / scale_w);
//    item.box.top = (int)(clamp(y1, 0, model_in_h) / scale_h);
//    item.box.right = (int)(clamp(x2, 0, model_in_w) / scale_w);
//    item.box.bottom = (int)(clamp(y2, 0, model_in_h) / scale_h);
        item.box.left = (int) (x1);
        item.box.top = (int) (y1);
        item.box.right = (int) (x2);
        item.box.bottom = (int) (y2);
        item.box.label = id;
        item.box.prop = obj_conf;
        item.box.name = labels.at(id).c_str();

        RecoverBox(cv::Size(img_width_ori, img_height_ori),
                   cv::Size(model_in_w, model_in_h),
                   x1, y1,x2,y2,
                   item.box.left, item.box.top, item.box.right, item.box.bottom,
                   resize_type==LETTERBOX);

        item.write_rect_box = true;
        opt.push_back(item);
    }
    ALGLogInfo << "yolo_post_process,quick_sort_indice_inverse  记录结果 之后"<< __FUNCTION__ << " " << __LINE__;
    return 0;
}

/*!
 * YOLO分类任务后处理
 * @param input0 rknn结果
 * @param qnt_zps zp
 * @param qnt_scales scales
 * @param labels 标签
 * @param[out] opt 结果
 */
void PostProcessCls(float *input0,  std::vector<int32_t> &qnt_zps,
                           std::vector<float> &qnt_scales,
                           std::vector<std::string> &labels,
                           std::list<NNetResult_t> &opt_v){
  const int cls_nums = labels.size();
  NNetResult_t item;
  for (int i =0; i<cls_nums; ++i){
    float prob = input0[i];
//    float prob_f =  deqnt_affine_to_f32(prob, qnt_zps[0], qnt_scales[0]);
    float prob_f = prob;
    item.category_v.push_back(prob_f);

  }

  opt_v.push_back(item);
}





/*!
 * 倾斜红细胞检测模型后处理, 以onnx输出[1,21504, 8] [1,15, 21504]为例
 * @param input0 [1,21504, 8], anchor回归后坐标, 21504个anchor, 8个坐标, 坐标已映射回原图,
 * @param input1 [1,15, 21504], anchor对应类别概率, 21504个anchor, 15个类别
 * @param conf_thr 置信度
 * @param opt_v 结果
 */
void PostProcessInclineRbc(float *input0, float  *input1,
                           const float& nms_thr, const float& conf_thr,
                           std::list<NNetResult_t> &opt_v){
  //21504个anchor, 注:该值随输入图像大小改变,调整模型输入图像大小时应当对该值进行调整
  //需要处理的类别idx
  std::vector<int> target_category_v{13};
  int anchor_nums = 21504;
  int box_cord_wh_nums = 8;//ppyoloe-r使用4个点,即8个xy保存poly坐标
  NNetResult_t net_result;

  std::vector<float> filterBoxes;
  std::vector<float> filterPoly;
  std::vector<float> objProbs;
  std::vector<int> classId;
  int box_idx = 0;
  //筛选thr
  for(int a=0; a<anchor_nums; ++a){
    for(int c=0; c<target_category_v.size(); ++c){
      float box_conf = input1[target_category_v[c]*anchor_nums+a];
      if(box_conf> conf_thr){
/*        net_result.polygon_v.push_back(target_category_v[c]); // 类别
        net_result.polygon_v.push_back(box_conf); //置信度




        //放入坐标
        for(int xy=0; xy<box_cord_wh_nums; ++xy){
          std::cout<<" "<<input0[a*box_cord_wh_nums+xy];
          net_result.polygon_v.push_back(int(input0[a*box_cord_wh_nums+xy]));
        }
        opt_v.push_back(net_result);*/
      classId.push_back(target_category_v[c]);
      objProbs.push_back(box_conf);
      for(int xy=0; xy<box_cord_wh_nums; ++xy){
          filterPoly.push_back(int(input0[a*box_cord_wh_nums+xy]));
        }
      std::vector<float> polygon_v(filterPoly.begin()+box_idx, filterPoly.begin()+box_idx+box_cord_wh_nums);
      //polygon 转 box
      std::vector<float> horizontal_points{polygon_v[0], polygon_v[2], polygon_v[4], polygon_v[6]};
      std::vector<float> vertical_points{polygon_v[1], polygon_v[3], polygon_v[5], polygon_v[7]};
      float left = *std::min_element(horizontal_points.begin(), horizontal_points.end());
      float top = *std::min_element(vertical_points.begin(), vertical_points.end());
      float right = *std::max_element(horizontal_points.begin(), horizontal_points.end());
      float bottom = *std::max_element(vertical_points.begin(), vertical_points.end());
      //x,y,w,h
      filterBoxes.push_back(left);
      filterBoxes.push_back(top);
      filterBoxes.push_back(right-left);
      filterBoxes.push_back(bottom-top);

      box_idx += box_cord_wh_nums;
      }

    }
  }
  //nms

  int validCount = classId.size();
  // no object detect
  if (validCount <= 0) {
    return ;
  }

  std::vector<int> indexArray;
  for (int i = 0; i < validCount; ++i) {
    indexArray.push_back(i);
  }

  quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);

  std::set<int> class_set(std::begin(classId), std::end(classId));

  for (auto c: class_set) {
    nms(validCount, filterBoxes, classId, indexArray, c, nms_thr);
  }

  /* box valid detect target */
  NNetResult_t item;
  for (int i = 0; i < validCount; ++i) {
    if (indexArray[i] == -1) {
      continue;
    }
    int n = indexArray[i];

    float obj_conf = objProbs[i];

    if (obj_conf < conf_thr) {
      continue;
    }
    float x1 = filterBoxes[n * 4 + 0];
    float y1 = filterBoxes[n * 4 + 1];
    float x2 = x1 + filterBoxes[n * 4 + 2];
    float y2 = y1 + filterBoxes[n * 4 + 3];
    int id = classId[n];

    std::vector<float> item_poly(filterPoly.begin()+n*box_cord_wh_nums, filterPoly.begin()+n*box_cord_wh_nums+box_cord_wh_nums);

    item.box.left = (int) (x1);
    item.box.top = (int) (y1);
    item.box.right = (int) (x2);
    item.box.bottom = (int) (y2);
    item.box.prop = obj_conf;
    item.box.name = "0";
    item.polygon_v = item_poly;
    item.write_rect_box = true;
    opt_v.push_back(item);
  }

}






//获取模型分割结果
void ProcessSeg(float *input0, const int& category_nums,
                const int& model_input_height, const int& model_input_width,
                const int& img_ori_height, const int& img_ori_width,
                std::vector<cv::Mat>& result){
  for(int i=0; i<category_nums; ++i){
    //CV_32FC1 CV_8SC1
    cv::Mat one_result(model_input_height, model_input_width, CV_32FC1, input0+i*(model_input_height*model_input_width));
    cv::Mat recovered_img;
    RecoverSeg(cv::Size(img_ori_width, img_ori_height),
               cv::Size(model_input_width, model_input_height),
               one_result, recovered_img);
    result.push_back(recovered_img);
  }
}


/*!
 * 倾斜红细胞分割模型后处理,以onnx输出[1,2,1024,1024]为例
 * @param input0 [1,2,1024,1024],为输出mask,2为类别,1024为mask高,宽
 * @param category_nums 模型类别数量
 * @param height mask高
 * @param width mask宽
 * @param pred_mask 指定类别的mask图, {0,1}
 */
void PostProcessSegRbc(float *input0, const int& category_nums,
                       const int& model_input_height, const int& model_input_width,
                       const int& img_ori_height, const int& img_ori_width,
                       std::list<NNetResult_t> &opt_v){
  //  std::vector<cv::Mat> mask_result;
  //  ProcessSeg(input0, category_nums,
  //             model_input_height, model_input_width,
  //             img_ori_height, img_ori_width,
  //             mask_result);
  //  //模型含有0,1两个类别
  //  cv::Mat back_prob(mask_result[0]);
  //  cv::Mat cell_prob(mask_result[1]);
  //  cv::Mat pred_mask;
  //  pred_mask = cell_prob>back_prob;
  //  pred_mask.convertTo(pred_mask, CV_8UC1, 1/255.0);
  //  NNetResult_t net_result;
  //  net_result.seg_v.push_back(pred_mask);
  //  opt_v.push_back(net_result);
  std::vector<cv::Mat> mask_result;
  ProcessSeg(input0, category_nums,
             model_input_height, model_input_width,
             img_ori_height, img_ori_width,
             mask_result);

  NNetResult_t net_result;
  net_result.seg_v = mask_result;
  opt_v.push_back(net_result);
}







/**
* 查找模型
* @param  ctx		神经网络上下文
* @param  mod_id	模型ID
* @return
 */
NNetModel_t *NNet_FindModel(NNetCtx_t *ctx, NNetGroup_e group_id, NNetModID_e mod_id) {
    for (uint32_t idx = 0; idx < NNET_CTX_MOD_LIST(ctx).size(); idx++) {
        NNetModel_t *mod = &NNET_CTX_MOD_LIST(ctx).at(idx);
        if (NNET_GROUP_ID(mod) == group_id && NNET_MOD_ID(mod) == mod_id) {
            return mod;
        }
    }
    return NULL;
}

#define CL_DIR_PATH "/alg/cl"
#define CL_DOC_NMS  "/nms.cl"
/**
* 神经网络初始化
* @param  none
* @return 神经网络上下文ID @ref NNetCtxID_t
 */
NNetCtxID_t NNet_Init(const std::string& cfg_path) {
    NNetCtx_t *ctx = (NNetCtx_t *) new NNetCtx_t;
    if (ctx == NULL) {
        return NULL;
    }
#if(NET_USE_TIMECNT)
    TimeCnt_Init("pre", 1);
    TimeCnt_Init("inf", 1);
    TimeCnt_Init("post", 1);
    TimeCnt_Init("nms", 1);
    TimeCnt_Init("nms_recover", 1);
#endif
#if USE_OPENCL
    std::stringstream ss;
    ss<<cfg_path<<CL_DIR_PATH<<CL_DOC_NMS;
    if(nms_cl.Init(ss.str())){
      ALGLogError<<"Failed to init nms cl"<<std::endl;
      return nullptr;
    }
#endif
    return (NNetCtxID_t) ctx;

}

/**
* 神经网络去初始化
* @param  ctx_id		神经网络上下文ID
* @return 0 success other fail
 */
int NNet_DeInit(NNetCtxID_t ctx_id) {
    NNetCtx_t *ctx = (NNetCtx_t *) ctx_id;
    // 回收rknn
    rknn_context rknn_ctx;
    int ret = 0;
    for (auto &mode: ctx->mod_list) {
        rknn_ctx = mode.rknnCtx;
        ret = rknn_destroy(rknn_ctx);
        if (ret != 0) {
            ALGLogError<<"Failed to destroy rknn model";
            return ret;
        }
    }
    // 回收opencl
#if USE_OPENCL
    nms_cl.DeInit();
#endif
    delete ctx;
    return 0;
}

static int NNet_MakeLabelsList(std::vector<std::string> &list, uint8_t *labels_data, uint32_t labels_size) {
    if (labels_data == NULL || labels_size == 0) {
        return -1;
    }
    for (uint32_t idx = 0; idx < labels_size; idx++) {
        if (labels_data[idx] == '\r' || labels_data[idx] == '\n') {
            labels_data[idx] = 0;
        }
    }
    for (uint32_t idx = 0; idx < labels_size; idx++) {
        if (labels_data[idx] != 0) {
            if (idx == 0 || labels_data[idx - 1] == 0) {
                std::string labels_name = (char *) ((long) labels_data + (long) idx);
                list.push_back(labels_name);
            }
        }
    }
    return 0;
}


int NNet_BindNpuCore(rknn_context *rknn_ctx, const NNetModID_e& mod_id,
                     const int& width, const int& height){
  int ret;
  rknn_core_mask core_mask= RKNN_NPU_CORE_0;
  if(mod_id==NNetModID_e::NNET_MODID_AI_CLARITY_FAR_NEAR||
      mod_id==NNetModID_e::NNET_MODID_AI_CLARITY_BASO_FAR_NEAR||
      mod_id==NNetModID_e::NNET_MODID_AI_CLARITY_COARSE){//清晰度算法绑定至2号核心
    core_mask = RKNN_NPU_CORE_2;
  }else if(width>=1024){//模型输入大于1024的,绑定至0,1号核心
    core_mask = RKNN_NPU_CORE_0_1;
  } else {
    core_mask = RKNN_NPU_CORE_0;//小尺寸图像绑定至0号核心,避免单核,多核切换
  }
  ALGLogInfo<<"Bind mode_id, core_mask "<<mod_id<<" "<<core_mask;
  ret = rknn_set_core_mask(*rknn_ctx, core_mask);
  if (ret < 0) {
    return -1;
  }
  return 0;

}

/**
* 添加神经网络模型
* @param  ctx_id			神经网络上下文ID
* @param  group_id			分组ID
* @param  mod_id			模型ID
* @param  mod_data			模型数据
* @param  mod_size			模型大小
* @param  labels_data		标签数据
* @param  labels_size		标签大小
* @param  multi_label_flag	多标签标记
* @return 0 success other fail
 */

int NNet_AddModel(NNetCtxID_t ctx_id, NNetGroup_e group_id, NNetModID_e mod_id,
                  uint8_t *mod_data, uint32_t mod_size,
                  uint8_t *labels_data, uint32_t labels_size,
                  const ResizeType &resize_type, const float &nms_thr,
                  const float &conf_thr, const NNetTypeID_e& net_type_id,
                  const std::vector<float>& float_param_v) {
    int ret = 0;
    NNetCtx_t *ctx = (NNetCtx_t *) ctx_id;
    if (ctx == NULL || mod_data == NULL || mod_size == 0 || labels_size == 0) {
        return -1;
    }

    if (NULL != NNet_FindModel(ctx, group_id, mod_id)) {
        return 0;
    }

    std::vector<std::string> labels;
    NNet_MakeLabelsList(labels, labels_data, labels_size);

    rknn_context rknn_ctx;
    ret = rknn_init(&rknn_ctx, mod_data, mod_size, 0, NULL);
    if (ret < 0) {
        printf("rknn_init error ret=%d  mod_id = %d\r\n", ret, mod_id);
        return -1;
    }
    // 多核调用
/*    rknn_core_mask core_mask = RKNN_NPU_CORE_0_1_2;
    ret = rknn_set_core_mask(rknn_ctx, core_mask);
    if (ret < 0) {
      printf("rknn_init error ret=%d\r\n", ret);
      return -1;
    }*/
    // Get weight and internal mem size
    rknn_mem_size mem_size;
    ret = rknn_query(rknn_ctx, RKNN_QUERY_MEM_SIZE, &mem_size,
                     sizeof(mem_size));
    if (ret != RKNN_SUCC) {
      printf("rknn_query fail! ret=%d\n", ret);
      return -1;
    }

    printf("mod_id: %d, total weight size: %d, total internal size: %d\n",
           mod_id, mem_size.total_weight_size, mem_size.total_internal_size);


    rknn_input_output_num io_num;
    ret = rknn_query(rknn_ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0) {
        printf("rknn_query ionum error ret=%d\r\n", ret);
        return -1;
    }

    rknn_tensor_attr input_attrs;
    input_attrs.index = 0;
    ret = rknn_query(rknn_ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs), sizeof(rknn_tensor_attr));
    if (ret < 0) {
        printf("rknn_query attr error ret=%d\r\n", ret);
        return -1;
    }

    int channel = 0;
    int width = 0;
    int height = 0;
    if (input_attrs.fmt == RKNN_TENSOR_NCHW) {
        channel = input_attrs.dims[1];
        height = input_attrs.dims[2];
        width = input_attrs.dims[3];
    } else {
        height = input_attrs.dims[1];
        width = input_attrs.dims[2];
        channel = input_attrs.dims[3];
    }

    std::vector<rknn_tensor_attr> output_attrs;
    for (int i = 0; i < io_num.n_output; i++) {
        rknn_tensor_attr attr;
        attr.index = i;
        ret = rknn_query(rknn_ctx, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(rknn_tensor_attr));
        output_attrs.push_back(attr);
    }

    NNetModel_t *mod_ins = (NNetModel_t *) new NNetModel_t;
    if (mod_ins == NULL) {
        return -1;
    }
    NNET_GROUP_ID(mod_ins) = group_id;
    NNET_MOD_ID(mod_ins) = mod_id;
    NNET_MOD_RKNN_CTX(mod_ins) = rknn_ctx;
    NNET_MOD_WIDTH(mod_ins) = width;
    NNET_MOD_HEIGHT(mod_ins) = height;
    NNET_MOD_CHANNEL(mod_ins) = channel;
    NNET_MOD_NET_NUM_IN(mod_ins) = io_num.n_input;
    NNET_MOD_NET_NUM_OUT(mod_ins) = io_num.n_output;
    NNET_MOD_LABELS(mod_ins) = labels;
    NNET_MOD_RKNN_ATTRS(mod_ins) = output_attrs;
    NNET_MOD_RESIZE_TYPE(mod_ins) = resize_type;
    NNET_MOD_NMS_THR(mod_ins) = nms_thr;
    NNET_MOD_CONF_THR(mod_ins) = conf_thr;
    NNET_MOD_NET_TYPE_ID(mod_ins) = net_type_id;
    NNET_MOD_FLOAT_PARAMS_V(mod_ins) = float_param_v;
    NNET_CTX_MOD_LIST(ctx).push_back(*mod_ins);



    return 0;
}
template<class T>
void PrintParams(const std::vector<T>& params){
  std::ostringstream os;
/*  decltype(params)::value_type;*/
  for(const auto& iter:params){
    os<<iter<<" ";
  }
  ALGLogInfo<<os.str();

}

#define ANCHORS_NUMS  18
int ParseAnchors(const std::vector<float>& anchors, float* config_anchors0,
                 float* config_anchors1,float* config_anchors2){
  if(anchors.size()!=ANCHORS_NUMS){
    ALGLogError<<"Anchor size must be "<<ANCHORS_NUMS<<" but "<<anchors.size()<<" was given";
    return -1;
  }
  std::copy(anchors.begin(),anchors.begin()+6, config_anchors0);
  std::copy(anchors.begin()+6,anchors.begin()+12, config_anchors1);
  std::copy(anchors.begin()+12,anchors.end(), config_anchors2);
  return 0;
}


int ParseModelPrams(float model_type_nums,
                    float nms_nums,
                    float conf_nums,
                    float anchor_nums,
                    float label_nums,
                    float reserved_float_param_nums,
                    float reserved_string_param_nums,
                    const std::vector<float>& model_type_v,
                    const std::vector<float>& nms_v,
                    const std::vector<float>& conf_v,
                    const std::vector<float>& anchors,
                    const std::vector<std::string>& labels,
                    const std::vector<float>& reserved_float_params,
                    const std::vector<std::string>& reserved_string_params,
                    NNetModel_t *mod_ins
                    ){
  try{
    ALGLogInfo<<"Required "<<model_type_nums<<" "<<nms_nums<<" "<<conf_nums<<" "<<anchor_nums<<" "<<label_nums
                << " "<<reserved_float_param_nums<<" "<<reserved_string_param_nums;
    ALGLogInfo<<"Given "<<model_type_v.size()<<" "<<nms_v.size()<<" "<<conf_v.size()<<" "<<anchors.size()<<" "<<labels.size()
                << " "<<reserved_float_params.size()<<" "<<reserved_string_params.size();
    if(model_type_nums!=model_type_v.size()||
        nms_nums!=nms_v.size()||
        conf_nums!=conf_v.size()||
        anchor_nums!=anchors.size()||
        reserved_float_param_nums!=reserved_float_params.size()||
        reserved_string_param_nums!=reserved_string_params.size()){
      ALGLogError<<"Xml param nums are not in consistent with required";
      return -1;
    }
    // 如果类别数量不是自定义类,且类别数量与模型输出类别数不等,则报错
    if(label_nums!=LABEL_NUMS_CUSTOM&&label_nums!= labels.size()){
      ALGLogError<<"Xml label param nums are not in consistent with required";
      return -2;
    }

    if(label_nums==LABEL_NUMS_CUSTOM){
      ALGLogWarning<<"Model category nums is determined by the xml doc. Make sure the xml is consistency with model. Or the result might be wrong";
    }

    ALGLogInfo<<"model type ";
    PrintParams( model_type_v);

    ALGLogInfo<<"nms";
    PrintParams( nms_v);

    ALGLogInfo<<"conf";
    PrintParams( conf_v);

    ALGLogInfo<<"anchor";
    PrintParams( anchors);

    ALGLogInfo<<"label ";
    PrintParams( labels);

    ALGLogInfo<<"reserved_float_params ";
    PrintParams( reserved_float_params);

    ALGLogInfo<<"reserved_string_params ";
    PrintParams( reserved_string_params);
    float& nms_thr = NNET_MOD_NMS_THR(mod_ins);
    float& conf_thr = NNET_MOD_CONF_THR(mod_ins);
    NNetTypeID_e model_type = (NNetTypeID_e) model_type_v[0];
    NNET_MOD_NET_TYPE_ID(mod_ins) = model_type;
    switch (model_type) {
      case NNET_TYPE_YOLO_RECT: {
        nms_thr = nms_v[0];
        conf_thr = conf_v[0];
        int ret = ParseAnchors(anchors, mod_ins->anchor0,
                               mod_ins->anchor1,mod_ins->anchor2);
        if(ret){
          return -2;
        }
        break;
      }
      case NNET_TYPE_PP_ROTATED_POLY:{
        nms_thr = nms_v[0];
        conf_thr = conf_v[0];
        break;
      }
      case NNET_TYPE_SEG_ALL:{
        conf_thr = conf_v[0];
        break;
      }
      case NNET_TYPE_CLS_ALL:{
        conf_thr = 0;
        break;
      }
      case NNET_TYPE_YOLO_RECT_UNQUAN:{
        nms_thr = nms_v[0];
        conf_thr = conf_v[0];
        int ret = ParseAnchors(anchors, mod_ins->anchor0,
                               mod_ins->anchor1,mod_ins->anchor2);
        if(ret){
          return -2;
        }
        break;
      }
      default:{
        ALGLogError<<"Unknown model type "<<model_type;
        return -2;
      }
    }
  }catch (std::exception& e){
    ALGLogError<<"Error happened in model init";
    return -3;
  }
  return 0;

}



//int NNet_AddModel(NNetCtxID_t ctx_id, NNetGroup_e group_id, NNetModID_e mod_id,
//                  uint8_t *mod_data, uint32_t mod_size,const ResizeType &resize_type,
//                  float model_type_nums,
//                  float nms_nums,
//                  float conf_nums,
//                  float anchor_nums,
//                  float label_nums,
//                  float reserved_float_param_nums,
//                  float reserved_string_param_nums,
//                  const std::vector<float>& model_type_v,
//                  const std::vector<float>& nms_v,
//                  const std::vector<float>& conf_v,
//                  const std::vector<float>& anchors,
//                  const std::vector<std::string>& labels,
//                  const std::vector<float>& reserved_float_params,
//                  const std::vector<std::string>& reserved_string_params
//                  ){
//  int ret = 0;
//  NNetCtx_t *ctx = (NNetCtx_t *) ctx_id;
//  if (ctx == NULL || mod_data == NULL || mod_size == 0) {
//    ALGLogError<<"Empty model mod_data, mode size: "<<mod_data<< "\n"<<mod_size;
//    return -1;
//  }
//  if (NULL != NNet_FindModel(ctx, group_id, mod_id)) {
//    return 0;
//  }
//
//  rknn_context rknn_ctx;
//  ret = rknn_init(&rknn_ctx, mod_data, mod_size, 0, NULL);
//  if (ret < 0) {
//    printf("rknn_init error ret=%d\r\n", ret);
//    return -1;
//  }
//  // 多核调用
//  /*    rknn_core_mask core_mask = RKNN_NPU_CORE_0_1_2;
//      ret = rknn_set_core_mask(rknn_ctx, core_mask);
//      if (ret < 0) {
//        printf("rknn_init error ret=%d\r\n", ret);
//        return -1;
//      }*/
//  // Get weight and internal mem size
//  rknn_mem_size mem_size;
//  ret = rknn_query(rknn_ctx, RKNN_QUERY_MEM_SIZE, &mem_size,
//                   sizeof(mem_size));
//  if (ret != RKNN_SUCC) {
//    printf("rknn_query fail! ret=%d\n", ret);
//    return -1;
//  }
//
//  printf("mod_id: %d, total weight size: %d, total internal size: %d\n",
//         mod_id, mem_size.total_weight_size, mem_size.total_internal_size);
//
//
//  rknn_input_output_num io_num;
//  ret = rknn_query(rknn_ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
//  if (ret < 0) {
//    printf("rknn_query ionum error ret=%d\r\n", ret);
//    return -1;
//  }
//
//  rknn_tensor_attr input_attrs;
//  input_attrs.index = 0;
//  ret = rknn_query(rknn_ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs), sizeof(rknn_tensor_attr));
//  if (ret < 0) {
//    printf("rknn_query attr error ret=%d\r\n", ret);
//    return -1;
//  }
//
//  int channel = 0;
//  int width = 0;
//  int height = 0;
//  if (input_attrs.fmt == RKNN_TENSOR_NCHW) {
//    channel = input_attrs.dims[1];
//    height = input_attrs.dims[2];
//    width = input_attrs.dims[3];
//  } else {
//    height = input_attrs.dims[1];
//    width = input_attrs.dims[2];
//    channel = input_attrs.dims[3];
//  }
//
//  std::vector<rknn_tensor_attr> output_attrs;
//  for (int i = 0; i < io_num.n_output; i++) {
//    rknn_tensor_attr attr;
//    attr.index = i;
//    ret = rknn_query(rknn_ctx, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(rknn_tensor_attr));
//    output_attrs.push_back(attr);
//  }
//
//  NNetModel_t *mod_ins = (NNetModel_t *) new NNetModel_t;
//  if (mod_ins == NULL) {
//    return -1;
//  }
//  ALGLogDebug<<"model_type_nums:"<<model_type_nums;
//  ALGLogDebug<<"num_nums:"<<nms_nums;
//
//
//  //配置文件中的属性
//  ret = ParseModelPrams(model_type_nums,
//                        nms_nums,
//                        conf_nums,
//                        anchor_nums,
//                        label_nums,
//                        reserved_float_param_nums,
//                        reserved_string_param_nums,
//                        model_type_v,
//                        nms_v,
//                        conf_v,
//                        anchors,
//                        labels,
//                        reserved_float_params,
//                        reserved_string_params,
//                        mod_ins
//                        );
//  if(ret){
//    return -2;
//  }
//
//
//  ret = NNet_BindNpuCore(&rknn_ctx, mod_id,
//                         width, height);
//  if(ret){
//    ALGLogError<<"Failed to set multi npu core";
//    return -2;
//  }
int NNet_AddModel(NNetCtxID_t ctx_id, NNetGroup_e group_id, NNetModID_e mod_id,
                  uint8_t *mod_data, uint32_t mod_size,const ResizeType &resize_type,
                  float model_type_nums,
                  float nms_nums,
                  float conf_nums,
                  float anchor_nums,
                  float label_nums,
                  float reserved_float_param_nums,
                  float reserved_string_param_nums,
                  const std::vector<float>& model_type_v,
                  const std::vector<float>& nms_v,
                  const std::vector<float>& conf_v,
                  const std::vector<float>& anchors,
                  const std::vector<std::string>& labels,
                  const std::vector<float>& reserved_float_params,
                  const std::vector<std::string>& reserved_string_params
){
    int ret = 0;
    printf("[Debug] Entering NNet_AddModel. ctx_id=%p, mod_data=%p, mod_size=%u\n",
           (void*)ctx_id, mod_data, mod_size); // 打印输入参数
    NNetCtx_t *ctx = (NNetCtx_t *) ctx_id;
    if (ctx == NULL || mod_data == NULL || mod_size == 0) {
        ALGLogError<<"Empty model mod_data, mode size: "<<mod_data<< "\n"<<mod_size;
        return -1;
    }
    printf("[Debug] Context valid. Checking existing models...\n");

    if (NULL != NNet_FindModel(ctx, group_id, mod_id)) {
        printf("[Debug] Model group_id=%d, mod_id=%d already exists\n", group_id, mod_id);
        return 0;
    }

    rknn_context rknn_ctx;
    printf("[Debug] Initializing rknn with mod_data=%p, mod_size=%u\n", mod_data, mod_size);
    ret = rknn_init(&rknn_ctx, mod_data, mod_size, 0, NULL);
    if (ret < 0) {
        printf("rknn_init error ret=%d mod_id= %d\r\n", ret, mod_id);
        return -1;
    }
    printf("[Debug] rknn_init success. Querying memory size...\n");

    rknn_mem_size mem_size;
    ret = rknn_query(rknn_ctx, RKNN_QUERY_MEM_SIZE, &mem_size, sizeof(mem_size));
    if (ret != RKNN_SUCC) {
        printf("rknn_query fail! ret=%d mod_id = %d\n", ret, mod_id);
        return -1;
    }
    std::cout << "增加模型的 group id: " << group_id << " 模型的 id: " << mod_id << "mod_size " << mod_size << "\n";
    printf("mod_id: %d, total weight size: %d, total internal size: %d\n",
           mod_id, mem_size.total_weight_size, mem_size.total_internal_size);

    rknn_input_output_num io_num;
    ret = rknn_query(rknn_ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0) {
        printf("rknn_query ionum error ret=%d\r\n", ret);
        return -1;
    }
    printf("[Debug] Model has %d inputs, %d outputs\n", io_num.n_input, io_num.n_output);

    rknn_tensor_attr input_attrs;
    input_attrs.index = 0;
    ret = rknn_query(rknn_ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs), sizeof(rknn_tensor_attr));
    if (ret < 0) {
        printf("rknn_query attr error ret=%d\r\n", ret);
        return -1;
    }

    // 打印输入张量维度信息
    printf("[Debug] Input tensor format: %d, n_dims: %d\n", input_attrs.fmt, input_attrs.n_dims);
    for(int i=0; i<input_attrs.n_dims; ++i){
        printf("input_attrs.dims[%d] = %d\n", i, input_attrs.dims[i]);
    }

    int channel = 0;
    int width = 0;
    int height = 0;
    if (input_attrs.fmt == RKNN_TENSOR_NCHW) {
        channel = input_attrs.dims[1];
        height = input_attrs.dims[2];
        width = input_attrs.dims[3];
    } else {
        height = input_attrs.dims[1];
        width = input_attrs.dims[2];
        channel = input_attrs.dims[3];
    }
    printf("[Debug] Parsed input dimensions: width=%d, height=%d, channel=%d\n", width, height, channel);

    std::vector<rknn_tensor_attr> output_attrs;
    for (int i = 0; i < io_num.n_output; i++) {
        rknn_tensor_attr attr;
        attr.index = i;
        ret = rknn_query(rknn_ctx, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(rknn_tensor_attr));
        if (ret < 0) {
            printf("[ERROR] Failed to query output %d attributes, ret=%d\n", i, ret);
            return -1;
        }
        output_attrs.push_back(attr);
        printf("[Debug] Output %d: n_dims=%d, fmt=%d\n", i, attr.n_dims, attr.fmt);
    }

    NNetModel_t *mod_ins = new NNetModel_t;
    printf("[Debug] Created mod_ins at %p\n", mod_ins);
    if (mod_ins == NULL) {
        printf("[ERROR] Failed to allocate NNetModel_t\n");
        return -1;
    }

    printf("[Debug] Parsing model parameters...\n");
    ret = ParseModelPrams(model_type_nums,
                          nms_nums,
                          conf_nums,
                          anchor_nums,
                          label_nums,
                          reserved_float_param_nums,
                          reserved_string_param_nums,
                          model_type_v,
                          nms_v,
                          conf_v,
                          anchors,
                          labels,
                          reserved_float_params,
                          reserved_string_params,
                          mod_ins
    );
    if(ret){
        printf("[ERROR] ParseModelParams failed with ret=%d\n", ret);
        delete mod_ins; // 避免内存泄漏
        return -2;
    }

    printf("[Debug] Binding NPU core...\n");
    ret = NNet_BindNpuCore(&rknn_ctx, mod_id, width, height);
    if(ret){
        ALGLogError<<"Failed to set multi npu core";
        delete mod_ins; // 清理内存
        return -2;
    }

    // 设置成员变量
    NNET_GROUP_ID(mod_ins) = group_id;
    NNET_MOD_ID(mod_ins) = mod_id;
    NNET_MOD_RKNN_CTX(mod_ins) = rknn_ctx;
    NNET_MOD_WIDTH(mod_ins) = width;
    NNET_MOD_HEIGHT(mod_ins) = height;
    NNET_MOD_CHANNEL(mod_ins) = channel;
    NNET_MOD_NET_NUM_IN(mod_ins) = io_num.n_input;
    NNET_MOD_NET_NUM_OUT(mod_ins) = io_num.n_output;
    NNET_MOD_LABELS(mod_ins) = labels;
    NNET_MOD_RKNN_ATTRS(mod_ins) = output_attrs;
    NNET_MOD_RESIZE_TYPE(mod_ins) = resize_type;
    NNET_MOD_RESERVED_FLOAT_PARAMS(mod_ins) = reserved_float_params;
    NNET_MOD_RESERVED_STRING_PARAMS(mod_ins) = reserved_string_params;

    printf("[Debug] Adding model to context. Current model count: %zu\n", NNET_CTX_MOD_LIST(ctx).size());
    NNET_CTX_MOD_LIST(ctx).push_back(*mod_ins);
    printf("[Debug] Model added successfully. New count: %zu\n", NNET_CTX_MOD_LIST(ctx).size());

//        delete mod_ins; // 注意：这里可能存在错误，因为push_back(*mod_ins)拷贝了对象，原指针需要是否应该保留？
//        // 此处可能需要修改为不立即delete，或者确保结构体的拷贝正确

    return 0;
}


//  NNET_GROUP_ID(mod_ins) = group_id;
//  NNET_MOD_ID(mod_ins) = mod_id;
//  NNET_MOD_RKNN_CTX(mod_ins) = rknn_ctx;
//  NNET_MOD_WIDTH(mod_ins) = width;
//  NNET_MOD_HEIGHT(mod_ins) = height;
//  NNET_MOD_CHANNEL(mod_ins) = channel;
//  NNET_MOD_NET_NUM_IN(mod_ins) = io_num.n_input;
//  NNET_MOD_NET_NUM_OUT(mod_ins) = io_num.n_output;
//  NNET_MOD_LABELS(mod_ins) = labels;
//  NNET_MOD_RKNN_ATTRS(mod_ins) = output_attrs;
//  NNET_MOD_RESIZE_TYPE(mod_ins) = resize_type;
//  NNET_MOD_RESERVED_FLOAT_PARAMS(mod_ins) = reserved_float_params;
//  NNET_MOD_RESERVED_STRING_PARAMS(mod_ins) = reserved_string_params;
//
//  NNET_CTX_MOD_LIST(ctx).push_back(*mod_ins);
//
//  return 0;
//}


//将数组中的点组织成成对的点
//void ConstructPoint(const std::vector<float>& xy_v, std::vector<cv::Point>& point){
//  if(xy_v.size()%2!=0) return;
//  int point_num = xy_v.size()/2;
//  for(int i =0; i<point_num; ++i){
//    point.emplace_back(xy_v[i*2], xy_v[i*2+1]);
//  }
//}

/*!
 * 查找指定模型的reserved_params,不同模型内的reserved_params含义不同,由模型部署方指定
 * @param ctx_id
 * @param group_id
 * @param mod_id
 * @param category_conf
 * @return
 */
int NNet_GetReservedFloatPrams(NNetCtxID_t ctx_id, NNetGroup_e group_id, NNetModID_e mod_id, std::vector<float>& reserved_float_params){
  NNetCtx_t *ctx = (NNetCtx_t *) ctx_id;
  NNetModel_t * mod_ins = NNet_FindModel(ctx, group_id, mod_id);
  if(mod_ins== nullptr){
    ALGLogError<<"Failed to find mod";
    return -1;
  }
  reserved_float_params = mod_ins->reserved_float_params;
  return 0;
}


/**
* 神经网络推理
* @param  ctx_id	神经网络上下文ID
* @param  mod_id	模型ID
* @param  img		神经网络输入图像
* @param  result	神经网络输出结果
* @return 0 success other fail
 */
int NNet_Inference(NNetCtxID_t ctx_id, NNetGroup_e group_id, NNetModID_e mod_id, NNetImg_t *img,
                   std::list<NNetResult_t> &result) {
//  usleep(700000);
//  std::cout<<"Do not run model"<<std::endl;
//  return 0;
#if(NET_USE_TIMECNT)
    std::string net_pre_name = "pre_" + std::to_string(mod_id);
    std::string net_inf_name = "inf_" + std::to_string(mod_id);
    std::string net_post_name ="post_" + std::to_string(mod_id);
    TimeCnt_Init(net_pre_name.c_str(), 1);
    TimeCnt_Init(net_inf_name.c_str(), 1);
    TimeCnt_Init(net_post_name.c_str(), 1);
    TimeCnt_Start(net_pre_name.c_str());
#endif
    result.clear();
    int ret;
    NNetCtx_t *ctx = (NNetCtx_t *) ctx_id;
    std::cout << "当前推理使用的模型 group id: " << group_id << " 模型的 id: "<< mod_id<< "\n";

    NNetModel_t *mod_ins = NNet_FindModel(ctx, group_id, mod_id);
    if (mod_ins == nullptr) {
        ALGLogError<<"Failed to find model, mod_ins nullptr 组id :"<<group_id<<" 模型id: "<<mod_id;
        return -1;
    }
    ALGLogInfo << __FUNCTION__<<"  "<<__LINE__<<"  当前推理使用的模型 group id: " << group_id << " 模型的 id: "<< mod_id <<"\n";
    std::cout << __FUNCTION__ << "  " << __LINE__ << "  当前推理使用的模型 group id: " << group_id << " 模型的 id: " << mod_id << "\n";

    rknn_context rknn_ctx = NNET_MOD_RKNN_CTX(mod_ins);
    int channel = NNET_MOD_CHANNEL(mod_ins);
    int netNumOutput = NNET_MOD_NET_NUM_OUT(mod_ins);
    int netNumInput = NNET_MOD_NET_NUM_IN(mod_ins);
    int modelWidth = NNET_MOD_WIDTH(mod_ins);
    int modelHeight = NNET_MOD_HEIGHT(mod_ins);
    std::vector <std::string> labels = NNET_MOD_LABELS(mod_ins);
    std::vector <rknn_tensor_attr> output_attrs = NNET_MOD_RKNN_ATTRS(mod_ins);//仅量化后模型需要该值,以进行反量化

    ALGLogInfo<< __FUNCTION__<<"  "<<__LINE__<<"  获取反量化的节点信息完成"<<"\n";

    ResizeType resize_type = NNET_MOD_RESIZE_TYPE(mod_ins);
    float nms_thr = NNET_MOD_NMS_THR(mod_ins);
    float conf_thr = NNET_MOD_CONF_THR(mod_ins);
    NNetTypeID net_type_id = NNET_MOD_NET_TYPE_ID(mod_ins);
    const std::vector<float>& float_parma_v = NNET_MOD_FLOAT_PARAMS_V(mod_ins);
    cv::Mat ipt;

    rknn_input inputs[1] = {0};
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = modelWidth * modelHeight * channel;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;

    void *resize_buf = nullptr;

    ALGLogInfo << __FUNCTION__ << "  " << __LINE__ << "模型的输入大小：" << " 宽： " << modelWidth << " 高：" << modelHeight << " 通道： "
               << channel <<" 模型的 id: " << mod_id << "\n";
    // std::cout << __FUNCTION__ << "  " << __LINE__ << "图片大小 ：" << " 宽： " << img->cols << " 高：" << img->rows << " 通道： " << img->channels() << "\n";
    // std::cout << __FUNCTION__ << "  " << __LINE__ << "resize_type  ：" << resize_type  << "\n";

    std::cout << __FUNCTION__ << "  " << __LINE__ << "resize_type  ：" << resize_type << "\n";
    std::cout << __FUNCTION__ << "  " << __LINE__ << "图片大小 ：" << " 宽： " << img->cols << " 高：" << img->rows << " 通道： " << img->channels()
              << "\n";

    if(resize_type==RGA_NORMAL){
      ret = resize_rga_output_buf(&resize_buf, *img, cv::Size(modelWidth, modelHeight));
      if (ret != 0)
      {
        ALGLogError<<"Resize with rga error";
        return -2;
      }

      inputs[0].buf = resize_buf;
    } else{
      ResizeImg(*img, ipt, cv::Size(modelWidth, modelHeight), resize_type);
      inputs[0].buf = (void*)ipt.data;

    }

    std::cout << __FUNCTION__ << "  " << __LINE__ << "图片大小 ：" << " 宽： " << ipt.cols << " 高：" << ipt.rows << " 通道： " << ipt.channels()
              << "\n";

    int ipt_width = ipt.cols;
    int ipt_height = ipt.rows;
#if(NET_USE_TIMECNT)
    TimeCnt_End(net_pre_name.c_str());
#endif


#if(NET_USE_TIMECNT)
    TimeCnt_Start(net_inf_name.c_str());
#endif
    ret = rknn_inputs_set(rknn_ctx, netNumInput, inputs);//耗时较多
    if(ret!=0){
      ALGLogError<<"Failed to set input";
      return -2;
    }
    ALGLogInfo << __FUNCTION__ << "  " << __LINE__ << " 模型的 id: " << mod_id <<" 模型设置输出完成 \n";

    rknn_output *outputs = new rknn_output[netNumOutput];
    memset(outputs, 0, netNumOutput * sizeof(rknn_output));
    // 后续均使用int8量化
    for (int i = 0; i < netNumOutput; i++) {
        if(net_type_id==NNetTypeID::NNET_TYPE_YOLO_RECT){
          outputs[i].want_float = 0;
        } else{
          outputs[i].want_float = 1;
        }
        outputs[i].is_prealloc = false;
    }

    ret = rknn_run(rknn_ctx, NULL);
    if(ret!=0){
      ALGLogError<<"Failed to run rknn";
      return -3;
    }

    ret = rknn_outputs_get(rknn_ctx, netNumOutput, outputs, NULL);
    ALGLogInfo << __FUNCTION__ << "  " << __LINE__ << " 模型的 id: " << mod_id << " 模型获取输出完成 \n";

    if(ret!=0){
      ALGLogError<<"Failed to get rknn outputs";
      return -4;
    }
    // post process
    float scale_w = (float) modelWidth / ipt_width;
    float scale_h = (float) modelHeight / ipt_height;
    // 获取量化属性
    std::vector<float> out_scales;
    std::vector <int32_t> out_zps;
    for (int i = 0; i < netNumOutput; ++i) {
      out_scales.push_back(output_attrs.at(i).scale);
      out_zps.push_back(output_attrs.at(i).zp);
    }
    // get anchor
    float* target_anchor0 = mod_ins->anchor0;
    float* target_anchor1 = mod_ins->anchor1;
    float* target_anchor2 = mod_ins->anchor2;
    ALGLogInfo << __FUNCTION__ << "  " << __LINE__  << " 模型的 id: " << mod_id << " 模型获取量化属性完成 \n";

#if(NET_USE_TIMECNT)
    TimeCnt_End(net_inf_name.c_str());
#endif

#if(NET_USE_TIMECNT)
    TimeCnt_Start(net_post_name.c_str());
#endif
    ALGLogInfo << __FUNCTION__ << "  " << __LINE__ << " 模型的 id: " << mod_id << " 模型开始后处理过程 \n";

    if(resize_type==RGA_NORMAL){
      free(resize_buf);
    }
    // 根据网络类型调用相应后处理算法
    switch (net_type_id) {
      case NNetTypeID::NNET_TYPE_YOLO_RECT:{

          ALGLogInfo << __FUNCTION__ << "  " << __LINE__ << " 模型的 id: " << mod_id << " NNetTypeID::NNET_TYPE_YOLO_RECT,量化后的检测模型 ";
          try {
              yolo_post_process((int8_t*)outputs[0].buf,
                                (int8_t*)outputs[1].buf,
                                (int8_t*)outputs[2].buf,
                                modelHeight,
                                modelWidth,
                                conf_thr,
                                nms_thr,
                                scale_w,
                                scale_h,
                                out_zps,
                                out_scales,
                                labels,
                                result,
                                img->rows,
                                img->cols,
                                resize_type,
                                target_anchor0,
                                target_anchor1,
                                target_anchor2);
        } catch (const cv::Exception &e) {
          ALGLogError<<"Error happened in yolo post process";
          return -3;
        } catch (const std::exception &e) {
          ALGLogError << "Error happened in yolo post process";
          return -3;
        }

        break;
      }
      case NNetTypeID::NNET_TYPE_YOLO_RECT_UNQUAN:{
        try{
            ALGLogInfo << __FUNCTION__ << "  " << __LINE__ << " 模型的 id: " << mod_id
                                                                                  << "NNetTypeID::NNET_TYPE_YOLO_RECT_UNQUAN,未量化的检测模型 ";
            yolo_post_process_unquantization((float*)outputs[0].buf,
                                             (float*)outputs[1].buf,
                                             (float*)outputs[2].buf,
                                             modelHeight,
                                             modelWidth,
                                             conf_thr,
                                             nms_thr,
                                             scale_w,
                                             scale_h,
                                             out_zps,
                                             out_scales,
                                             labels,
                                             result,
                                             img->rows,
                                             img->cols,
                                             resize_type,
                                             target_anchor0,
                                             target_anchor1,
                                             target_anchor2);
        } catch (const cv::Exception &e) {
          ALGLogError<<"Error happened in yolo post process";
          return -3;
        } catch (const std::exception &e) {
          ALGLogError << "Error happened in yolo post process";
          return -3;
        }

        break;
      }
      case NNetTypeID::NNET_TYPE_PP_ROTATED_POLY:{
        try{
          ALGLogInfo << __FUNCTION__ << "  " << __LINE__<< " 模型的 id: " << mod_id <<"NNetTypeID::NNET_TYPE_PP_ROTATED_POLY,旋转检测框模型";
          PostProcessInclineRbc((float *)outputs[0].buf,
                                (float *)outputs[1].buf, nms_thr, conf_thr,
                                result);
        } catch (const cv::Exception &e) {
          ALGLogError<<"Error happened in yolo post process";
          return -3;
        } catch (const std::exception &e) {
          ALGLogError << "Error happened in yolo post process";
          return -3;
        }

        break;

      }

      case NNetTypeID::NNET_TYPE_SEG_ALL:{
        try{
          ALGLogInfo << __FUNCTION__ << "  " << __LINE__
                     << " 模型的 id: " << mod_id << "NNetTypeID::NNET_TYPE_SEG_ALL,分割模型";
          PostProcessSegRbc((float *)outputs[0].buf, labels.size(), modelHeight,
                            modelWidth, img->rows, img->cols, result);
        } catch (const cv::Exception &e) {
          ALGLogError<<"Error happened in yolo post process";
          return -3;
        } catch (const std::exception &e) {
          ALGLogError << "Error happened in yolo post process";
          return -3;
        }

        break;
      }

      case NNetTypeID::NNET_TYPE_CLS_ALL:{
        try{
            ALGLogInfo << __FUNCTION__ << "  " << __LINE__ << " 模型的 id: " << mod_id << "NNetTypeID::NNET_TYPE_CLS_ALL,分类模型";
            PostProcessCls((float*)outputs[0].buf, out_zps, out_scales, labels, result);
        } catch (const cv::Exception &e) {
          ALGLogError<<"Error happened in yolo post process";
          return -3;
        } catch (const std::exception &e) {
          ALGLogError << "Error happened in yolo post process";
          return -3;
        }

        break;
      }

      default:
        ALGLogError<<"Error net type "<<net_type_id;
        return -5;
    }
    ALGLogInfo << __FUNCTION__ << "  " << __LINE__ << "rknn_outputs_release,释放模型的输出内存之前";
    rknn_outputs_release(rknn_ctx, netNumOutput, outputs);
    ALGLogInfo << __FUNCTION__ << "  " << __LINE__ << "rknn_outputs_release,释放模型的输出内存之后";


    // std::cout << " NNet_Inference 1879 group_id: " << group_id << " mod_id: " << mod_id << std::endl;
    // for (int i = 0; i < result.begin()->category_v.size(); i++) {
    //     std::cout << " index: " << i << " ret: " << result.begin()->category_v[i];
    // }

    ALGLogInfo << __FUNCTION__ << "  " << __LINE__ << "模型推理完成\n";

    std::cout << __FUNCTION__ << "  " << __LINE__ << "  当前推理使用的模型 group id: " << group_id << " 模型的 id: " << mod_id << "模型推理完成\n";
#if(NET_USE_TIMECNT)
    TimeCnt_End(net_post_name.c_str());
#endif
    return 0;
}

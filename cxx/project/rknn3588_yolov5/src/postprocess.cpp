// Copyright (c) 2021 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "postprocess.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <omp.h>
#include <set>
#include <vector>
#include "utils.h"
#define LABEL_NALE_TXT_PATH "./labels_list.txt"

static char* labels[OBJ_CLASS_NUM];

#define USE_OPENCL 1
#if USE_OPENCL
#include "NmsCl.h"
#endif
#if USE_OPENCL
ALG_CL::NmsCl nms_cl;
#endif
const int anchor0[6] = { 10, 13, 16, 30, 33, 23 };
const int anchor1[6] = { 30, 61, 62, 45, 59, 119 };
const int anchor2[6] = { 116, 90, 156, 198, 373, 326 };

inline static int clamp(float val, int min, int max) { return val > min ? (val < max ? val : max) : min; }

char* readLine(FILE* fp, char* buffer, int* len)
{
    int ch;
    int i = 0;
    size_t buff_len = 0;

    buffer = (char*)malloc(buff_len + 1);
    if (!buffer)
        return NULL; // Out of memory

    while ((ch = fgetc(fp)) != '\n' && ch != EOF) {
        buff_len++;
        void* tmp = realloc(buffer, buff_len + 1);
        if (tmp == NULL) {
            free(buffer);
            return NULL; // Out of memory
        }
        buffer = (char*)tmp;

        buffer[i] = (char)ch;
        i++;
    }
    buffer[i] = '\0';

    *len = buff_len;

    // Detect end
    if (ch == EOF && (i == 0 || ferror(fp))) {
        free(buffer);
        return NULL;
    }
    return buffer;
}

int readLines(const char* fileName, char* lines[], int max_line)
{
    FILE* file = fopen(fileName, "r");
    char* s;
    int i = 0;
    int n = 0;

    if (file == NULL) {
        printf("Open %s fail!\n", fileName);
        return -1;
    }

    while ((s = readLine(file, s, &n)) != NULL) {
        lines[i++] = s;
        if (i >= max_line)
            break;
    }
    fclose(file);
    return i;
}

int loadLabelName(const char* locationFilename, char* label[])
{
    printf("loadLabelName %s\n", locationFilename);
    readLines(locationFilename, label, OBJ_CLASS_NUM);

#if USE_OPENCL
    if (nms_cl.Init(R"(./nms.cl)")) {
        return -1;
    }
#endif
    return 0;
}

static float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1,
    float ymax1)
{
    float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
    float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
    float i = w * h;
    float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) + (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;
    return u <= 0.f ? 0.f : (i / u);
}

static int nms(int validCount, std::vector<float>& outputLocations, std::vector<int> classIds, std::vector<int>& order,
    int filterId, float threshold)
{
    // for (int i = 0; i < validCount; ++i) {
    //     int n = order[i];
    //     if (n == -1 || classIds[n] != filterId) {
    //         continue;
    //     }
    //     for (int j = i + 1; j < validCount; ++j) {
    //         int m = order[j];
    //         if (m == -1 || classIds[m] != filterId) {
    //             continue;
    //         }
    //         float xmin0 = outputLocations[n * 4 + 0];
    //         float ymin0 = outputLocations[n * 4 + 1];
    //         float xmax0 = outputLocations[n * 4 + 0] + outputLocations[n * 4 + 2];
    //         float ymax0 = outputLocations[n * 4 + 1] + outputLocations[n * 4 + 3];

    //         float xmin1 = outputLocations[m * 4 + 0];
    //         float ymin1 = outputLocations[m * 4 + 1];
    //         float xmax1 = outputLocations[m * 4 + 0] + outputLocations[m * 4 + 2];
    //         float ymax1 = outputLocations[m * 4 + 1] + outputLocations[m * 4 + 3];

    //         float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

    //         if (iou > threshold) {
    //             order[j] = -1;
    //         }
    //     }
    // }
    // return 0;
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
            order[i] = -1; // 确保无效检测被正确标记
            continue;
        }
        int n = order[i];
        xmin[i] = outputLocations[n * 4 + 0];
        ymin[i] = outputLocations[n * 4 + 1];
        xmax[i] = xmin[i] + outputLocations[n * 4 + 2];
        ymax[i] = ymin[i] + outputLocations[n * 4 + 3];
    }
    // 优化的NMS主循环
    int i =0;
//#pragma omp parallel for collapse(2)
#pragma omp parallel for
    for (i = 0; i < validCount; ++i) {
        if (order[i] == -1 || suppressed[i] || classIds[i] != filterId)
        {
            order[i] = -1;
            continue;
        }
        int n = order[i];
        // 并行处理后续边界框
//#pragma omp parallel for num_threads(8)
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
    i=0;
    for (i = 0; i < validCount; ++i) {
        if (suppressed[i]) {
            order[i] = -1;
        }
    }

    return 0;
}

static int quick_sort_indice_inverse(std::vector<float>& input, int left, int right, std::vector<int>& indices)
{
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

static float sigmoid(float x) { return 1.0 / (1.0 + expf(-x)); }

static float unsigmoid(float y) { return -1.0 * logf((1.0 / y) - 1.0); }

inline static int32_t __clip(float val, float min, float max)
{
    float f = val <= min ? min : (val >= max ? max : val);
    return f;
}

static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale)
{
    float dst_val = (f32 / scale) + zp;
    int8_t res = (int8_t)__clip(dst_val, -128, 127);
    return res;
}

static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) { return ((float)qnt - (float)zp) * scale; }


static int process(int8_t* input, int* anchor, int grid_h, int grid_w, int height, int width, int stride,
    std::vector<float>& boxes, std::vector<float>& objProbs, std::vector<int>& classId, float threshold,
    int32_t zp, float scale)
{
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    int8_t thres_i8 = qnt_f32_to_affine(threshold, zp, scale);
    for (int a = 0; a < 3; a++) {
        for (int i = 0; i < grid_h; i++) {
            for (int j = 0; j < grid_w; j++) {
                int8_t box_confidence = input[(PROP_BOX_SIZE * a + 4) * grid_len + i * grid_w + j];
                if (box_confidence >= thres_i8) {
                    int offset = (PROP_BOX_SIZE * a) * grid_len + i * grid_w + j;
                    int8_t* in_ptr = input + offset;
                    float box_x = (deqnt_affine_to_f32(*in_ptr, zp, scale)) * 2.0 - 0.5;
                    float box_y = (deqnt_affine_to_f32(in_ptr[grid_len], zp, scale)) * 2.0 - 0.5;
                    float box_w = (deqnt_affine_to_f32(in_ptr[2 * grid_len], zp, scale)) * 2.0;
                    float box_h = (deqnt_affine_to_f32(in_ptr[3 * grid_len], zp, scale)) * 2.0;
                    box_x = (box_x + j) * (float)stride;
                    box_y = (box_y + i) * (float)stride;
                    box_w = box_w * box_w * (float)anchor[a * 2];
                    box_h = box_h * box_h * (float)anchor[a * 2 + 1];
                    box_x -= (box_w / 2.0);
                    box_y -= (box_h / 2.0);

                    int8_t maxClassProbs = in_ptr[5 * grid_len];
                    int maxClassId = 0;
                    for (int k = 1; k < OBJ_CLASS_NUM; ++k) {
                        int8_t prob = in_ptr[(5 + k) * grid_len];
                        if (prob > maxClassProbs) {
                            maxClassId = k;
                            maxClassProbs = prob;
                        }
                    }
                    if (maxClassProbs > thres_i8) {
                        objProbs.push_back((deqnt_affine_to_f32(maxClassProbs, zp, scale)) * (deqnt_affine_to_f32(box_confidence, zp, scale)));
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
    return validCount;
}

/**
static int process(int8_t* input, int* anchor, int grid_h, int grid_w, int height, int width, int stride,
    std::vector<float>& boxes, std::vector<float>& objProbs, std::vector<int>& classId, float threshold,
    int32_t zp, float scale)
*/
static int process_unquantization(float* input, int* anchor, int grid_h, int grid_w, int height, int width, int stride,
    std::vector<float>& boxes, std::vector<float>& objProbs, std::vector<int>& classId, float threshold,
    int32_t zp, float scale)
{
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    float thres = unsigmoid(threshold);
    //    int8_t thres_i8   = qnt_f32_to_affine(threshold, zp, scale);
    for (int a = 0; a < 3; a++) {
        for (int i = 0; i < grid_h; i++) {
            for (int j = 0; j < grid_w; j++) {
                float box_confidence = input[(PROP_BOX_SIZE * a + 4) * grid_len + i * grid_w + j];
                if (box_confidence >= thres) {
                    int offset = (PROP_BOX_SIZE * a) * grid_len + i * grid_w + j;
                    float* in_ptr = input + offset;
                    float box_x = sigmoid(*in_ptr) * 2.0 - 0.5;
                    float box_y = sigmoid(in_ptr[grid_len]) * 2.0 - 0.5;
                    float box_w = sigmoid(in_ptr[2 * grid_len]) * 2.0;
                    float box_h = sigmoid(in_ptr[3 * grid_len]) * 2.0;
                    box_x = (box_x + j) * (float)stride;
                    box_y = (box_y + i) * (float)stride;
                    box_w = box_w * box_w * (float)anchor[a * 2];
                    box_h = box_h * box_h * (float)anchor[a * 2 + 1];
                    box_x -= (box_w / 2.0);
                    box_y -= (box_h / 2.0);

                    float maxClassProbs = in_ptr[5 * grid_len];
                    int maxClassId = 0;
                    for (int k = 1; k < OBJ_CLASS_NUM; ++k) {
                        float prob = in_ptr[(5 + k) * grid_len];
                        if (prob > maxClassProbs) {
                            maxClassId = k;
                            maxClassProbs = prob;
                        }
                    }
                    if (maxClassProbs > thres) {
                        objProbs.push_back(sigmoid(maxClassProbs) * sigmoid(box_confidence));
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
    return validCount;
}


int post_process(int8_t* input0, int8_t* input1, int8_t* input2, int model_in_h, int model_in_w, float conf_threshold,
    float nms_threshold, BOX_RECT pads, float scale_w, float scale_h, std::vector<int32_t>& qnt_zps,
    std::vector<float>& qnt_scales, detect_result_group_t* group)
{
    static int init = -1;
    if (init == -1) {
        int ret = 0;
        ret = loadLabelName(LABEL_NALE_TXT_PATH, labels);
        if (ret < 0) {
            return -1;
        }

        init = 0;
    }
    memset(group, 0, sizeof(detect_result_group_t));

    std::vector<float> filterBoxes;
    std::vector<float> objProbs;
    std::vector<int> classId;

    // stride 8
    int stride0 = 8;
    int grid_h0 = model_in_h / stride0;
    int grid_w0 = model_in_w / stride0;
    int validCount0 = 0;
    validCount0 = process(input0, (int*)anchor0, grid_h0, grid_w0, model_in_h, model_in_w, stride0, filterBoxes, objProbs,
        classId, conf_threshold, qnt_zps[0], qnt_scales[0]);

    // stride 16
    int stride1 = 16;
    int grid_h1 = model_in_h / stride1;
    int grid_w1 = model_in_w / stride1;
    int validCount1 = 0;
    validCount1 = process(input1, (int*)anchor1, grid_h1, grid_w1, model_in_h, model_in_w, stride1, filterBoxes, objProbs,
        classId, conf_threshold, qnt_zps[1], qnt_scales[1]);

    // stride 32
    int stride2 = 32;
    int grid_h2 = model_in_h / stride2;
    int grid_w2 = model_in_w / stride2;
    int validCount2 = 0;
    validCount2 = process(input2, (int*)anchor2, grid_h2, grid_w2, model_in_h, model_in_w, stride2, filterBoxes, objProbs,
        classId, conf_threshold, qnt_zps[2], qnt_scales[2]);

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

    std::set<int> class_set(std::begin(classId), std::end(classId));

    for (auto c : class_set) {
        nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold);
    }

    int last_count = 0;
    group->count = 0;
    /* box valid detect target */
    for (int i = 0; i < validCount; ++i) {
        if (indexArray[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE) {
            continue;
        }
        int n = indexArray[i];

        float x1 = filterBoxes[n * 4 + 0] - pads.left;
        float y1 = filterBoxes[n * 4 + 1] - pads.top;
        float x2 = x1 + filterBoxes[n * 4 + 2];
        float y2 = y1 + filterBoxes[n * 4 + 3];
        int id = classId[n];
        float obj_conf = objProbs[i];

        group->results[last_count].box.left = (int)(clamp(x1, 0, model_in_w) / scale_w);
        group->results[last_count].box.top = (int)(clamp(y1, 0, model_in_h) / scale_h);
        group->results[last_count].box.right = (int)(clamp(x2, 0, model_in_w) / scale_w);
        group->results[last_count].box.bottom = (int)(clamp(y2, 0, model_in_h) / scale_h);
        group->results[last_count].prop = obj_conf;
        char* label = labels[id];
        strncpy(group->results[last_count].name, label, OBJ_NAME_MAX_SIZE);

        // printf("result %2d: (%4d, %4d, %4d, %4d), %s\n", i, group->results[last_count].box.left,
        // group->results[last_count].box.top,
        //        group->results[last_count].box.right, group->results[last_count].box.bottom, label);
        last_count++;
    }
    group->count = last_count;

    return 0;
}

int post_process__unquantization(float* input0, float* input1, float* input2, int model_in_h, int model_in_w,
    float conf_threshold, float nms_threshold, BOX_RECT pads, float scale_w, float scale_h,
    std::vector<int32_t>& qnt_zps, std::vector<float>& qnt_scales,
    detect_result_group_t* group)
{
    static int init = -1;
    if (init == -1) {
        int ret = 0;
        ret = loadLabelName(LABEL_NALE_TXT_PATH, labels);
        if (ret < 0) {
            return -1;
        }

        init = 0;
    }
    memset(group, 0, sizeof(detect_result_group_t));

    std::vector<float> filterBoxes;
    std::vector<float> objProbs;
    std::vector<int> classId;

    // stride 8
    int stride0 = 8;
    int grid_h0 = model_in_h / stride0;
    int grid_w0 = model_in_w / stride0;
    int validCount0 = 0;
    validCount0 = process_unquantization(input0, (int*)anchor0, grid_h0, grid_w0, model_in_h, model_in_w, stride0, filterBoxes, objProbs,
        classId, conf_threshold, qnt_zps[0], qnt_scales[0]);

    // stride 16
    int stride1 = 16;
    int grid_h1 = model_in_h / stride1;
    int grid_w1 = model_in_w / stride1;
    int validCount1 = 0;
    validCount1 = process_unquantization(input1, (int*)anchor1, grid_h1, grid_w1, model_in_h, model_in_w, stride1, filterBoxes, objProbs,
        classId, conf_threshold, qnt_zps[1], qnt_scales[1]);

    // stride 32
    int stride2 = 32;
    int grid_h2 = model_in_h / stride2;
    int grid_w2 = model_in_w / stride2;
    int validCount2 = 0;
    validCount2 = process_unquantization(input2, (int*)anchor2, grid_h2, grid_w2, model_in_h, model_in_w, stride2, filterBoxes, objProbs,
        classId, conf_threshold, qnt_zps[2], qnt_scales[2]);

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
    std::set<int> class_set(std::begin(classId), std::end(classId));

    std::cout << "模型输出的预测框,进入到nms的数量: " << validCount << std::endl;
#if USE_OPENCL

    float box_sorted[indexArray.size() * 4];
    float box_expand_cls[indexArray.size() * 4];
    int img_height_ori = 3024;
    int img_width_ori = 4016;
    const int max_wh = std::max(img_height_ori, img_width_ori);

#pragma omp parallel for
    for (int i = 0; i < indexArray.size(); i++) {
        int target_position = i * 4;
        int ori_position = indexArray[i];
        memcpy(box_sorted + target_position, &filterBoxes[0] + 4 * ori_position, sizeof(float) * 4);
        box_sorted[target_position + 2] += box_sorted[target_position]; // x1,y1,w,h->x1,y1, x2,y2
        box_sorted[target_position + 3] += box_sorted[target_position + 1];

        // 根据box类别对nms中的box进行偏移, 与yolo逻辑一致
        int box_cls = classId[i] + 1;
        int expand_ratio = box_cls * max_wh;
        box_expand_cls[target_position] = box_sorted[target_position] * expand_ratio;
        box_expand_cls[target_position + 1] = box_sorted[target_position + 1] * expand_ratio;
        box_expand_cls[target_position + 2] = box_sorted[target_position + 2] * expand_ratio;
        box_expand_cls[target_position + 3] = box_sorted[target_position + 3] * expand_ratio;
    }

    std::vector<int> keep_box;
    // opencl nms
    TICK(gpunms)
    if (nms_cl.Forward(box_expand_cls, validCount, nms_threshold, keep_box)) {
        return -2;
    }
    TOCK(gpunms)

    int last_count = 0;
    group->count = 0;
    for (int i = 0; i < validCount; ++i) {
        if (keep_box[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE) {
            continue;
        }
        int n = i;
        float x1 = box_sorted[n * 4 + 0];
        float y1 = box_sorted[n * 4 + 1];
        float x2 = box_sorted[n * 4 + 2];
        float y2 = box_sorted[n * 4 + 3];
        int   id = 0;
        float obj_conf = objProbs[n];
        if (obj_conf < conf_threshold) {
            continue;
        }

        RecoverBox(
            cv::Size(img_width_ori, img_height_ori),
            cv::Size(model_in_w, model_in_h),
            x1,
            y1,
            x2,
            y2,
            group->results[last_count].box.left,
            group->results[last_count].box.top,
            group->results[last_count].box.right,
            group->results[last_count].box.bottom,
            1);

        // group->results[last_count].box.left = (int)(clamp(x1, 0, model_in_w) / scale_w);
        // group->results[last_count].box.top = (int)(clamp(y1, 0, model_in_h) / scale_h);
        // group->results[last_count].box.right = (int)(clamp(x2, 0, model_in_w) / scale_w);
        // group->results[last_count].box.bottom = (int)(clamp(y2, 0, model_in_h) / scale_h);
        group->results[last_count].prop = obj_conf;
        char* label = labels[id];
        strncpy(group->results[last_count].name, label, OBJ_NAME_MAX_SIZE);
        last_count++;
    }
    group->count = last_count;
#else
    TICK(cpunms)
    for (auto c : class_set) {
        nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold);
    }
    TOCK(cpunms)
    int last_count = 0;
    group->count = 0;
    /* box valid detect target */
    for (int i = 0; i < validCount; ++i) {
        if (indexArray[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE) {
            continue;
        }
        int n = indexArray[i];

        float x1 = filterBoxes[n * 4 + 0] - pads.left;
        float y1 = filterBoxes[n * 4 + 1] - pads.top;
        float x2 = x1 + filterBoxes[n * 4 + 2];
        float y2 = y1 + filterBoxes[n * 4 + 3];
        int id = classId[n];
        float obj_conf = objProbs[i];

        group->results[last_count].box.left = (int)(clamp(x1, 0, model_in_w) / scale_w);
        group->results[last_count].box.top = (int)(clamp(y1, 0, model_in_h) / scale_h);
        group->results[last_count].box.right = (int)(clamp(x2, 0, model_in_w) / scale_w);
        group->results[last_count].box.bottom = (int)(clamp(y2, 0, model_in_h) / scale_h);
        group->results[last_count].prop = obj_conf;
        char* label = labels[id];
        strncpy(group->results[last_count].name, label, OBJ_NAME_MAX_SIZE);

        // printf("result %2d: (%4d, %4d, %4d, %4d), %s\n", i, group->results[last_count].box.left,
        // group->results[last_count].box.top,
        //        group->results[last_count].box.right, group->results[last_count].box.bottom, label);
        last_count++;
    }
    group->count = last_count;
#endif
    return 0;
}

void deinitPostProcess()
{
    for (int i = 0; i < OBJ_CLASS_NUM; i++) {
        if (labels[i] != nullptr) {
            free(labels[i]);
            labels[i] = nullptr;
        }
    }
#if USE_OPENCL
    nms_cl.DeInit();
#endif
}

void RecoverBox(const cv::Size& origin_size, const cv::Size& processed_size,
    const float& src_x1, const float& src_y1,
    const float& src_x2, const float& src_y2,
    int& x1, int& y1,
    int& x2, int& y2, bool letterbox)
{
    float x1_f, x2_f, y1_f, y2_f;
    if (letterbox) {
        const float in_w = origin_size.width;
        const float in_h = origin_size.height;
        float tar_w = processed_size.width;
        float tar_h = processed_size.height;
        float r = std::min(float(tar_h) / in_h, float(tar_w) / in_w);
        float inside_w = in_w * r;
        float inside_h = in_h * r;
        float padd_w = tar_w - inside_w;
        float padd_h = tar_h - inside_h;
        float top = (padd_h / 2 - 0.0000001f);
        float bottom = (padd_h - top + 0.1f);
        float left = (padd_w / 2 - 0.0000001f);
        float right = (padd_w - left + 0.1f);

        x1_f = src_x1 - left;
        x2_f = src_x2 - left;
        y1_f = src_y1 - top;
        y2_f = src_y2 - top;
        x1 = std::round(x1_f / r);
        x2 = std::round(x2_f / r);
        y1 = std::round(y1_f / r);
        y2 = std::round(y2_f / r);
    } else {
        x1_f = src_x1 / (float)processed_size.width;
        x2_f = src_x2 / (float)processed_size.width;
        y1_f = src_y1 / (float)processed_size.height;
        y2_f = src_y2 / (float)processed_size.height;
        x1 = x1_f * (float)origin_size.width;
        x2 = x2_f * (float)origin_size.width;
        y1 = y1_f * (float)origin_size.height;
        y2 = y2_f * (float)origin_size.height;
    }
    cv::Rect RectCLX;
    cv::Point c1(x1, y1);
    cv::Point c2(x2, y2);
    RectCLX = cv::Rect(c1, c2) & cv::Rect(0, 0, origin_size.width, origin_size.height);
    x1 = RectCLX.x;
    y1 = RectCLX.y;
    x2 = RectCLX.x + RectCLX.width;
    y2 = RectCLX.y + RectCLX.height;
}
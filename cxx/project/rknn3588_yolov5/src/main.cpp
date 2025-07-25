
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define _BASETSD_H
#include <iostream>


#include "utils.h"
#include "NmsCl.h"
int main(){

    std::cout << "进入程序" << std::endl;
    const int numBoxes = 50000; // 候选框数量
    const float iouThreshold = 0.35f; // IoU 阈值
    const int filterClassId = 0; // 类别过滤条件（例如只处理类别 0）
    // 生成模拟数据
    std::vector<float> outputLocations;
    std::vector<int> classIds;
    std::vector<float> prob_vec;

    // 初始化 order 数组
    std::vector<int> order(numBoxes);
    for (int i = 0; i < numBoxes; ++i) {
        order[i] = i; // 初始情况下所有候选框都是有效的
    }

    // 记录执行 NMS 的时间
    ALG_CL::NmsCl nms_cl;
    nms_cl.Init(R"(./nms.cl)");
    std::vector<int> keep_box;
    for(int i =0;i< 5;i++){
        utils::generateRandomData(numBoxes, outputLocations, prob_vec, classIds);
        // 记录执行 NMS 的时间
        TICK(cpunms)
        // 执行 NMS
        utils::nms(numBoxes, outputLocations, classIds, order, filterClassId, iouThreshold);
        TOCK(cpunms)
        TICK(gpunms)
        if (nms_cl.Forward(outputLocations.data(), numBoxes, iouThreshold, keep_box)) {
            return -2;
        }
        TOCK(gpunms)
    }

    // 统计剩余的有效候选框数量
    int validCountAfterNms = 0;
    for (const auto& idx : keep_box) {
        if (idx != -1) {
            validCountAfterNms++;
        }
    }
    nms_cl.DeInit();
    // 输出结果
    std::cout << "经过 NMS 后，剩余有效候选框数量: " << validCountAfterNms << std::endl;
    return EXIT_SUCCESS;
}
/**
 * @FilePath     : /connector/src/custom/Curved_Bow_Detect_Group.h
 * @Description  :
 * @Author       : naonao 1319144981@qq.com
 * @Version      : 0.0.1
 * @LastEditors  : naonao 1319144981@qq.com
 * @LastEditTime : 2024-01-08 13:25:50
 * @Copyright    : G AUTOMOBILE RESEARCH INSTITUTE CO.,LTD Copyright (c) 2024.
 **/
#pragma once
#include "../framework/BaseAlgoGroup.h"

class InferenceEngine;
class Z_MaleAlgoGroup : public BaseAlgoGroup {
public:
    Z_MaleAlgoGroup();
    virtual ~Z_MaleAlgoGroup();

private:
    DCLEAR_ALGO_GROUP_REGISTER(Z_MaleAlgoGroup)
};

/**
 * @FilePath     : /connector/src/custom/curved_bow_train_group.h
 * @Description  :
 * @Author       : error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
 * @Version      : 0.0.1
 * @LastEditors  : naonao 1319144981@qq.com
 * @LastEditTime : 2024-01-08 14:03:35
 * @Copyright    : G AUTOMOBILE RESEARCH INSTITUTE CO.,LTD Copyright (c) 2024.
 **/

#pragma once
#include "../framework/BaseAlgoGroup.h"

class InferenceEngine;

class Curved_Bow_TrainAlgoGroup : public BaseAlgoGroup
{
public:
    Curved_Bow_TrainAlgoGroup();
    virtual ~Curved_Bow_TrainAlgoGroup();

private:
    DCLEAR_ALGO_GROUP_REGISTER(Curved_Bow_TrainAlgoGroup)
};
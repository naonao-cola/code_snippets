#pragma once
#include "../framework/BaseAlgoGroup.h"
/**
 * @FilePath     : /snd_circlip/src/custom/ExampleAlgoGroup.h
 * @Description  :
 * @Author       : naonao 1319144981@qq.com
 * @Version      : 0.0.1
 * @LastEditors  : naonao 1319144981@qq.com
 * @LastEditTime : 2023-11-16 10:11:59
 * @Copyright    : G AUTOMOBILE RESEARCH INSTITUTE CO.,LTD Copyright (c) 2023.
**/

class InferenceEngine;
class CircleAlgoGroup : public BaseAlgoGroup
{
public:
    CircleAlgoGroup();
    virtual ~CircleAlgoGroup();

private:
    DCLEAR_ALGO_GROUP_REGISTER(CircleAlgoGroup)
};

#pragma once
#include "../../modules/tv_algo_base/src/framework/BaseAlgoGroup.h"

class InferenceEngine;
class FrontAlgoGroup : public BaseAlgoGroup
{
public:
    FrontAlgoGroup();
    virtual ~FrontAlgoGroup();

private:
    DCLEAR_ALGO_GROUP_REGISTER(FrontAlgoGroup)
};

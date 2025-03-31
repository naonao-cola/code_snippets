#pragma once
#include "../framework/BaseAlgoGroup.h"

class InferenceEngine;
class ExampleAlgoGroup : public BaseAlgoGroup
{
public:
    ExampleAlgoGroup();
    virtual ~ExampleAlgoGroup();

private:
    DCLEAR_ALGO_GROUP_REGISTER(ExampleAlgoGroup)
};

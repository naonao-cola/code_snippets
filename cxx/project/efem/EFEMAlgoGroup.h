#pragma once
#include "../framework/BaseAlgoGroup.h"

class InferenceEngine;
class EFEMAlgoGroup : public BaseAlgoGroup
{
public:
    EFEMAlgoGroup();
    virtual ~EFEMAlgoGroup();

private:
    DCLEAR_ALGO_GROUP_REGISTER(EFEMAlgoGroup)
};

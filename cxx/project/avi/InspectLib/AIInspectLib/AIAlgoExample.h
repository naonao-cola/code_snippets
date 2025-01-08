#pragma once
#include "AIAlgoBase.h"


class AIAlgoExample : public AIAlgoBase
{
public:
    DECLARE_SINGLETON(AIAlgoExample);
protected:
    virtual void OnPreProcess(TaskInfoPtr spTaskInfo);
    virtual void OnPostProcess(ModelResultPtr spResult, AlgoResultPtr& algoResult, HANDLE& hInspEnd);
public:
    bool is_init{ false };
private:
    HIDE_CREATE_METHODS(AIAlgoExample)
};


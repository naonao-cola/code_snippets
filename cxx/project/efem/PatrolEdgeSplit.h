#pragma once
#include "../framework/BaseAlgo.h"
#include "Define.h"
#include "PubFunc.h"
class PatrolEdgeSplit : public BaseAlgo, public PatrolEdge
{

public:
    PatrolEdgeSplit();
    ~PatrolEdgeSplit();
    AlgoResultPtr RunAlgo(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results);
	void RunAlgoSplit(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results, AlgoResultPtr algo_result, json judgeParams, std::vector<stBLOB_FEATURE>&	BlobResultTotal);
	
private:
    std::once_flag flag;
    //ÅÐ¶¨²ÎÊý
    STRU_DEFECT_ITEM EdgeDefectJudgment[MAX_JUDGE_NUM];
    DCLEAR_ALGO_GROUP_REGISTER(PatrolEdgeSplit)
};
#pragma once
#include "../framework/BaseAlgo.h"
#include "Define.h"
#include "PubFunc.h"


class PatrolEdgeBroken : public BaseAlgo, public PatrolEdge
{
public:
    PatrolEdgeBroken();
    ~PatrolEdgeBroken();
    AlgoResultPtr RunAlgo(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results);
	void RunAlgoBroken(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results, AlgoResultPtr algo_result, json judgeParams, std::vector<stBLOB_FEATURE>&	BlobResultTotal);
    void obviousBroken(cv::Mat Mask, int obviousThr, stBLOB_FEATURE& obviousFet);
    void JudgeConer(cv::Mat Coner);
private:
    std::once_flag flag;
    //ÅÐ¶¨²ÎÊý
    STRU_DEFECT_ITEM EdgeDefectJudgment[MAX_JUDGE_NUM];
    DCLEAR_ALGO_GROUP_REGISTER(PatrolEdgeBroken)
};
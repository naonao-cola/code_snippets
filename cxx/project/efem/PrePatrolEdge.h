/*****************************************************************//**
 * \file   PrePatrolEdge.h
 * \brief  
 * 
 * \author Ronnie
 * \date   May 2024
 *********************************************************************/
#pragma once
#include "../framework/BaseAlgo.h"
#include "PubFunc.h"


class PrePatrolEdge : public BaseAlgo, PatrolEdge
{
public:
    PrePatrolEdge();
    ~PrePatrolEdge();
    virtual AlgoResultPtr RunAlgo(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results);
    std::tuple<std::string, json> get_task_info(InferTaskPtr task, std::vector<AlgoResultPtr> pre_results, std::map<std::string, json> param_map);
private:
    DCLEAR_ALGO_GROUP_REGISTER(PrePatrolEdge)
  
};
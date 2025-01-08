#include "stdafx.h"
#include "AIAlgoExample.h"
#include "AIRuntime/AIRuntimeInterface.h"
#include "AIRuntime/logger.h"

#include <iostream>
#include <Windows.h>
#include <string>


AIAlgoExample::AIAlgoExample()
{
	m_nModelId = 0;
	LOG_INFO(">> AIAlgoExample()");
}

AIAlgoExample::~AIAlgoExample()
{
	LOG_INFO(">> ~AIAlgoExample()");
}

void AIAlgoExample::OnPreProcess(TaskInfoPtr spTaskInfo)
{
	auto spInspParam = std::static_pointer_cast<tAlgoInspParam>(spTaskInfo->inspParam);
	spInspParam->algoResults.tt.startTime = std::chrono::system_clock::now();
	spTaskInfo->taskId = spInspParam->nImageNum * 100 + spInspParam->nAlgNum;
	auto aiRuntime = GetAIRuntime();
	aiRuntime->CommitInferTask(spTaskInfo);

}


void AIAlgoExample::OnPostProcess(ModelResultPtr spResult, AlgoResultPtr& algoResult, HANDLE& hInspEnd)
{
	LOG_INFO("[AIAlgoExample] OnPostProcess image done! TaskId:[{}].", spResult->taskInfo->taskId);
	std::shared_ptr<stTaskInfo> taskInfo = spResult->taskInfo;
	auto spInspParam = std::static_pointer_cast<tAlgoInspParam>(taskInfo->inspParam);
	hInspEnd = spInspParam->hInspectEnd;

	auto pAlgoResults = &spInspParam->algoResults;

	pAlgoResults->tt.endTime = std::chrono::system_clock::now();
	pAlgoResults->tt.costTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(pAlgoResults->tt.endTime - pAlgoResults->tt.startTime).count();
	auto resultItem = std::make_shared<stModelResult>();

	
	pAlgoResults->vecResults.push_back(resultItem);
	//algoResult.reset(&pAlgoResults);
}
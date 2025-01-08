#pragma once
#include <memory>


#include "AIRuntime/AIRuntimeInterface.h"
#include "../../VSAlgorithmTask/DefineInterface.h"
#include "../FeatureExtraction.h"

using InspParamPtr = std::shared_ptr<void>;
using AlgoResultPtr = std::shared_ptr<void>;

class IAlgoResultListener
{
public:
	virtual void OnModelResult(AlgoResultPtr spResult) = 0;
};

class AIInspectAlgoInterface {
public:
	/**
	 * Start the inference task for Inspect detection.
	 * 
	 * \param spInspParam
	 * \return eAIErrorCode
	 */
	virtual  eAIErrorCode StartRunAlgo(InspParamPtr spInspParam) = 0;

	/**
	 * Initialize the Inspect algorithm and allocate the necessary resources..
	 * 
	 * \param preThrdCnt: The number of pre-processint threads.
	 * \param preThrdPriority: The priority of the pre-processing threads. The value of priority must be one of [-2, -1, 0, 1, 2].
	 * \param postThrdCnt: The number of post-processint threads.
	 * \param postThrdPriority: The priority of the poset-processing threads. The value of priority must be one of [-2, -1, 0, 1, 2].
	 */
	virtual void Initialize(int preThrdCnt, int preThrdPriority, int postThrdCnt, int postThrdPriority) = 0;

	/**
	 * Release the resources related to the Inspect algorithm..
	 */
	virtual void DeInitialize() = 0;

	virtual void RegisterAlgoResultListener(IAlgoResultListener* listener) = 0;
	virtual void UnRegisterAlgoResultListener(IAlgoResultListener* listener) = 0;

	void SetResultInfo(std::shared_ptr<vector<tBLOB_FEATURE>> pResultBlob, cv::Mat& matSrcBufferRGB) {
		m_pResultBlob = pResultBlob;
		m_matSrcBufferRGB = matSrcBufferRGB;
	}

protected:
	std::shared_ptr<vector<tBLOB_FEATURE>> m_pResultBlob;
	cv::Mat m_matSrcBufferRGB;
};

/**
 * The supported types of algorithms in the current algorithm library. When creating or deleting an algorithm, the algorithm types need to be updated.
 */
enum AIInspectAlgo {
	AIInspectAlgoDemo
};

/**
 * Return a singleton pointer of the Inspect algorithm.
 * \param algo: the type of algorithm
 */
AIInspectAlgoInterface* GetAIInspectAlgo(AIInspectAlgo algo);

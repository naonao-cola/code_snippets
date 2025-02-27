#include "stdafx.h"
#include "AIAlgoBase.h"
#include "AIAlgoDefines.h"
#include "AIRuntime/logger.h"
#include "AIRuntime/AIRuntimeDataStruct.h"
#include "AIRuntime/AIRuntimeInterface.h"
#include <mutex>

AIAlgoBase::AIAlgoBase():
	m_queuePrep(PREP_QUEUE_MAX_SIZE),
	m_queuePost(POST_QUEUE_MAX_SIZE),
	m_bStopped(false)
{
	LOG_INFO(">> AIAlgoBase()");
	m_pResultBlob = std::make_shared<vector<tBLOB_FEATURE>>();
}

AIAlgoBase::~AIAlgoBase()
{
	LOG_INFO(">> ~AIAlgoBase() +");
	DeInitialize();
	LOG_INFO(">> ~AIAlgoBase() -");
}

eAIErrorCode AIAlgoBase::StartRunAlgo(InspParamPtr spInspParam)
{
	if (m_vecPrepThrds.size() == 0 || m_vecPostThrds.size() == 0)
	{
		Initialize(DEF_PRE_THRD_CNT, DEF_PRE_THRD_PRIORITY, DEF_POST_THRD_CNT, DEF_POST_THRD_PRIORITY);
	}

	{
		std::unique_lock<std::mutex> l(m_queLock);
		while (!m_queuePrep.try_push(spInspParam))
		{
			LOG_INFOW("********* ERROR StartRunAlgo fail. queueSize: {}.", m_queuePrep.size());
			/*inspParam->inspectEnd
			return nullptr;*/
		}
		cond_.notify_one();
	}
	auto inspParam = std::static_pointer_cast<tAlgoInspParam>(spInspParam);
	LOG_INFO(">>>> START RUN ---- ImageNo: {}  algoNo:{}.", inspParam->nImageNum, inspParam->nAlgNum);
	return E_OK;
}

void AIAlgoBase::Initialize(int preThrdCnt, int preThrdPriority, int postThrdCnt, int postThrdPriority)
{
	LOG_INFO(">> AIAlgoBase::Initialize() +");
	if (m_vecPrepThrds.size() > 0 || m_vecPostThrds.size() > 0) {
		LOG_INFOW("Warning. Algo Already Initialized.");
		return;
	}
	if (preThrdCnt < 1) {
		LOG_INFOW("Wrong preprocess thread count param preThrdCnt: {}  {}", preThrdCnt, postThrdCnt);
	}
	if (postThrdCnt < 1) {
		std::cout << "Wrong postprocess thread count param preThrdCnt:" << preThrdCnt << "postThrdCnt:" << postThrdCnt << std::endl;
	}
	if (preThrdPriority < -2 || preThrdPriority > 2) {
		std::cout << "Wrong preprocess thread priority level:" << preThrdPriority << std::endl;
		preThrdPriority = 0;
	}
	if (postThrdPriority < -2 || postThrdPriority > 2) {
		std::cout << "Wrong postprocess thread priority level:" << postThrdPriority << std::endl;
		postThrdPriority = 0;
	}

	for (int i = 0; i < preThrdCnt; i++)
	{
		std::thread* thrd = new std::thread(&AIAlgoBase::PreProcessWorker, this);
		SetThreadPriority(thrd->native_handle(), preThrdPriority);
		m_vecPrepThrds.push_back(thrd);
	}
	for (int i = 0; i < postThrdCnt; i++)
	{
		std::thread* thrd = new std::thread(&AIAlgoBase::PostProcessWorker, this);
		SetThreadPriority(thrd->native_handle(), postThrdPriority);
		m_vecPostThrds.push_back(thrd);
	}

	GetAIRuntime()->RegisterResultListener(m_nModelId, this);
	LOG_INFO(">> AIAlgoBase::Initialize() -");
}

void AIAlgoBase:: DeInitialize()
{
	LOG_INFO(">> AIAlgoBase::DeInitialize() +");
	m_bStopped = true;
	cond_.notify_all();
	for (int i = 0; i < m_vecPrepThrds.size(); i++)
	{
		m_vecPrepThrds[i]->join();
		delete m_vecPrepThrds[i];
	}
	for (int i = 0; i < m_vecPostThrds.size(); i++)
	{
		m_vecPostThrds[i]->join();
		delete m_vecPostThrds[i];
	}
	if (m_vecPrepThrds.size() > 0) m_vecPrepThrds.clear();
	if (m_vecPostThrds.size() > 0) m_vecPostThrds.clear();

	auto aiRuntime = GetAIRuntime();
	aiRuntime->UnregisterResultListener(this);
	// 取消注册模型结果回调
	UnRegisterAlgoResultListener(NULL);
	LOG_INFO(">> AIAlgoBase::DeInitialize() -");
}

// 注册算法结果回调，当OnPostProcess返回结果时进行回调
void AIAlgoBase::RegisterAlgoResultListener(IAlgoResultListener* listener)
{
	if (listener != NULL) {
		m_vecResultListener.push_back(listener);
	}
}

// 取消注册结果回调
void AIAlgoBase::UnRegisterAlgoResultListener(IAlgoResultListener* listener)
{
	if (listener == NULL) {
		m_vecResultListener.clear();
	}
	else {

		auto iter = find(m_vecResultListener.begin(), m_vecResultListener.end(), listener);
		if (iter != m_vecResultListener.end()) {
			m_vecResultListener.erase(iter);
		}
	}
}

// 前处理工作线程函数
void AIAlgoBase::PreProcessWorker()
{
	while (true)
	{
		InspParamPtr inspParam;

		// wait for preprocess task
		{
			std::unique_lock<std::mutex> l(m_queLock);
			cond_.wait(l, [&](){
				LOG_INFO("recive one");
				return m_bStopped || !m_queuePrep.empty();
			});
			
			bool found = m_queuePrep.try_pop(inspParam);
			if (m_bStopped) break;

			if (!found) {
				//std::cout << "[PreProcessWorker] wait input image timeout!" << std::endl;

				continue;
			}
		}

		LOG_INFO("starting preprocess.....");
		// 算法预处理
		TaskInfoPtr spTaskInfo = make_shared<stTaskInfo>();
		spTaskInfo->inspParam = inspParam;
		spTaskInfo->modelId = m_nModelId;
		// test code
		std::shared_ptr<tAlgoInspParam> param = reinterpret_pointer_cast<tAlgoInspParam>(inspParam);
		spTaskInfo->imageData = {param->imgdata};
		// 调用子类实现的预处理方法
		//OnPreProcess(spTaskInfo);
		LOG_INFO("preprocess finish.....");
		// 预处理结束，调用AIRuntime模型推理
		auto aiRuntime = GetAIRuntime();
		aiRuntime->CommitInferTask(spTaskInfo);
		if (m_bStopped) break;
	}
}

// 后处理工作线程函数
void AIAlgoBase::PostProcessWorker()
{
	while (true)
	{
		ModelResultPtr pModelResult;
		bool found = m_queuePost.try_pop(pModelResult);
		if (m_bStopped) break;

		if (!found) {
			//std::cout << "[PostProcessWorker] wait model result timeout!" << std::endl;
			continue;
		}

		AlgoResultPtr algoResult;
		HANDLE hInspEnd = NULL;
		OnPostProcess(pModelResult, algoResult, hInspEnd);

		std::shared_ptr<stTaskInfo> taskInfo = pModelResult->taskInfo;
		auto spInspParam = std::static_pointer_cast<tAlgoInspParam>(taskInfo->inspParam);

		if (hInspEnd) {
			SetEvent(hInspEnd);
		}
		//spInspParam->inspectEnd->set_value(E_OK);
		LOG_INFO("Start executing callback funtions....");
		if (algoResult && m_vecResultListener.size() > 0)
		{
			for (auto iter = m_vecResultListener.begin(); iter != m_vecResultListener.end(); iter++)
			{
				(*iter)->OnModelResult(algoResult);
			}
		}

		if (m_bStopped) break;
	}
}



// AIRuntime 模型推理结果回调
void AIAlgoBase::OnModelResult(ModelResultPtr spResult)
{
	auto taskInfo = spResult->taskInfo;
	for(auto boxs : spResult->itemList) {
		for(auto box : boxs) {
			LOG_INFO("\n{}", box.Info());
		}
	}
	if (taskInfo->modelId != m_nModelId) {
		return;
	}

	if (!m_queuePost.try_push(spResult))
	{
		LOG_INFOW("Queue model result fail!");
	}
	WriteInBlob(spResult);
	DrawResult(spResult);
}

void AIAlgoBase::WriteInBlob(ModelResultPtr spResult)
{
	// write results of detection into pResultBlob

	//m_pResultBlob->From_AI = true;
	//m_pResultBlob->nImageNumber = spResult->taskInfo->orgImageId;
	std::vector<POINT> ptLT;
	std::vector<POINT> ptRT;
	std::vector<POINT> ptLB;
	std::vector<POINT> ptRB;
	std::vector<double>aiConfidence;
	std::vector<int>aiLabel;
	m_pResultBlob->clear();
	m_pResultBlob->shrink_to_fit();
	//int& nDefectCount = m_pResultBlob->nDefectCount;
	for (auto itemList : spResult->itemList) {
		for (auto item : itemList) {
			if (item.points.size() < 2) continue;
			tBLOB_FEATURE temp;
			long x1 = static_cast<long>(item.points[0].x);
			long y1 = static_cast<long>(item.points[0].y);
			long x2 = static_cast<long>(item.points[1].x);
			long y2 = static_cast<long>(item.points[1].y);
			temp.rectBox = cv::Rect(x1, y1, std::abs(x2 - x1), std::abs(y2 - y1));
			temp.fromAI = true;
			temp.confidence = item.confidence;
			temp.AICode = item.code;
			m_pResultBlob->push_back(temp);
		}
	}// end for
}

void AIAlgoBase::DrawResult(ModelResultPtr spResult)
{

}
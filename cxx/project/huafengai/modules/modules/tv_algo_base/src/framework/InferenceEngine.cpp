#include "InferenceEngine.h"
#include "BaseAlgoGroup.h"
#include "../utils/Logger.h"
#include "ErrorDefine.h"
#include "../utils/Utils.h"
#include "tv_license.h"

#include "TivalCore.h"
#include <AIRuntimeInterface.h>
#define TASK_THRD_POOL_SIZE_KEY "task_thread_pool_size"
#define ALGO_THRD_POOL_SIZE_KEY "algo_thread_pool_size"

#define LICENSE_SOLUTION_ID 2
#define MODULE_IDS          "100"

InferenceEngine::InferenceEngine()
	: m_task_thread_pool(nullptr),
	m_algo_thread_pool(nullptr),
	m_result_callback(nullptr),
	m_log_callback(nullptr),
	m_future_results(300)
{
	m_algo_mgr = AlgoManager::get_instance();

}

InferenceEngine::~InferenceEngine()
{
	//LOGI("InferenceEngine::~InferenceEngine ++");
	if (!m_stopped) {
		Destroy();
	}

	//LOGI("InferenceEngine::~InferenceEngine --");
}

ErrorCode InferenceEngine::Init()
{
//#if USE_LICENSE
//	int ret = TVLicense::Instance()->Init(LICENSE_SOLUTION_ID);
//	if (ret != 0) {
//		LOGE("InferenceEngine init fail. license error: {}", ret);
//		return ErrorCode::LICENSE_ERROR;
//
//	}
//#endif
	Tival::TivalCore::Init(2);
	return ErrorCode::OK;
}





ErrorCode InferenceEngine::LicenseVerify()
{
	//int ret = TVLicense::Instance()->Verify(MODULE_IDS);
	//if (ret != 0) {
	//	LOGE("License verify fail. modules:{} err:{}", std::string(MODULE_IDS), ret);
	//	return ErrorCode::LICENSE_ERROR;

	//}
	return ErrorCode::OK;
}

void InferenceEngine::Destroy()
{
	// 停止线程池
	if (m_task_thread_pool != nullptr)m_task_thread_pool->exit();
	if (m_algo_thread_pool != nullptr)m_algo_thread_pool->exit();
	// 停止回调线程
	m_stopped = true;
	{
		std::unique_lock<std::mutex> l(m_queue_lock);
		m_cond.notify_all();
	}
	if (m_result_thread != nullptr)  m_result_thread->join();
	// 清空提交task产生的future对象
	while (!m_future_results.empty()) {
		std::future<FinalResultPtr> fr;
		m_future_results.pop(fr);
	}
	// 释放算法对象资源
	m_algo_mgr->Destroy();
	// 清空参数json
	m_common_cfg = {};
	m_algo_cfg = {};
	// 销毁对象
	if (m_task_thread_pool != nullptr) delete m_task_thread_pool;
	if (m_algo_thread_pool != nullptr) delete m_algo_thread_pool;
	if (m_result_thread != nullptr) delete m_result_thread;
	m_task_thread_pool = nullptr;
	m_algo_thread_pool = nullptr;
	m_result_thread = nullptr;
#if USE_AI_DETECT
    GetAIRuntime()->DestoryRuntime();
#endif
    TVLicense::Instance()->Destroy();
}

ErrorCode InferenceEngine::ConfigSystemParams(const json& common_cfg)
{
	LOGI("ConfigSystemParams: {}", Utils::DumpJson(common_cfg));
	if (m_task_thread_pool != nullptr || m_algo_thread_pool != nullptr)
	{
		LOGE("Common params already configured!!");
		return ErrorCode::WRONG_STATE;
	}
	m_task_thread_pool = new ThreadPool(Utils::GetProperty(common_cfg, TASK_THRD_POOL_SIZE_KEY, 1));
	m_algo_thread_pool = new ThreadPool(Utils::GetProperty(common_cfg, ALGO_THRD_POOL_SIZE_KEY, 10));
	m_result_thread = new std::thread(&InferenceEngine::ResultCallbackWorker, this);
	m_log_level = (LogLv)Utils::GetProperty(common_cfg, "log_level", 0);
	m_common_cfg = std::move(common_cfg);
	m_stopped = false;

	// 初始化AIRuntime, 加载模型
    json ai_models = Utils::GetProperty(common_cfg, "ai_models", json::array());
    int ret = eAIErrorCode::E_OK;
    if (ai_models.size() > 0) {
        stAIConfigInfo ai_cfg(common_cfg);
        ret = GetAIRuntime()->InitRuntime(ai_cfg);

        if (ret != eAIErrorCode::E_OK) {
            LOGE("AIRuntime InitRuntime fail! code: {}", ret);
            return ErrorCode::AI_INIT_FAIL;
        }

        for (auto ai_param : ai_models) {
            stAIModelInfo model_cfg;
            json m_InferParam = Utils::GetProperty(ai_param, "InferParam", json::array());
            model_cfg.modelId = Utils::GetProperty(ai_param, "model_id", -1);
            model_cfg.modelVersion = Utils::GetProperty(ai_param, "model_version", 1);
            model_cfg.modelName = Utils::GetProperty(ai_param, "model_name", std::string(""));
            model_cfg.modelPath = Utils::GetProperty(ai_param, "model_path", std::string(""));
            model_cfg.modelBackend = Utils::GetProperty(ai_param, "model_backend", std::string("tensorrt"));
            model_cfg.inferParam.confidenceThreshold = Utils::GetProperty(ai_param, "conf_threshold", 0.5);
            model_cfg.inferParam.enableDetMat = Utils::GetProperty(m_InferParam, "enableDetMat", 0);
            model_cfg.inferParam.useDilat = Utils::GetProperty(m_InferParam, "useDilat", 1);
            model_cfg.inferParam.kernelSize = Utils::GetProperty(m_InferParam, "kernelSize", 3);
            model_cfg.modleLabelPath = Utils::GetProperty(ai_param, "label_path", std::string(""));
            model_cfg.algoType = (eAIAlgoType)Utils::GetProperty(ai_param, "algo_type", 1);
            model_cfg.inferParam.maxBatchSize        = Utils::GetProperty(ai_param, "batch_size", 1);
            model_cfg.inferParam.nmsThreshold        = Utils::GetProperty(ai_param, "nms_iou", 0.5);
            model_cfg.inferParam.optBatchSize        = Utils::GetProperty(ai_param, "optBatchSize", 1);
            model_cfg.inferParam.segThreshold        = Utils::GetProperty(ai_param, "segThreshold", 0.5);
            model_cfg.inferParam.construct_dim();

            // stAIModelInfo model_cfg;
            // model_cfg.modelId = Utils::GetProperty(ai_param, "model_id", -1);
            // model_cfg.modelVersion = Utils::GetProperty(ai_param, "model_version", 1);
            // model_cfg.modelName = Utils::GetProperty(ai_param, "model_name", std::string(""));
            // model_cfg.modelPath = Utils::GetProperty(ai_param, "model_path", std::string(""));
            // model_cfg.modelBackend = Utils::GetProperty(ai_param, "model_backend", std::string("tensorrt"));
            // model_cfg.inferParam.confidenceThreshold = Utils::GetProperty(ai_param, "conf_threshold", 0.5);
            // model_cfg.inferParam.maxBatchSize = Utils::GetProperty(ai_param, "batch_size", 1);
            // model_cfg.inferParam.nmsThreshold = Utils::GetProperty(ai_param, "nms_iou", 0.5);
            // model_cfg.inferParam.maxObjectNums = Utils::GetProperty(ai_param, "max_obj_num", 100);
            // model_cfg.modleLabelPath = Utils::GetProperty(ai_param, "label_path", std::string(""));
            // model_cfg.algoType = (eAIAlgoType)Utils::GetProperty(ai_param, "algo_type", 1);

            if (model_cfg.modelId < 0) {
                LOGE("Wrong model ID: {}", model_cfg.modelId);
                continue;
            }

            ret = GetAIRuntime()->CreateModle(model_cfg);
            if (ret != eAIErrorCode::E_OK) {
                LOGE("AIRuntime CreateModle fail! err:{} model_name:{} ret:{}", model_cfg.modelName, model_cfg.modelId, ret);
                return ErrorCode::AI_INIT_FAIL;
            }

            LOGI("AIRuntime CreateModle success! {}", model_cfg.modelPath);

            if (model_cfg.dims.size() >= 4) {
                int n = model_cfg.dims[0];
                int c = model_cfg.dims[1];
                int h = model_cfg.dims[2];
                int w = model_cfg.dims[3];
                LOGI("Model input shapes: {} x {} x {} x {}", n, c, h, w);
                // 模型预热
                TaskInfoPtr _task = std::make_shared<stTaskInfo>();
                cv::Mat image(h, w, CV_8UC(c));
                _task->imageData  = { image };
                _task->modelId    = model_cfg.modelId;
                _task->taskId     = 0;
                GetAIRuntime()->RunInferTask(_task);
            }
        }
    }
	return ErrorCode::OK;
}

ErrorCode InferenceEngine::ConfigAlgoParams(const json& algo_all_cfg)
{
	LOGI("AlgoConfig: {}", Utils::DumpJson(algo_all_cfg));
	ErrorCode ecode = m_algo_mgr->ConfigAlgoParams(algo_all_cfg);
	if (ecode == ErrorCode::OK) {
		m_algo_cfg = std::move(algo_all_cfg);
	}
	return ecode;
}

ErrorCode InferenceEngine::CommitInferTask(InferTaskPtr task)
{
	LOGI("CommitInferTask: 000");
	LOGI("CommitInferTask: {}", Utils::DumpJson(task->image_info));
	if (m_task_thread_pool == nullptr) {
		return ErrorCode::NOT_READY;
	}
	LOGI("CommitInferTask: 111");
	if (!m_future_results.try_push(m_task_thread_pool->enqueue([=]() {
		return this->RunTask(task); }))) {

		LOGE("********* ERROR CommitInferTask fail. result queue full: {}.", m_future_results.size());
		return ErrorCode::QUEUE_OVERFLOW;
	}
	LOGI("CommitInferTask: 222");
	m_cond.notify_one();
	return ErrorCode::OK;
}

void InferenceEngine::RegisterResultCallback(ResultCallbackFunc callback)
{
	LOGI("RegisterResultCallback success!");
	m_result_callback = callback;
}

void InferenceEngine::RegisterLogCallback(LogCallbackFunc callback)
{
	LOGI("RegisterLogCallback success!");
	m_log_callback = callback;
}

void InferenceEngine::ReportLog(LogLv level, std::string msg)
{
	if (level >= m_log_level && m_log_callback != nullptr)
	{
		m_log_callback(int(level), msg.c_str());
	}
}

FinalResultPtr InferenceEngine::RunTask(InferTaskPtr task)
{
	LOGI("RunTask: 11");
	std::string type_id = task->image_info["type_id"];
	std::string img_name = task->image_info["img_name"];
	LOGI("RunTask: 22");
	LOGI("RunTask start, type_id:{} img_name:{}", type_id, img_name);

	BaseAlgoGroup* pGroup = m_algo_mgr->GetAlgoGroupByID(type_id);
	if (pGroup == nullptr)
	{
		LOGE("RunTask fail!! Can not find AlgoGroup for type_id:{}", type_id);
		return {};
	}

	FinalResultPtr final_result = pGroup->RunGroup(task);
	LOGI("RunTask complete. type_id:{} img_name:{}", type_id, img_name);
	LOGI("RunTask: 33");
	return final_result;
}

void InferenceEngine::ResultCallbackWorker()
{
	while (true)
	{
		// {} 作用域用于限制lock的释放，否则commit_task会被阻塞
		{
			std::unique_lock<std::mutex> lock(m_queue_lock);
			m_cond.wait(lock, [&]() {
				return m_stopped || !m_future_results.empty();
				});
		}

		if (m_stopped) break;

		std::future<FinalResultPtr> future_results;
		//Sleep(10);
		bool found = m_future_results.try_pop(future_results);
		bool is_valid = future_results.valid();


		if (found && !is_valid) {
			LOGE("future_results is null. SKIP.")
				LOGE("m_future_results size {}.", m_future_results.size());

		}

		FinalResultPtr results;
		try {
			results = future_results.get();
		}
		catch (const std::exception& ex) {

			std::cout << ex.what() << std::endl;;
		}


		if (m_result_callback == nullptr) {
			LOGW("Result callback is null. SKIP.")
		}
		else if (results == nullptr) {
			LOGW("Result is null. SKIP.");
		}
		else {
			m_result_callback(Utils::DumpJson(results->image_info).c_str(), Utils::DumpJson(results->results).c_str());
		}
	}
	LOGI("Result thread exit.");
}
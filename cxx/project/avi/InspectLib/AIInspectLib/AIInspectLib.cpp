#include "stdafx.h"

#include "AIInspectLib.h"
#include "AIAlgoDefines.h"
#include "AIInspectDefine.h"
#include "AIInpectAlgoInterface.h"

#include "../../VSAlgorithmTask/Define.h"
#include "../InspectLibLog.h"

#include "AIRuntime/AIRuntimeDataStruct.h"
#include "AIRuntime/AIRuntimeInterface.h"
#include "AIRuntime/AIRuntimeUtils.h"
#include "AIRuntime/logger.h"
#include "../../VSAlgorithmTask/DICS_B11.h"

#include <opencv2/opencv.hpp>

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

#include "../InspectActive.h"
#include "../FeatureExtraction.h"

InspectLibLog		cAIInspectLibLog;



using namespace std;

bool ReadAIModelInfo(int modelId, const char* modelInfoJsonPath, json& modelInfo) {
	json modelAllInfo = read_json_from_file(modelInfoJsonPath);
	if (modelAllInfo == NULL) return false;
	modelInfo = get_param<json>(modelAllInfo, std::to_string(modelId), NULL);
	return true;
}

extern "C" AFX_API_EXPORT   long	AI_ModelSetParamter(double* dPara) {
	// Initilize AIRuntime	
	double testParm[] = { 1, 1, 1024, 1024, 1024, 1, 0, 1, 0, 1, 0, 1, 10, 0.4, 0.2, 1024, 0 };
	if (dPara == nullptr) {
		dPara = testParm;
	}
	stAIConfigInfo runtimeIni;
	if (dPara != nullptr) {
		runtimeIni.usePinMemory				= static_cast<bool>(dPara[AI_INSPECTION_PARAM_USEPINMEM]);
		runtimeIni.workSpaceSize			= static_cast<int>(dPara[AI_INSPECTION_PARAM_WORKSPACESIZE]);
		runtimeIni.GPUCachSize				= static_cast<int>(dPara[AI_INSPECTION_PARAM_GPUCACHSIZE]);
		runtimeIni.CPUCachSize				= static_cast<int>(dPara[AI_INSPECTION_PARAM_CPUCACHSIZE]);
		runtimeIni.inferThreadCnt			= static_cast<int>(dPara[AI_INSPECTION_PARAM_PREPROCESSTHREADCNT]);
		runtimeIni.preProcessThreadPriority = static_cast<int>(dPara[AI_INSPECTION_PARAM_PREPROCESSTHREADPRI]);
		runtimeIni.inferThreadCnt			= static_cast<int>(dPara[AI_INSPECTION_PARAM_HOSTPROCESSTHREADCNT]);
		runtimeIni.inferThreadPriority		= static_cast<int>(dPara[AI_INSPECTION_PARAM_HOSTPROCESSTHREADPRI]);
	}

	auto obj = GetAIRuntime();
	json config = json::object();
	stAIModelInfo model;
	if (dPara != nullptr) {
		config["modelID"] = static_cast<int>(dPara[AI_INSPECTION_PARAM_MODELID]);
		//model.modelId = static_cast<int>(dPara[AI_INSPECTION_PARAM_MODELID]);
		int type = static_cast<int>(dPara[AI_INSPECTION_PARAM_MODELVERSION]);
		config["type"] = type;
		config["model_version"]			= static_cast<int>(dPara[AI_INSPECTION_PARAM_MODELVERSION]);
		config["gpuId"]					= { static_cast<int>(dPara[AI_INSPECTION_PARAM_GPUINDEX]) };
		config["max_batch_size"]		= static_cast<int>(dPara[AI_INSPECTION_PARAM_MAXBATCHSIZE]);
		config["confidence_threshold"]	= static_cast<float>(dPara[AI_INSPECTION_PARAM_CONFIDENCETHR]);
		config["nms_threshold"]			= static_cast<float>(dPara[AI_INSPECTION_PARAM_NMSTHR]);
		config["max_objects"]			= static_cast<int>(dPara[AI_INSPECTION_PARAM_MAXOBJECTNUMS]);
	}

	obj->UpdateModleParam(config);
	return 0;
}

class CbWriteResultBlob :public IAlgoResultListener {
public:
	CbWriteResultBlob(stDefectInfo* pResultBlob, cv::Mat& matSrcBufferRGB) :m_pResultBlob(pResultBlob), m_matSrcBufferRGB(matSrcBufferRGB) {}
	virtual void OnModelResult (AlgoResultPtr spResultt) override
	{
		// write results of detection into pResultBlob
		auto spResultList = reinterpret_pointer_cast<tAlgoResults>(spResultt);
		for (auto spResult : spResultList->vecResults) {
			m_pResultBlob->From_AI = true;
			m_pResultBlob->nImageNumber = spResult->taskInfo->orgImageId;
			std::vector<POINT> ptLT;
			std::vector<POINT> ptRB;
			std::vector<double>aiConfidence;
			std::vector<int>aiLabel;

			for (auto itemList : spResult->itemList) {
				for (auto item : itemList) {
					for (auto point : item.points) {
						POINT lt;
						lt.x = static_cast<long>(point.x);
						lt.y = static_cast<long>(point.y);
						ptLT.emplace_back(lt);

						POINT rb;
						rb.x = static_cast<long>(point.x);
						rb.y = static_cast<long>(point.y);
						ptRB.emplace_back(rb);
						aiConfidence.emplace_back(item.confidence);
						aiLabel.emplace_back(item.code);
					}
				}
			}// end for
		}
	}

private:
	stDefectInfo* m_pResultBlob;
	cv::Mat m_matSrcBufferRGB;

};

/**
 * AI Inspect Demo.
 * 
 * \param matSrcBuffer				src image mat
 * \param matSrcBufferRGB			render image mat
 * \param EngineerDefectJudgment	
 * \param pResultBlob				detect result
 * \param strAlgLog					the name of algorithm
 */

extern "C" AFX_API_EXPORT long AI_DICSInspect(cv::Mat & matSrcBuffer,cv::Mat & matSrcBufferRGB,int* nCommonPara,wchar_t strPath[][1024],STRU_DEFECT_ITEM * EngineerDefectJudgment,stDefectInfo * pResultBlob,wchar_t* strAlgLog,cv::Rect cutRoi) {
	clock_t tBeforeTime = cAIInspectLibLog.writeInspectLog(E_ALG_TYPE_COMMON_LIB, __FUNCTION__, _T("Start"), strAlgLog);
	cAIInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("Buf Start"));
	long	nErrorCode = E_ERROR_CODE_TRUE;
	if (matSrcBuffer.empty())	return E_ERROR_CODE_EMPTY_BUFFER;

	HANDLE endEvents[2];
	auto InspectAlgoDemo = GetAIInspectAlgo(AIInspectAlgo::AIInspectAlgoDemo);

	// Save results in vector<tBLOB_FEATURE>
	std::shared_ptr<vector<tBLOB_FEATURE>> detResult = std::make_shared<vector<tBLOB_FEATURE>>();
	InspectAlgoDemo->SetResultInfo(detResult, matSrcBufferRGB);
	//matSrcBuffer.copyTo(matSrcBufferRGB);
	//InspectAlgoDemo->RegisterAlgoResultListener(new CbWriteResultBlob(pResultBlob, matSrcBufferRGB));
	auto inspParam = std::make_shared<tAlgoInspParam>();
	inspParam->imgdata = matSrcBuffer.clone();

	inspParam->hInspectEnd = CreateEvent(NULL, TRUE, FALSE, NULL);
	endEvents[0] = inspParam->hInspectEnd;

	auto rst_future = InspectAlgoDemo->StartRunAlgo(inspParam);
	int nRet = WaitForMultipleObjects(1, endEvents, TRUE, 15000);
	if (nRet == WAIT_OBJECT_0)
	{
		tAlgoResults results = inspParam->algoResults;
		if (results.vecResults.size() == 0) {
			int x = 0;
		}
		int taskId = inspParam->nImageNum * 100 + inspParam->nAlgNum;
		cAIInspectLibLog.writeInspectLog(E_ALG_TYPE_COMMON_LIB, __FUNCTION__, _T("Algo [%d]\tcode:{%d}\tTT:{%d}ms"), strAlgLog);
	}
	else if (nRet == WAIT_TIMEOUT)
	{
		cAIInspectLibLog.writeInspectLog(E_ALG_TYPE_COMMON_LIB, __FUNCTION__, _T("------- WAIT_TIMEOUT ------"), strAlgLog);
	}
	cAIInspectLibLog.writeInspectLogTime(E_ALG_TYPE_COMMON_LIB, tBeforeTime, __FUNCTION__, _T("End"));


	CFeatureExtraction cFeatureExtraction;
	cFeatureExtraction.SetMem(NULL);
	cv::Mat matJudegeBuf, matJudegeResBuf;

	nErrorCode = cFeatureExtraction.DoDefectAIDectectJudgment(matSrcBuffer,matSrcBufferRGB,nCommonPara, E_DEFECT_COLOR_DARK, _T("Color_Dirty_Mura"), EngineerBlockDefectJudge, pResultBlob, detResult,cutRoi,0);

		//绘制结果轮廓(Light SkyBlue)
	cFeatureExtraction.DrawBlob(matSrcBufferRGB, cv::Scalar(135, 206, 250), BLOB_DRAW_BOUNDING_BOX, true);

		if (!USE_ALG_CONTOURS)	//保存结果轮廓
		cFeatureExtraction.SaveTxt(nCommonPara, strPath[1], true);
	return nErrorCode;

}

/**
 * Initilize the AI Runtime and Create AI Models by Config.
 * 
 * \param (type: json) AI Runtime Config contain of AI Runtime Config and AI models config.
 * \return 
 */
extern "C" AFX_API_EXPORT long AI_Initialization(const std::string& fconfig)
{
	//Parse the config
	auto config = read_json_from_file(fconfig.c_str());

	// Initilize the AI Runtime
	stAIConfigInfo aiIniConfig(config["initConfig"]);
	GetAIRuntime()->InitRuntime(aiIniConfig);

	// Initilize AI models.
	GetAIRuntime()->CreateModle(config["modelInfo"]);
	return 0;
}

extern "C" AFX_API_EXPORT	long	AI_CompileTensorRTModel(double* dPara, int* nCommonPara, wchar_t* strAlgPath, wchar_t* strAlgLog) 
{
	return 0;
}


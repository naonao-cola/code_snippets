#pragma once
#ifndef AIINSPINFO_H
#define AIINSPINFO_H

#include "Define.h"
#include "DefineInterface_SITE.h"
#include "stdafx.h"
#include "VSAlgorithmTask.h"
#include "../InspectLib/AIInspectLib/AIRuntime/AIRuntimeDataStruct.h"
#include "../InspectLib/AIInspectLib/AIRuntime/AIRuntimeInterface.h"
#include "../InspectLib/AIInspectLib/AIRuntime/AIRuntimeUtils.h"

#define MAX_MODEL_COUNT 5

//AI算法参数struct
enum ENUM_PARA_AI_ALL_PARAMS
{

	E_MURA3_ENABLE = 93,
	E_MURA3_MB_MODELID,
	E_MURA3_MD_MODELID,
	E_MURA3_CONFIDENCE,
	E_MURA3_REJUDGE,

	E_MURA4_ENABLE = 22,
	E_MURA4_MB_MODELID,
	E_MURA4_MD_MODELID,
	E_MURA4_CONFIDENCE,
	E_MURA4_REJUDGE,

	E_MURA_WHITE_ENABLE = 118,
	E_MURA_WHITE_MB_MODELID,
	E_MURA_WHITE_MD_MODELID,
	E_MURA_WHITE_CONFIDENCE,
	E_MURA_WHITE_REJUDGE,

	E_MURA_GRAY64_ENABLE = 129,
	E_MURA_GRAY64_MB_MODELID,
	E_MURA_GRAY64_MD_MODELID,
	E_MURA_GRAY64_CONFIDENCE,
	E_MURA_GRAY64_REJUDGE
};

struct AIReJudgeParam
{
	BOOL AIEnable{ 0 };
	int modelID[MAX_MODEL_COUNT]{0};
	double confidence {0};
	bool rejudge{ false };
};

//AI 结果struct
struct STRU_AI_INFO
{
	CString	strPanelID;
	CString imgName;
	int		imgNum;
	int		algoNum;
	CString	defectCode;		
	cv::Point	defectLocation[2];
	CString		label;
	double confidence;	
	cv::Mat detect_mat;
	std::vector<int> defectNoList;

	STRU_AI_INFO()
	{
		strPanelID = _T("");
		imgName = _T("");
		imgNum = 0;
		algoNum = 0;
		defectCode = _T("");
		label = _T("");
		confidence = 0.0;
		for (int i = 0; i < 2; i++) {
			defectLocation[i].x = 0.0;
			defectLocation[i].y = 0.0;
		}
	}
};

using AIInfoPtr = std::shared_ptr<STRU_AI_INFO>;

class AIReJudge {
public:
	AIReJudge();
    ~AIReJudge();

	void AICSVSave(CString csvfile, int img_pos, AIInfoPtr ai_info);
	void AIDetectImgSave(TaskInfoPtr spTaskInfo, CString sava_path);
	void AIDetectImgSave(int num, CString savePath, AIInfoPtr ai_info);
	AIReJudgeParam GetAIParam(double* dPara, const int algo_num, const int img_num);
	void AIReslutSave(CString saveDir, int img_pos, AIInfoPtr ai_info, AIReJudgeParam conf);

    //多张结果返回
	void SaveInferenceResultMult(CString csvfile, CString saveDir, AIReJudgeParam conf, std::vector<std::vector<stResultItem>> itemList, std::shared_ptr<stTaskInfo> taskInfo);

public:
    std::string LABEL[2] = { "ME0300&MU300_NG", "ME0300&MU300_OK" };

};

#endif
#include "stdafx.h"
#include "AIReJudge.h"
//#include "stdafx.h"

AIReJudge::AIReJudge() {

}

AIReJudge::~AIReJudge() {

}

//void AIReJudge::OnModelResult(ModelResultPtr spResult){
//    theApp.WriteLog(eLOGTEST, eLOGLEVEL_SIMPLE, FALSE, TRUE, _T("AI Calbk ResultNum. Num = %d"), spResult->itemList.size());
//    //SaveInferenceResultOnce(spResult->itemList);
//	SaveInferenceResultMult(spResult->itemList, spResult->taskInfo);
//}

void AIReJudge::SaveInferenceResultMult(CString csvfile, CString saveDir, AIReJudgeParam conf, std::vector<std::vector<stResultItem>> itemList, std::shared_ptr<stTaskInfo> taskInfo) {
	auto inspParam = std::static_pointer_cast<STRU_AI_INFO>(taskInfo->inspParam);
	int img_pos = 0;
	for (auto imgRst : itemList) {
		for (auto item : imgRst) {

			CString cstr(LABEL[item.code].c_str());
			theApp.WriteLog(eLOGTEST, eLOGLEVEL_SIMPLE, FALSE, TRUE, _T("AI Rturn Result. cls = %s,   confidence = %0.3f"), cstr, item.confidence);
						//1、if (conf > algo_param[AI置信度位置])

			inspParam->confidence = item.confidence;
			inspParam->label = cstr;
			inspParam->detect_mat = taskInfo->imageData[img_pos];
						//保存缺陷图像
			AIReslutSave(saveDir, img_pos, inspParam, conf);
						//2、存放复判的结果 （OK>0.975  NG>0.975 Unknown<0.975）·
			//AICSVSave(csvfile, img_pos, inspParam);
			img_pos++;
		}
	}
}

void AIReJudge::AICSVSave(CString csvfile, int img_pos, AIInfoPtr ai_info)
{
	FILE* out = NULL;

	if (img_pos == 0)
		_wfopen_s(&out, csvfile, _T("wt"));
	else
		_wfopen_s(&out, csvfile, _T("at"));

	if (out == NULL)		return;

	if (img_pos == 0)
	{
		fprintf_s(out, "No					,\
							   PANEl_ID      ,\
								Img_Color ,\
 						       AI_cls		    ,\
							  AI_confidence		,,");
		fprintf_s(out, "\n");
	}
	char strPanelID[MAX_PATH] = { 0, };
	memset(strPanelID, 0, sizeof(char) * MAX_PATH);
	WideCharToMultiByte(CP_ACP, 0, ai_info->strPanelID, -1, strPanelID, sizeof(strPanelID), NULL, NULL);

	char imgName[MAX_PATH] = { 0, };
	memset(imgName, 0, sizeof(char) * MAX_PATH);
	WideCharToMultiByte(CP_ACP, 0, ai_info->imgName, -1, imgName, sizeof(imgName), NULL, NULL);

	char label[MAX_PATH] = { 0, };
	memset(label, 0, sizeof(char) * MAX_PATH);
	WideCharToMultiByte(CP_ACP, 0, ai_info->label, -1, label, sizeof(label), NULL, NULL);
	fprintf_s(out, "%d,%s,%s,%s,%0.3f,",
		img_pos,
		strPanelID,
		imgName,
		label,
		ai_info->confidence
	);
	fprintf_s(out, "\n");
	fclose(out);
	out = NULL;
}

void AIReJudge::AIDetectImgSave(TaskInfoPtr spTaskInfo, CString sava_path) {

	//cv::imwrite()
	int i = 0;
	CString save_num;

	for (auto img : spTaskInfo->imageData) {
		CString sava_JPG = sava_path;
		save_num.Format(_T("\\%d.jpg"), i);
		sava_JPG.Append(save_num);
		cv::imwrite((LPCSTR)(CStringA)sava_JPG.GetBuffer(), img);
	}
}
void AIReJudge::AIDetectImgSave(int num, CString sava_path, AIInfoPtr ai_info) {
		CString save_info;
		CString sava_JPG = sava_path;
		save_info.Format(_T("\\%s_%s_%s_%d.jpg"), ai_info->strPanelID, ai_info->imgName, ai_info->label,  num);
		sava_JPG.Append(save_info);
		cv::imwrite((LPCSTR)(CStringA)sava_JPG.GetBuffer(), ai_info->detect_mat);
}

AIReJudgeParam AIReJudge::GetAIParam(double* dPara, const int algo_num, const int img_num)
{
	AIReJudgeParam AIParam;
	switch (algo_num)
	{
	case E_ALGORITHM_MURA_NORMAL:
		break;
	case E_ALGORITHM_MURA:
	{
		if (img_num == E_IMAGE_CLASSIFY_AVI_GRAY_64)
		{
			AIParam.AIEnable = dPara[E_MURA_GRAY64_ENABLE];
			AIParam.modelID[0] = dPara[E_MURA_GRAY64_MB_MODELID];
			AIParam.modelID[1] = dPara[E_MURA_GRAY64_MD_MODELID];
			AIParam.confidence = dPara[E_MURA_GRAY64_CONFIDENCE];
			AIParam.rejudge = dPara[E_MURA_GRAY64_REJUDGE];
		}
		else if (img_num == E_IMAGE_CLASSIFY_AVI_WHITE)
		{
			AIParam.AIEnable = dPara[E_MURA_WHITE_ENABLE];
			AIParam.modelID[0] = dPara[E_MURA_WHITE_MB_MODELID];
			AIParam.modelID[1] = dPara[E_MURA_WHITE_MD_MODELID];
			AIParam.confidence = dPara[E_MURA_WHITE_CONFIDENCE];
			AIParam.rejudge = dPara[E_MURA_WHITE_REJUDGE];
		}
		break;
	}

	case E_ALGORITHM_MURA3:
	{
		AIParam.AIEnable = dPara[E_MURA3_ENABLE];
		AIParam.modelID[0] = dPara[E_MURA3_MB_MODELID];
		AIParam.modelID[1] = dPara[E_MURA3_MD_MODELID];
		AIParam.confidence = dPara[E_MURA3_CONFIDENCE];
		AIParam.rejudge = dPara[E_MURA3_REJUDGE];
		break;
	}

	case E_ALGORITHM_MURA4:
	{
		AIParam.AIEnable = dPara[E_MURA4_ENABLE];
		AIParam.modelID[0] = dPara[E_MURA4_MB_MODELID];
		AIParam.modelID[1] = dPara[E_MURA4_MD_MODELID];
		AIParam.confidence = dPara[E_MURA4_CONFIDENCE];
		AIParam.rejudge = dPara[E_MURA4_REJUDGE];
		break;
	}

	default:
		break;
	}
	return AIParam;
}

void AIReJudge::AIReslutSave(CString saveDir, int img_pos, AIInfoPtr ai_info, AIReJudgeParam conf)
{
	//OK filepath
	CString okPath = saveDir + _T("\\OK");
	CreateDirectory(okPath, NULL);
	//NG
	CString ngPath = saveDir + _T("\\NG");
	CreateDirectory(ngPath, NULL);
	//unknown
	CString unknownPath = saveDir + _T("\\UnKnown");
	CreateDirectory(unknownPath, NULL);

	if (ai_info->confidence > conf.confidence) {
				//AI 复判为NG的缺陷  过滤
		if (ai_info->label == LABEL[0].c_str()) {
			AIDetectImgSave(img_pos, ngPath, ai_info);
		}
				//AI 复判为OK的缺陷  过滤
		else {
			AIDetectImgSave(img_pos, okPath, ai_info);
		}
	}
	else {
				//AI 复判为不确定的缺陷 
		AIDetectImgSave(img_pos, unknownPath, ai_info);
	}
}
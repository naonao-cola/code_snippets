#include "stdafx.h"
#include "AIInspInfo.h"
//#include "stdafx.h"


ModelResultListenerME::ModelResultListenerME() {

}

ModelResultListenerME::~ModelResultListenerME() {

}

void ModelResultListenerME::OnModelResult(ModelResultPtr spResult){
    theApp.WriteLog(eLOGTEST, eLOGLEVEL_SIMPLE, FALSE, TRUE, _T("AI Calbk ResultNum. Num = %d"), spResult->itemList.size());
    SaveInferenceResultOnce(spResult->itemList);
}

void ModelResultListenerME::SaveInferenceResultOnce(std::vector<std::vector<stResultItem>> itemList) {
	if (*nImageDefectCount == 0)
		_wfopen_s(&out, strfilePath, _T("wt"));
    for (auto i : itemList[0]) {
        double conf = 1 / (1 + exp(-i.confidence));
        CString cstr(LABEL[i.code].c_str());
        theApp.WriteLog(eLOGTEST, eLOGLEVEL_SIMPLE, FALSE, TRUE, _T("AI Rturn Result. cls = %s,   confidence = %0.3f"), cstr, conf);
        //AIReslutSave()
		if (out != NULL) {
			fprintf_s(out, "%d,%d,%d,%s,%f,",
				(*nImageDefectCount)++,
				cstr,
				conf
			);
		}

    }
		//关闭本地文件写入
	if (out != NULL) {
		fclose(out);
		out = NULL;
		*nImageDefectCount = NULL;
	}
}

void ModelResultListenerME::SaveInferenceResultMult(std::vector<std::vector<stResultItem>> itemList) {

    for (auto i : itemList) {
        for (auto j : i) {
            double conf = 1 / (1 + exp(-j.confidence));
            CString cstr(LABEL[j.code].c_str());
            theApp.WriteLog(eLOGTEST, eLOGLEVEL_SIMPLE, FALSE, TRUE, _T("AI Rturn Result. cls = %s,   confidence = %0.3f"), cstr, conf);
        }
    }
}

void ModelResultListenerME::AIReslutSave(stDefectInfo* resultBlob)
{
	if (resultBlob == NULL)	return;

	int	nDefectNum = 0;

	

	if (nImageDefectCount == NULL)
		nImageDefectCount = &nDefectNum;

	if (*nImageDefectCount == 0)
		_wfopen_s(&out, strfilePath, _T("wt"));
	else
		_wfopen_s(&out, strfilePath, _T("at"));

	if (out == NULL)		return;

	if (*nImageDefectCount == 0)
	{
		fprintf_s(out, "No					,\
 						AI_cls				,\
						AI_confidence		,,");
		fprintf_s(out, "\n");
	}
	//char szPath[MAX_PATH] = { 0, };
	//for (int nFori = 0; nFori < resultBlob->nDefectCount; nFori++)
	//{
	//	memset(szPath, 0, sizeof(char) * MAX_PATH);
	//	WideCharToMultiByte(CP_ACP, 0, theApp.GetDefectTypeName(resultBlob->nDefectJudge[nFori]), -1, szPath, sizeof(szPath), NULL, NULL);

	//	//USES_CONVERSION;
	//	//char *cstrName = W2A( theApp.GetDefectTypeName(resultBlob->nDefectJudge[nFori]) );

	//	fprintf_s(out, "%d,%s,%d,%d,%d,",
	//		(*nImageDefectCount)++,
	//		szPath,
	//		resultBlob->nDefectColor[nFori],
	//		resultBlob->AI_CODE,
	//		resultBlob->AI_Confidence
	//	);
	//	fprintf_s(out, "\n");
	//}
	fclose(out);
	out = NULL;
}

void ModelResultListenerME::AIDetectImgSave(TaskInfoPtr spTaskInfo, CString sava_path) {

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
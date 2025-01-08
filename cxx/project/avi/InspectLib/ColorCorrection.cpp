
/************************************************************************/
//SVI色彩校正相关源
//修改日期:18.02.20
/************************************************************************/

#include "StdAfx.h"
#include "ColorCorrection.h"

CColorCorrection::CColorCorrection(void)
{
	bLoad = false;
	memset(m_dCoefficient, 0, sizeof(double) * COLOR_CORRECTION_COUNT);
}

CColorCorrection::~CColorCorrection(void)
{
	bLoad = false;
	memset(m_dCoefficient, 0, sizeof(double) * COLOR_CORRECTION_COUNT);
}

//颜色校正值Load
long CColorCorrection::ColorCorrectionLoad(CString strFileName)
{
	//如果已加载
	if (bLoad)	return E_ERROR_CODE_TRUE;

	//文件存在与否需要确认
	CFileFind find;
	BOOL bFindFile = FALSE;

	bFindFile = find.FindFile(strFileName);
	find.Close();

	//如果文件不存在
	if (!bFindFile)
	{
		return E_ERROR_CODE_CCD_EMPTY_FILE;
	}

	//////////////////////////////////////////////////////////////////////////

	char szFileName[256] = { 0, };
	WideCharToMultiByte(CP_ACP, 0, strFileName, -1, szFileName, sizeof(szFileName), NULL, NULL);

	FILE* out = NULL;
	fopen_s(&out, szFileName, "r");

	if (!out)	return E_ERROR_CODE_TRUE;

	double dTemp;

	for (int m = 0; m < COLOR_CORRECTION_COUNT; m++)
	{
		//检索
		fscanf_s(out, "%lf\n", &dTemp);

		m_dCoefficient[m] = dTemp;
	}

	fclose(out);
	out = NULL;

	bLoad = true;

	return E_ERROR_CODE_TRUE;
}

//获取颜色校正值
double* CColorCorrection::GetColorCorrection()
{
	//如果Load不可用
	if (!bLoad)	return NULL;

	return m_dCoefficient;
}
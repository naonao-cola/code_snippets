
/************************************************************************/
//与CCD Defect相关的源
//修改日期:17.07.10
/************************************************************************/

#include "StdAfx.h"
#include "DefectCCD.h"

//设置以上:CCD Defect Delete
//低于设置值:CCD Defect Offset
#define THRESH_GV	20

CDefectCCD::CDefectCCD(void)
{
	//初始化向量
	vector <tCCD_DEFECT>().swap(ptIndexsDelete);
	vector <tCCD_DEFECT>().swap(ptIndexsOffset);

	bLoad = false;
}

CDefectCCD::~CDefectCCD(void)
{
	//初始化向量
	vector <tCCD_DEFECT>().swap(ptIndexsDelete);
	vector <tCCD_DEFECT>().swap(ptIndexsOffset);

	bLoad = false;
}

long CDefectCCD::DefectCCDLoad(CString strFileName, CString strFileName2)
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

	char szFileName[256] = { 0, };
	WideCharToMultiByte(CP_ACP, 0, strFileName, -1, szFileName, sizeof(szFileName), NULL, NULL);

	FILE* out = NULL;
	fopen_s(&out, szFileName, "r");

	if (!out)	return E_ERROR_CODE_TRUE;

	tCCD_DEFECT pt, ptOld;
	for (int m = 0; ; m++)
	{
		//检索
		fscanf_s(out, "%d,%d,%d\n", &pt.x, &pt.y, &pt.gv);

		//如果值相同,请退出
		if (pt.x == ptOld.x &&
			pt.y == ptOld.y)
			break;

		//设置以上:CCD Defect Delete
		//设置以下:CCD Defect Offset
		if (pt.gv >= THRESH_GV)
			ptIndexsDelete.push_back(pt);
		else
			ptIndexsOffset.push_back(pt);

		//复制旧值
		ptOld = pt;
	}

	fclose(out);
	out = NULL;

	/////////////////////////////////////////////////////////////////////////////////////////
		//如果存在LJH DEAD_CCD文件,请一起加载并添加

	bFindFile = FALSE;
	bFindFile = find.FindFile(strFileName2);
	find.Close();

	if (bFindFile)
	{
		char szFileName[256] = { 0, };
		WideCharToMultiByte(CP_ACP, 0, strFileName2, -1, szFileName, sizeof(szFileName), NULL, NULL);

		FILE* out = NULL;
		fopen_s(&out, szFileName, "r");

		if (!out)	return E_ERROR_CODE_TRUE;

		tCCD_DEFECT pt, ptOld;
		for (int m = 0; ; m++)
		{
			//检索
			fscanf_s(out, "%d,%d,%d\n", &pt.x, &pt.y, &pt.gv);

			//如果值相同,请退出
			if (pt.x == ptOld.x &&
				pt.y == ptOld.y)
				break;

			//对于Dead CCD,偏移没有意义...
			ptIndexsDelete.push_back(pt);

			//复制旧值
			ptOld = pt;
		}

		fclose(out);
		out = NULL;
	}
	////////////////////////////////////////////////////////////////////////////////////////

	bLoad = true;

	return E_ERROR_CODE_TRUE;
}

long CDefectCCD::DefectCCDSave(cv::Mat& matSrcBuffer, CString strFileName, CString strFileName2)
{
	//如果没有画面缓冲区
	if (matSrcBuffer.empty())	return E_ERROR_CODE_CCD_EMPTY_BUFFER;

	//不包括PS画面
	if (matSrcBuffer.cols > 10000)		return E_ERROR_CODE_CCD_PS_BUFFER;

	//画面Bright/Dark确认
	cv::Scalar m = cv::mean(matSrcBuffer);

	//初始化向量
	vector <tCCD_DEFECT>().swap(ptIndexsDelete);
	vector <tCCD_DEFECT>().swap(ptIndexsOffset);

	//LJH遮光影像的平均值不超过20...
	if ((double)(m[0]) < 20.)
	{
		tCCD_DEFECT pt;
		char szFileName[256] = { 0, };
		WideCharToMultiByte(CP_ACP, 0, strFileName, -1, szFileName, sizeof(szFileName), NULL, NULL);

		//修改Unicode环境错误"t"->"wt"
		FILE* out = NULL;
		fopen_s(&out, szFileName, "wt");

		//文件错误
		if (!out)	return E_ERROR_CODE_TRUE;

		for (int y = 0; y < matSrcBuffer.rows; y++)
		{
			BYTE* ptr = (BYTE*)matSrcBuffer.ptr(y);

			for (int x = 0; x < matSrcBuffer.cols; x++, ptr++)
			{
				//0 GV时跳过
				if (*ptr == 0)	continue;

				pt.x = x;
				pt.y = y;
				pt.gv = (*ptr);

				//设置以上:CCD Defect Delete
				//设置以下:CCD Defect Offset
				if (pt.gv >= THRESH_GV)
					ptIndexsDelete.push_back(pt);
				else
					ptIndexsOffset.push_back(pt);

				//写入文件
				fprintf_s(out, "%d,%d,%d\n", pt.x, pt.y, pt.gv);
			}
		}
		fclose(out);
		out = NULL;
	}
	else
	{
		tCCD_DEFECT pt;
		char szFileName[256] = { 0, };
		WideCharToMultiByte(CP_ACP, 0, strFileName2, -1, szFileName, sizeof(szFileName), NULL, NULL);

		//修改Unicode环境错误"t"->"wt"
		FILE* out = NULL;
		fopen_s(&out, szFileName, "wt");

		//文件错误
		if (!out)	return E_ERROR_CODE_TRUE;

		for (int y = 1; y < matSrcBuffer.rows - 1; y++)
		{
			BYTE* ptr = (BYTE*)matSrcBuffer.ptr(y);

			for (int x = 1; x < matSrcBuffer.cols - 1; x++, ptr++)
			{
				//如果比点灯区域的平均值暗60%,则注册
				if (*ptr / (double)(m[0]) < 0.6)
				{
					pt.x = x;
					pt.y = y;
					pt.gv = 0;

					//写入文件
					fprintf_s(out, "%d,%d,%d\n", pt.x, pt.y, pt.gv);
					continue;
				}
			}
		}
		fclose(out);
		out = NULL;

		AfxMessageBox(_T("Dead CCD Save OK"));
	}

	bLoad = true;

	return E_ERROR_CODE_TRUE;
}

//获取CCD不良删除数量
int CDefectCCD::GetDefectCCDDeleteCount()
{
	//如果Load不可用
	if (!bLoad)	return 0;

	return (int)ptIndexsDelete.size();
}

//获取CCD不良校正数量
int CDefectCCD::GetDefectCCDOffsetCount()
{
	//如果Load不可用
	if (!bLoad)	return 0;

	return (int)ptIndexsOffset.size();
}

//删除CCD故障
long CDefectCCD::DeleteDefectCCD(cv::Mat& matSrcBuffer, int nSize, int nPS)
{
	if (!bLoad)					return E_ERROR_CODE_CCD_NOT_LOAD;
	if (matSrcBuffer.empty())	return E_ERROR_CODE_CCD_EMPTY_BUFFER;
	if (nSize < 0)					return E_ERROR_CODE_CCD_WARNING_PARA;

#ifdef _DEBUG
#else
#pragma omp parallel for
#endif
	for (int i = 0; i < GetDefectCCDDeleteCount(); i++)
	{
		cv::Rect rect;

		//Size设置
		rect.x = ptIndexsDelete[i].x * nPS - nSize;
		rect.y = ptIndexsDelete[i].y * nPS - nSize;
		rect.width = nSize + nSize + nPS;
		rect.height = nSize + nSize + nPS;

		//涂成黑色
		cv::rectangle(matSrcBuffer, rect, cv::Scalar(0, 0, 0), -1);
	}

	return E_ERROR_CODE_TRUE;
}

//自动删除CCD故障
long CDefectCCD::DeleteAutoDefectCCD(cv::Mat& matSrcBuffer, float fGV, float fBkGV, int nPS, CMatBuf* cMem)
{
	if (matSrcBuffer.empty())	return E_ERROR_CODE_CCD_EMPTY_BUFFER;

	CMatBuf cMatBufTemp;

	//CCD故障数量
	long	nCountCCD = 0;

	//如果处于PS模式
	if (nPS == 2)
	{
		//Temp内存分配
		cMatBufTemp.SetMem(cMem);
		cv::Mat matInBuf = cMatBufTemp.GetMat(matSrcBuffer.size(), CV_32FC1);
		cv::Mat matOutBuf = cMatBufTemp.GetMat(matSrcBuffer.size(), CV_32FC1);
		cv::Mat matKernel1 = cv::Mat::zeros(4, 4, CV_8UC1);
		cv::Mat matKernel2 = cv::Mat::zeros(4, 4, CV_8UC1);

		//创建Kernel
		for (int y = 0; y < 4; y++)
		{
			for (int x = 0; x < 4; x++)
			{
				if (x == 0 ||
					x == 3 ||
					y == 0 ||
					y == 3)
				{
					// matKernel1
					matKernel1.at<uchar>(y, x) = (uchar)1;
				}
				else
				{
					// matKernel2
					matKernel2.at<uchar>(y, x) = (uchar)1;
				}
			}
		}

		//获取外部GV
		cv::filter2D(matSrcBuffer, matOutBuf, CV_32FC1, matKernel1);

		//获取内部GV
		cv::filter2D(matSrcBuffer, matInBuf, CV_32FC1, matKernel2);

		//自动判定为CCD不良
		for (int y = 1; y < matSrcBuffer.rows; y++)
		{
			float* ptrIn = (float*)matInBuf.ptr(y);
			float* ptrOut = (float*)matOutBuf.ptr(y);

			ptrIn++;
			ptrOut++;

			for (int x = 1; x < matSrcBuffer.cols; x++, ptrIn++, ptrOut++)
			{
				if (*ptrIn >= fGV &&	//内部亮度？以上
					*ptrOut <= fBkGV)//外部亮度？以下(仅用于Black...)
				{
					//也存在内部亮度不相似的情况...
					cv::rectangle(matSrcBuffer, cv::Rect(x - 1, y - 1, 2, 2), cv::Scalar(0, 0, 0), -1);

					//CCD故障数量
					nCountCCD++;
				}
			}
		}

		//取消分配
		matInBuf.release();
		matOutBuf.release();
		matKernel1.release();
		matKernel2.release();

		clock_t tBeforeTime = cInspectLibLog.writeInspectLog(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("DeleteAutoDefectCCD End."));

		SetLog(&cInspectLibLog, tBeforeTime, tBeforeTime, NULL);

		if (m_cInspectLibLog->Use_AVI_Memory_Log) {
			writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
			writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
		}
	}
	//如果不是PS模式
	else if (nPS == 1)
	{
		//分配Temp内存
		cMatBufTemp.SetMem(cMem);
		cv::Mat matOutBuf = cMatBufTemp.GetMat(matSrcBuffer.size(), CV_32FC1);
		cv::Mat matKernel = cv::Mat::ones(3, 3, CV_8UC1);
		matKernel.at<uchar>(1, 1) = (uchar)0;

		//获取外部GV
		cv::filter2D(matSrcBuffer, matOutBuf, CV_32FC1, matKernel);

		//自动判定为CCD不良
		for (int y = 0; y < matSrcBuffer.rows; y++)
		{
			uchar* ptrIn = (uchar*)matSrcBuffer.ptr(y);
			float* ptrOut = (float*)matOutBuf.ptr(y);

			for (int x = 0; x < matSrcBuffer.cols; x++, ptrIn++, ptrOut++)
			{
				if (*ptrIn >= fGV && //内部亮度？以上
					*ptrOut <= fBkGV)//外部亮度？以下(仅用于Black...)
				{
					*ptrIn = (uchar)0;

					//CCD故障数量
					nCountCCD++;
				}
			}
		}

		//取消分配
		matOutBuf.release();
		matKernel.release();

		clock_t tBeforeTime = cInspectLibLog.writeInspectLog(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("DeleteAutoDefectCCD End."));

		SetLog(&cInspectLibLog, tBeforeTime, tBeforeTime, NULL);
		if (m_cInspectLibLog->Use_AVI_Memory_Log) {
			writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Fixed USE Memory"), cMatBufTemp.Get_FixMemory());
			writeInspectLog_Memory(E_ALG_TYPE_AVI_MEMORY, __FUNCTION__, _T("Auto USE Memory"), cMatBufTemp.Get_AutoMemory());
		}
	}

	//return E_ERROR_CODE_TRUE;
	return nCountCCD;
}

//CCD不良校正
long CDefectCCD::OffsetDefectCCD(cv::Mat& matSrcBuffer, int nSize, int nPS)
{
	if (!bLoad)				return E_ERROR_CODE_CCD_NOT_LOAD;
	if (matSrcBuffer.empty())	return E_ERROR_CODE_CCD_EMPTY_BUFFER;
	if (nSize < 0)				return E_ERROR_CODE_CCD_WARNING_PARA;

	//图像大小
	long	nWidth = (long)matSrcBuffer.cols;//画面宽度大小
	long	nHeight = (long)matSrcBuffer.rows;//影像垂直大小

#ifdef _DEBUG
#else
#pragma omp parallel for
#endif
	for (int i = 0; i < GetDefectCCDOffsetCount(); i++)
	{
		//根据PS模式的范围
		int nSX = ptIndexsOffset[i].x * nPS - nSize;
		int nSY = ptIndexsOffset[i].y * nPS - nSize;
		int nEX = nSX + nPS + nSize + nSize;
		int nEY = nSY + nPS + nSize + nSize;
		int nOffset = ptIndexsOffset[i].gv;

		//异常处理
		if (nSX < 0)			nSX = 0;
		if (nSY < 0)			nSY = 0;
		if (nEX >= nWidth)	nEX = nWidth - 1;
		if (nEY >= nHeight)	nEY = nHeight - 1;

		for (int y = nSY; y < nEY; y++)
		{
			for (int x = nSX; x < nEX; x++)
			{
				//因为只使用Black Patern...
				int nGV = matSrcBuffer.at<uchar>(y, x) - nOffset;

				//异常处理
				if (nGV < 0)	nGV = 0;

				//重新放入值
				matSrcBuffer.at<uchar>(y, x) = (uchar)nGV;
			}
		}
	}

	return E_ERROR_CODE_TRUE;
}
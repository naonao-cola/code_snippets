
/*****************************************************************************
  File Name		: ImageSave.h
  Version		: ver 1.0
  Create Date	: 2015.03.06
  Description	:图像存储相关函数
  Abbreviations	: 
 *****************************************************************************/

#pragma once

#include "TaskList.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//

void ImageSave(cv::Mat& MatSrcBuffer, TCHAR* szName, ...)
{
	if( MatSrcBuffer.empty() )	return;

		//获取可变参数文件路径,创建路径并保存图像-Todo。优化
	va_list vaList;
	va_start(vaList, szName);
	TCHAR* cBuffer = NULL;

	if (szName != NULL)
	{
		int len = _vscwprintf( szName, vaList ) + 1;

		cBuffer = new TCHAR[sizeof(TCHAR)*len];
		memset(cBuffer,0,sizeof(TCHAR)*len);

		if (cBuffer)
			vswprintf(cBuffer, szName, (va_list)vaList);
	}
	va_end(vaList);

	CString strPath(cBuffer);	
	SHCreateDirectoryEx(NULL, strPath.Left(strPath.GetLength() - (strPath.GetLength() - strPath.ReverseFind(_T('\\')))), NULL);

	SAFE_DELETE(cBuffer);

	char* pTemp = CSTR2PCH(strPath);
		//保存画面
	try{
//修改与TIFF格式存储速度相关的OpenCV源

//			//未压缩

		{
			cv::imwrite(pTemp, MatSrcBuffer);
		}
	}
	catch(cv::Exception& ex){
		theApp.WriteLog(eLOGPROC, eLOGLEVEL_DETAIL, TRUE, TRUE, _T("Exception ImageSave() : %s"), ex.what());
	}

	SAFE_DELETE_ARR(pTemp);	
}

//	const int   MAX_COUNT_OF_ASYNC_IMAGE_SAVE = 0;
//	const int	MAX_JPG_IMAGE_SAVE_SIZE_Y	= 20000;

//
//	//限制最大并发执行数量。防止过度滥用

//	const char	*strPath_		= strPath;

//

//

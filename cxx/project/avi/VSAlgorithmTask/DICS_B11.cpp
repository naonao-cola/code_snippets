#include "stdafx.h"
#include "DICS_B11.h"

#include <algorithm>

#include "VSAlgorithmTask.h"
#include "Define.h"

DICS_B11::DICS_B11(void)
{
}
DICS_B11::~DICS_B11(void)
{
}

CString strCam[2] = {
	_T("Coaxial"),
	_T("Side")
};

void DICS_B11::DICSStart(cv::Mat& DicsImg, cv::Mat& matOrigin, cv::Point* ptCorner, int InspType, CString INIPath, CString ID, CString Patturn, int nCameraNum, double* dAlgPara, CString strDrive)
{
	thread StartThread(StartSave, std::ref(DicsImg), matOrigin, ptCorner, InspType, INIPath, ID, Patturn, nCameraNum, dAlgPara, strDrive);
	StartThread.detach();
}

void DICS_B11::SaveStart(cv::Mat& matROI, cv::Mat& matDics, CString INIPath, CString ID, CString Patturn, int InspType, int nCameaNum, CString strDrive)
{
	CString strDirectoryPath = CheckDirectory(INIPath, ID, strDrive);
	thread StartThread(SaveImage, matROI, matDics, strDirectoryPath, Patturn, InspType, nCameaNum);
	StartThread.detach();
}

void DICS_B11::StartSave(cv::Mat& DicsImg, cv::Mat matOrigin, cv::Point* ptCorner, int InspType, CString INIPath, CString ID, CString Patturn, int nCameraNum, double* dAlgPara, CString strDrive)
{
	try
	{
		switch (InspType)
		{
		case INSP_AVI:
			if (dAlgPara[E_PARA_AVI_DICS_TEXT] < 1)
				return;
			break;
		case INSP_SVI:
			if (dAlgPara[E_PARA_SVI_DICS_TEXT] < 1)
				return;
			break;
		default:
			return;
		}

				//	else if(dAlgPara[E_PARA_SVI_DICS_TEXT]>1)	//模拟模式存储路径不同
		// 		Path.Format(_T("D:\\DICS"));

		CRect rectROI = CRect(
			min(ptCorner[E_CORNER_LEFT_TOP].x, ptCorner[E_CORNER_LEFT_BOTTOM].x),
			min(ptCorner[E_CORNER_LEFT_TOP].y, ptCorner[E_CORNER_RIGHT_TOP].y),
			max(ptCorner[E_CORNER_RIGHT_TOP].x, ptCorner[E_CORNER_RIGHT_BOTTOM].x),
			max(ptCorner[E_CORNER_LEFT_BOTTOM].y, ptCorner[E_CORNER_RIGHT_BOTTOM].y));

		Rect ROI;
		ROI.x = rectROI.left;
		ROI.y = rectROI.top;
		ROI.width = rectROI.right - rectROI.left;
		ROI.height = rectROI.bottom - rectROI.top;

				//确认图像是否正常
				//如果画面缓冲区为NULL,则跳过
		if (!matOrigin.empty())
		{
			//theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, FALSE, _T("--------DICS_%s_%s_Directory Check--------"), ID, Patturn);
			CString strDirectoryPath = CheckDirectory(INIPath, ID, strDrive);

						//根据扫描(AVI,SVI)进行图像预处理
			cv::Mat matROI, matPreprocessing;

			matOrigin(ROI).copyTo(matROI);

			//theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, FALSE, _T("--------DICS_%s_%s_Image Processing--------"), ID, Patturn);
			switch (InspType)
			{
			case INSP_AVI:
				if (!CheckPattern(Patturn, INIPath))
				{
					//theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, FALSE, _T("--------DICS_%s_%s_END--------"), ID, Patturn);
					return;
				}
				if (Patturn == _T("DUST"))
					Preprocessing_AVI_DUST(matROI, matPreprocessing, dAlgPara);
				else
					Preprocessing_AVI(matROI, matPreprocessing, Patturn, dAlgPara);
				break;
			case INSP_SVI:
				Preprocessing_SVI(matROI, matPreprocessing, Patturn, dAlgPara);
				break;
			default:
				break;
			}

			//theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, FALSE, _T("--------DICS_%s_%s_Image Save--------"), ID, Patturn);
						//保存图像

			matPreprocessing.copyTo(DicsImg);
			SaveImage(matROI, matPreprocessing, strDirectoryPath, Patturn, InspType, nCameraNum);

			matROI.release();
			//matPreprocessing.release();
			//theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, FALSE, _T("--------DICS_%s_%s_END--------"), ID, Patturn);
		}

	}
	catch (CException* e)
	{
		//theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, FALSE, _T("--------DICS_%s_%s_error--------"), ID, Patturn);
	}
}

//DICS字母2023/08/03
void DICS_B11::Generate(cv::Mat matOrigin, cv::Mat& matDics, cv::Mat& matROI, cv::Point* ptCorner, int InspType, CString INIPath, CString ID, CString Patturn, int nCameraNum, double* dAlgPara, CString strDrive)
{
	try
	{
		switch (InspType)
		{
		case INSP_AVI:
			if (dAlgPara[E_PARA_AVI_DICS_TEXT] < 1)
				break;
			break;
		case INSP_SVI:
			if (dAlgPara[E_PARA_SVI_DICS_TEXT] < 1)
				break;
			break;
		default:
			break;
		}

		CRect rectROI = CRect(
			min(ptCorner[E_CORNER_LEFT_TOP].x, ptCorner[E_CORNER_LEFT_BOTTOM].x),
			min(ptCorner[E_CORNER_LEFT_TOP].y, ptCorner[E_CORNER_RIGHT_TOP].y),
			max(ptCorner[E_CORNER_RIGHT_TOP].x, ptCorner[E_CORNER_RIGHT_BOTTOM].x),
			max(ptCorner[E_CORNER_LEFT_BOTTOM].y, ptCorner[E_CORNER_RIGHT_BOTTOM].y));

		Rect ROI;
		ROI.x = rectROI.left;
		ROI.y = rectROI.top;
		ROI.width = rectROI.right - rectROI.left;
		ROI.height = rectROI.bottom - rectROI.top;

				//确认图像是否正常
				//如果画面缓冲区为NULL,则跳过
		if (!matOrigin.empty())
		{
			matOrigin(ROI).copyTo(matROI);

			//theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, FALSE, _T("-----AI---DICS_%s_%s_Image Processing--------"), ID, Patturn);
			switch (InspType)
			{
			case INSP_AVI:
				if (!CheckPattern(Patturn, INIPath))
				{
					//theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, FALSE, _T("----AI----DICS_%s_%s_END--------"), ID, Patturn);
					break;
				}
				if (Patturn == _T("DUST"))
					Preprocessing_AVI_DUST(matROI, matDics, dAlgPara);
				else
					Preprocessing_AVI(matROI, matDics, Patturn, dAlgPara);
				break;
			case INSP_SVI:
				Preprocessing_SVI(matROI, matDics, Patturn, dAlgPara);
				break;
			default:
				break;
			}

			//theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, FALSE, _T("----AI----DICS_%s_%s_END--------"), ID, Patturn);
		}
	}
	catch (CException* e)
	{
		theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, FALSE, _T("-----AI---DICS_%s_%s_error--------"), ID, Patturn);

	}
}

//随着2022.05.25概念的改变,添加到测试用途
//该函数将按日期创建存储文件夹,并删除过期的文件夹
//虽然在算法测试中进行该逻辑看起来不合适,但基于2022.05.25,客户希望进行功能测试,因此添加了该功能
CString DICS_B11::CheckDirectory(CString INIPath, CString ID, CString strDrive)
{
		//从INI读取保存的Drive
	TCHAR tcharDriver[100] = { 0, };
	CString strDriver;
	GetPrivateProfileString(_T("DICS"), _T("Drive"), _T("D:"), tcharDriver, _MAX_PATH, INIPath);
	strDriver = tcharDriver;
	//strDriver = strDrive;
		//DICD文件夹
	CString strDICS;
	strDICS.Format(_T("%s\\DICS"), strDriver);

		//检查文件夹如果没有该文件夹,则创建该文件夹
	DWORD result;
	if (((result = GetFileAttributes(strDICS)) == -1) || !(result&FILE_ATTRIBUTE_DIRECTORY)) {
		CreateDirectory(strDICS, NULL);
	}

		//如果不确定是否存在当前日期的文件夹,则创建
	time_t timer = time(NULL);
	struct tm* t = localtime(&timer);

	CString strDate;
	strDate.Format(_T("%s\\%d%02d%02d"), strDICS, t->tm_year + 1900, t->tm_mon + 1, t->tm_mday);

	if (((result = GetFileAttributes(strDate)) == -1) || !(result&FILE_ATTRIBUTE_DIRECTORY)) {
		CreateDirectory(strDate, NULL);
	}

		//Cell ID文件夹
	CString strCellID;
	strCellID.Format(strDate + _T("\\") + ID);

	if (((result = GetFileAttributes(strCellID)) == -1) || !(result&FILE_ATTRIBUTE_DIRECTORY)) {
		CreateDirectory(strCellID, NULL);
	}

	// 
		//	//删除一周前的文件夹？
	// 	timer = time(NULL) - (7 * 24 * 60 * 60);
	// 	t = localtime(&timer);
	// 
	// 	CString strDate2;
	// 	strDate2.Format(_T("%s\\%d%02d%02d"), Path, t->tm_year + 1900, t->tm_mon + 1, t->tm_mday);
	// 
		//	//如果没有则跳过
	// 	if (((result = GetFileAttributes(strDate2)) == -1) || !(result&FILE_ATTRIBUTE_DIRECTORY)) {
	// 		return strDate;
	// 	}
	// 
	 	//DeleteDirectoryFile(strDate2);

	return strCellID;
}

void DICS_B11::DeleteDirectoryFile(CString Path)
{
		//删除文件夹中的所有文件
	CFileFind finder;

	BOOL bWorking = finder.FindFile((CString)Path + "/*.*");

	while (bWorking)
	{
		bWorking = finder.FindNextFile();
		if (finder.IsDots() == FALSE)
		{
			if (finder.IsDirectory())
			{
				DeleteDirectoryFile(finder.GetFilePath());
			}
			else
			{
				DeleteFile(finder.GetFilePath());
			}
		}

	}
	finder.Close();

		//删除文件夹
	RemoveDirectory(Path);
}

void DICS_B11::Preprocessing_AVI(cv::Mat& matROI, cv::Mat& matPreprocessing, CString Patturn, double* dAlgPara)
{
	try
	{
				//获取参数
		// 2022.06.24 
		int nImageSizeRatio = (int)dAlgPara[E_PARA_AVI_DICS_IMAGE_RATIO];//

		int nGaussianSize = (int)dAlgPara[E_PARA_AVI_DICS_GAUSSIAN_SIZE];
		double nGaussianSigma = dAlgPara[E_PARA_AVI_DICS_GAUSSIAN_SIGMA];

		int nCount = (int)dAlgPara[E_PARA_AVI_DICS_CONTRAST_COUNT];
		int nMin_Offset = (int)dAlgPara[E_PARA_AVI_DICS_CONTRAST_MIN_OFFSET];
		int nMax_Offset = (int)dAlgPara[E_PARA_AVI_DICS_CONTRAST_MAX_OFFSET];

		//if(Patturn==_T("G128"))	//对于G3模式,请单独使用参数客户要求2023.02.07HMG(对于G3模式,名称为G128)
		//{
		//	nGaussianSize = (int)dAlgPara[E_PARA_AVI_DICS_GAUSSIAN_SIZE_G3];
		//	nGaussianSigma = dAlgPara[E_PARA_AVI_DICS_GAUSSIAN_SIGMA_G3];

		//	nCount = (int)dAlgPara[E_PARA_AVI_DICS_CONTRAST_COUNT_G3];
		//	nMin_Offset = (int)dAlgPara[E_PARA_AVI_DICS_CONTRAST_MIN_OFFSET_G3];
		//	nMax_Offset = (int)dAlgPara[E_PARA_AVI_DICS_CONTRAST_MAX_OFFSET_G3];
		//}

		if (nCount <= 0)
			nCount = 150;

		if (nMin_Offset < 0)
			nMin_Offset = 10;

		if (nMax_Offset < 0)
			nMax_Offset = 10;

		/////////////////////////////////////

		if (nGaussianSize % 2 == 0)
		{
			nGaussianSize += 1;
		}
		//2023.03.27_xf

		//theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, FALSE, _T("--------DICS_%s_Gaussian Blur--------"), Patturn);
		cv::Mat matGaussian;
		cv::GaussianBlur(matROI, matGaussian, cv::Size(nGaussianSize, nGaussianSize), nGaussianSigma);

				//减小图像大小
		//theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, FALSE, _T("--------DICS_%s_ReSize--------"), Patturn);
		int nWidth = matGaussian.cols * nImageSizeRatio / 100;//
		int nHeight = matGaussian.rows * nImageSizeRatio / 100;//

		cv::resize(matGaussian, matGaussian, cv::Size(nWidth, nHeight));//

				//使用高斯模糊平滑图像
				//为了更方便地承认村上的不良行为,需要平滑图像

		//theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, FALSE, _T("--------DICS_%s_Contrast--------"), Patturn);
				//使用Contrast(图像平滑)进行亮度对比度放大
				//不良认同感增加
		Contrast(matGaussian, matPreprocessing, nMin_Offset, nMax_Offset, nCount);

		matGaussian.release();
	}
	catch (CMemoryException* e)
	{
		throw e;
	}
	catch (CFileException* e)
	{
		throw e;
	}
	catch (CException* e)
	{
		throw e;
	}
}

void DICS_B11::Preprocessing_AVI_DUST_PAD(cv::Mat& matROI, double* dAlgPara, CString Patturn, CString strDirectoryPath)
{
	if (Patturn != _T("DUST")) {
		return;
	}
	try
	{
				//获取参数
		// 2022.06.24 
		cv::Mat matPreprocessing, padROI;
		cv::Rect padRoi = cv::Rect(11588, 2000, 2520, 7020);
		int nImageSizeRatio = (int)dAlgPara[E_PARA_AVI_DICS_IMAGE_RATIO];	//

		int nShiftRanage = (int)dAlgPara[E_PARA_AVI_DICS_SHIFTRANGE_DUST];

		int nGaussianSize = (int)dAlgPara[E_PARA_AVI_DICS_GAUSSIAN_SIZE_DUST];
		double nGaussianSigma = dAlgPara[E_PARA_AVI_DICS_GAUSSIAN_SIGMA_DUST];

		int nMaxSize = (int)dAlgPara[E_PARA_AVI_DICS_MAX_FILTER_SIZE_DUST];

		if (nMaxSize < 3)	nMaxSize = 3;

		/////////////////////////////////////

				//减小图像大小
				//对于AVI,为了提高运算速度和容量,减小图像大小
		//int nWidth = matROI.cols;
		//int nHeight = matROI.rows;

		bool bFlag = true;
		//cv::Mat t_img = cv::imread("D:\\IMTC_DATA\\AVI\\B61C290006C1BAL01-DO\\00_DUST_CAM00.bmp");
		//cv::Rect t_roi = cv::Rect(15108,2088,1152,6348);
		//t_img(t_roi).copyTo(matROI);

				//与Dust检查前处理相同
		CString Patturn = _T("PAD");
		//theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, FALSE, _T("--------DICS_%s_Shift Copy--------"), Patturn);
		matROI(padRoi).copyTo(padROI);
		// 1) Shift Copy
		cv::Mat matTemp = cv::Mat::zeros(padROI.size(), padROI.type());
		ShiftCopy(padROI, matTemp, nShiftRanage, 0, 1, 0);

		//theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, FALSE, _T("--------DICS_%s_Gaussian Blur--------"), Patturn);
		// 2) GaussianVlur
		cv::Mat matGaussian;
		cv::GaussianBlur(matTemp, matGaussian, cv::Size(nGaussianSize, nGaussianSize), nGaussianSigma);

		//theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, FALSE, _T("--------DICS_%s_MinMax--------"), Patturn);
		// 3) Max
		cv::Mat	StructElem = cv::getStructuringElement(MORPH_RECT, cv::Size(nMaxSize, nMaxSize), cv::Point(-1, -1));
		cv::morphologyEx(matGaussian, matGaussian, MORPH_DILATE, StructElem);	// MORPH_DILATE (Max)

		//theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, FALSE, _T("--------DICS_%s_Resize--------"), Patturn);
		//int nWidth = matGaussian.cols * nImageSizeRatio / 50;//
		//int nHeight = matGaussian.rows * nImageSizeRatio / 50;//
		cv::resize(matGaussian, matPreprocessing, cv::Size(806, 2022));//
		CString strTemp;
		strTemp.Format(_T("%s\\%s_Contrast.jpg"), strDirectoryPath, Patturn);
		cv::imwrite((cv::String)(CStringA)strTemp, matPreprocessing);
		//cv::imwrite("D:\\IMTC_DATA\\AVI\\B61C290006C1BAL01-DO\\pad_dics.jpg", matPreprocessing);
		matTemp.release();
		matGaussian.release();
		StructElem.release();
		matPreprocessing.release();
	}
	catch (CMemoryException* e)
	{
		throw e;
	}
	catch (CFileException* e)
	{
		throw e;
	}
	catch (CException* e)
	{
		throw e;
	}
}

void DICS_B11::Preprocessing_AVI_DUST(cv::Mat& matROI, cv::Mat& matPreprocessing, double* dAlgPara)
{
	try
	{
				//获取参数
		// 2022.06.24 
		int nImageSizeRatio = (int)dAlgPara[E_PARA_AVI_DICS_IMAGE_RATIO];	//

		int nShiftRanage = (int)dAlgPara[E_PARA_AVI_DICS_SHIFTRANGE_DUST];

		int nGaussianSize = (int)dAlgPara[E_PARA_AVI_DICS_GAUSSIAN_SIZE_DUST];
		double nGaussianSigma = dAlgPara[E_PARA_AVI_DICS_GAUSSIAN_SIGMA_DUST];

		int nMaxSize = (int)dAlgPara[E_PARA_AVI_DICS_MAX_FILTER_SIZE_DUST];

		if (nMaxSize < 3)	nMaxSize = 3;

		/////////////////////////////////////

				//减小图像大小
				//对于AVI,为了提高运算速度和容量,减小图像大小
		//int nWidth = matROI.cols;
		//int nHeight = matROI.rows;

		bool bFlag = true;
		//cv::Mat t_img = cv::imread("D:\\IMTC_DATA\\AVI\\B61C290006C1BAL01-DO\\00_DUST_CAM00.bmp");
		//cv::Rect t_roi = cv::Rect(15108,2088,1152,6348);
		//t_img(t_roi).copyTo(matROI);

				//与Dust检查前处理相同
		CString Patturn = _T("DUST");
		//theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, FALSE, _T("--------DICS_%s_Shift Copy--------"), Patturn);

		// 1) Shift Copy
		cv::Mat matTemp = cv::Mat::zeros(matROI.size(), matROI.type());
		ShiftCopy(matROI, matTemp, nShiftRanage, 0, 1, 0);

		//theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, FALSE, _T("--------DICS_%s_Gaussian Blur--------"), Patturn);
		// 2) GaussianVlur
		cv::Mat matGaussian;
		cv::GaussianBlur(matTemp, matGaussian, cv::Size(nGaussianSize, nGaussianSize), nGaussianSigma);

		//theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, FALSE, _T("--------DICS_%s_MinMax--------"), Patturn);
		// 3) Max
		cv::Mat	StructElem = cv::getStructuringElement(MORPH_RECT, cv::Size(nMaxSize, nMaxSize), cv::Point(-1, -1));
		cv::morphologyEx(matGaussian, matGaussian, MORPH_DILATE, StructElem);	// MORPH_DILATE (Max)

		//theApp.WriteLog(eLOGPROC, eLOGLEVEL_BASIC, TRUE, FALSE, _T("--------DICS_%s_Resize--------"), Patturn);
		int nWidth = matGaussian.cols * nImageSizeRatio / 100;//
		int nHeight = matGaussian.rows * nImageSizeRatio / 100;//
		cv::resize(matGaussian, matPreprocessing, cv::Size(nWidth, nHeight));//
		//cv::imwrite("D:\\IMTC_DATA\\AVI\\B61C290006C1BAL01-DO\\pad_dics.jpg", matPreprocessing);
		matTemp.release();
		matGaussian.release();
		StructElem.release();
	}
	catch (CMemoryException* e)
	{
		throw e;
	}
	catch (CFileException* e)
	{
		throw e;
	}
	catch (CException* e)
	{
		throw e;
	}
}

void DICS_B11::Preprocessing_SVI(cv::Mat& matROI, cv::Mat& matPreprocessing, CString Patturn, double* dAlgPara)
{
		//获取参数
	int nCount = dAlgPara[E_PARA_SVI_CONTRAST_COUNT];
	int nMin_Offset = dAlgPara[E_PARA_SVI_CONTRAST_MIN_OFFSET];
	int nMax_Offset = dAlgPara[E_PARA_SVI_CONTRAST_MAX_OFFSET];

	if (nCount <= 0)
		nCount = 150;

	if (nMin_Offset == NULL)
		nMin_Offset = 10;

	if (nMax_Offset == NULL)
		nMax_Offset = 10;

		//SVI即使不单独减小图像大小,也不会对容量造成影响

		//使用高斯模糊平滑图像
		//为了更方便地承认CM不良,需要平滑图像吗？

		//SVI必须将图像分割为三个通道(R,G,B仅进行该通道)

	cv::Mat matChannel[3];
	cv::split(matROI, matChannel);
	if (Patturn == _T("R") || Patturn == _T("G") || Patturn == _T("B"))
	{

		if (Patturn == _T("B"))
		{
			Contrast(matChannel[0], matChannel[0], nMin_Offset, nMax_Offset, nCount);
		}
		else if (Patturn == _T("G"))
		{
			Contrast(matChannel[1], matChannel[1], nMin_Offset, nMax_Offset, nCount);
		}
		else if (Patturn == _T("R"))
		{
			Contrast(matChannel[2], matChannel[2], nMin_Offset, nMax_Offset, nCount);
		}

		merge(matChannel, 3, matPreprocessing);
	}
	else
	{
		// 		Contrast(matChannel[0], matChannel[0], nMin_Offset, nMax_Offset, nCount);
		// 		Contrast(matChannel[1], matChannel[1], nMin_Offset, nMax_Offset, nCount);
		// 		Contrast(matChannel[2], matChannel[2], nMin_Offset, nMax_Offset, nCount);
		// 		merge(matChannel, 3, matPreprocessing);
		ContrastColor(matROI, matPreprocessing, nMin_Offset, nMax_Offset, nCount);
		//Contrast(matROI, matPreprocessing, nMin_Offset, nMax_Offset, nCount);
	}

}

void DICS_B11::SaveImage(cv::Mat& matROI, cv::Mat& matPreprocessing, CString  strDirectoryPath, CString Patturn, int InspType, int nCameaNum)
{

	CString strTemp;
	// 		strTemp.Format(_T("%s\\%s_Origin.jpg"), strDirectoryPath, Patturn);
	//  		cv::imwrite((cv::String)(CStringA)strTemp, matROI);
	// 
	// 		strTemp.Format(_T("%s\\%s_Contrast.jpg"), strDirectoryPath, Patturn);
	// 		cv::imwrite((cv::String)(CStringA)strTemp, matPreprocessing);

	switch (InspType)
	{
	case INSP_AVI:
		// 			strTemp.Format(_T("%s\\%s_Origin.jpg"), strDirectoryPath, Patturn);
		// 			cv::imwrite((cv::String)(CStringA)strTemp, matROI);

		strTemp.Format(_T("%s\\%s_Contrast.jpg"), strDirectoryPath, Patturn);
		cv::imwrite((cv::String)(CStringA)strTemp, matPreprocessing);
		break;
	case INSP_SVI:
		// 			strTemp.Format(_T("%s\\%s_%s_Origin.jpg"), strDirectoryPath, strCam[nCameaNum], Patturn);
		// 			cv::imwrite((cv::String)(CStringA)strTemp, matROI);

		strTemp.Format(_T("%s\\%s_%s_Contrast.jpg"), strDirectoryPath, strCam[nCameaNum], Patturn);
		cv::imwrite((cv::String)(CStringA)strTemp, matPreprocessing);
		break;
	default:
		break;

	}

}

void DICS_B11::Contrast(cv::Mat& matSrc, cv::Mat& matDst, int nMin, int nMax, int nCount)
{
		//创建直方图
	float LUT[256] = { 0, };

	cv::MatIterator_<uchar> itSrc, endSrc;
	itSrc = matSrc.begin<uchar>();
	endSrc = matSrc.end<uchar>();

	for (; itSrc != endSrc; itSrc++)
	{
		LUT[((uchar)*itSrc)]++;
	}

		//平均值是？
	int nMean = cv::mean(matSrc)[0];

	// Auto
	// Min
	int nAutoMin = 255;

	for (int nGV = nMean; nGV > -1; nGV--)
	{
		if (LUT[(uchar)nGV] > nCount && nGV < nAutoMin)
			nAutoMin = nGV;

		if (LUT[(uchar)nGV] < nCount)
			break;
	}
	nAutoMin -= nMin;
	if (nAutoMin < 0)
		nAutoMin = 0;

	// Mat
	int nAutoMax = 0;

	for (int nGV = nMean; nGV < 256; nGV++)
	{
		if (LUT[(uchar)nGV] > nCount && nGV > nAutoMax)
			nAutoMax = nGV;

		if (LUT[(uchar)nGV] < nCount)
			break;
	}
	nAutoMax += nMax;
	if (nAutoMax > 255)
		nAutoMax = 255;

	double dVal = 255.0 / (nAutoMax - nAutoMin);
	cv::subtract(matSrc, nAutoMin, matDst);
	cv::multiply(matDst, dVal, matDst);

}

void DICS_B11::ContrastColor(cv::Mat& matSrc, cv::Mat& matDst, int nMin, int nMax, int nCount)
{
		//Gray转换
	cv::Mat matGray;
	cv::cvtColor(matSrc, matGray, CV_BGR2GRAY);

		//创建直方图
	float LUT[256] = { 0, };

	cv::MatIterator_<uchar> itSrc, endSrc;
	itSrc = matGray.begin<uchar>();
	endSrc = matGray.end<uchar>();

	for (; itSrc != endSrc; itSrc++)
	{
		LUT[((uchar)*itSrc)]++;
	}

		//平均值是？
	int nMean = cv::mean(matGray)[0];

	// Auto
	// Min
	int nAutoMin = 255;

	for (int nGV = nMean; nGV > -1; nGV--)
	{
		if (LUT[(uchar)nGV] > nCount && nGV < nAutoMin)
			nAutoMin = nGV;

		if (LUT[(uchar)nGV] < nCount)
			break;
	}
	nAutoMin -= nMin;
	if (nAutoMin < 0)
		nAutoMin = 0;

	// Mat
	int nAutoMax = 0;

	for (int nGV = nMean; nGV < 256; nGV++)
	{
		if (LUT[(uchar)nGV] > nCount && nGV > nAutoMax)
			nAutoMax = nGV;

		if (LUT[(uchar)nGV] < nCount)
			break;
	}
	nAutoMax += nMax;
	if (nAutoMax > 255)
		nAutoMax = 255;

	double dVal = 255.0 / (nAutoMax - nAutoMin);
	cv::subtract(matSrc, nAutoMin, matDst);
	cv::multiply(matDst, dVal, matDst);

}
// 2022.07.04
bool DICS_B11::CheckPattern(CString Patturn, CString INIPath)
{
	TCHAR tcharPattern[100] = { 0, };
	CString strPattern, str;
	GetPrivateProfileString(_T("DICS"), _T("Unused Pattern"), _T(""), tcharPattern, _MAX_PATH, INIPath);
	strPattern = tcharPattern;

	int nCount = 0;

	while (1)
	{
		if (AfxExtractSubString(str, strPattern, nCount++, ','))
		{
			if (Patturn == str)
				return false;
		}
		else
		{
			return true;
		}
	}
}

// 2022.07.29
void DICS_B11::ShiftCopy(cv::Mat& matSrcImage, cv::Mat& matDstImage, int nShiftX, int nShiftY, int nShiftLoopX, int nShiftLoopY)
{
		//如果没有缓冲区。
	if (matSrcImage.empty())		return;

		//异常处理
	if (nShiftX < 0)		nShiftX = 0;
	if (nShiftY < 0)		nShiftY = 0;

	nShiftLoopX++;
	nShiftLoopY++;

	//////////////////////////////////////////////////////////////////////////

		//源&结果
	cv::Mat matSrcBuf, matDstBuf;

		//缓冲区分配和初始化
	matSrcImage.copyTo(matSrcBuf);
	matSrcImage.copyTo(matDstBuf);

		//画面大小
	int nImageSizeX = matSrcBuf.cols;
	int nImageSizeY = matSrcBuf.rows;

		//临时缓冲区
	cv::Mat matTempBuf1, matTempBuf2;

		//x方向
	int nOffsetX = 0;
	for (int x = 1; x < nShiftLoopX; x++)
	{
		nOffsetX = x * nShiftX;

		matTempBuf1 = matDstBuf(cv::Rect(0, 0, nImageSizeX - nOffsetX, nImageSizeY));
		matTempBuf2 = matSrcBuf(cv::Rect(nOffsetX, 0, nImageSizeX - nOffsetX, nImageSizeY));

				//积分不良时,可能会打开不应该点亮的部分的数组
				//如果覆盖的话,不良现象就会消失,所以无法使用
		cv::add(matTempBuf1, matTempBuf2, matTempBuf1);
		//cv::max(matTempBuf1, matTempBuf2, matTempBuf1);
	}

		//y方向
	int nOffsetY = 0;
	matDstBuf.copyTo(matSrcBuf);
	for (int y = 1; y < nShiftLoopY; y++)
	{
		nOffsetY = y * nShiftY;

		matTempBuf1 = matDstBuf(cv::Rect(0, 0, nImageSizeX, nImageSizeY - nOffsetY));
		matTempBuf2 = matSrcBuf(cv::Rect(0, nOffsetY, nImageSizeX, nImageSizeY - nOffsetY));

				//积分不良时,可能会打开不应该点亮的部分的数组
				//如果覆盖的话,不良现象就会消失,所以无法使用
		cv::add(matTempBuf1, matTempBuf2, matTempBuf1);
		//cv::max(matTempBuf1, matTempBuf2, matTempBuf1);
	}

		//除以相加的数目
	//matDstBuf /= (nShiftLoopX * nShiftLoopY);

		//因为只复制了左,上方向,所以移动到中央
	nOffsetX /= 2;
	nOffsetY /= 2;
	//matDstImage = cv::Mat::zeros(matDstBuf.size(), matDstBuf.type());
	matDstBuf(cv::Rect(0, 0, matDstBuf.cols - nOffsetX, matDstBuf.rows - nOffsetY)).copyTo(matDstImage(cv::Rect(nOffsetX, nOffsetY, matDstBuf.cols - nOffsetX, matDstBuf.rows - nOffsetY)));

		//缓冲区关闭
	matTempBuf1.release();
	matTempBuf2.release();
	matSrcBuf.release();
	matDstBuf.release();

}
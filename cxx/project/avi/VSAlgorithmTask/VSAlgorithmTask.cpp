
//VSAlgorithmTask.cpp:定义应用程序的类行为。
//

#include "stdafx.h"
#include "VSAlgorithmTask.h"
#include <conio.h>
#include "Markup.h"
#include <omp.h>
#include <codecvt>
#include <locale>
#include "DllInterface.h"	

#include "../InspectLib/AIInspectLib/AIRuntime/AIRuntimeDataStruct.h"
#include "../InspectLib/AIInspectLib/AIRuntime/AIRuntimeInterface.h"
#include "../InspectLib/AIInspectLib/AIRuntime/AIRuntimeUtils.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

// CVSAlgorithmTaskApp

BEGIN_MESSAGE_MAP(CVSAlgorithmTaskApp, CWinApp)
	ON_COMMAND(ID_HELP, &CWinApp::OnHelp)
END_MESSAGE_MAP()

//创建CVS AlgorithmTaskApp

CVSAlgorithmTaskApp::CVSAlgorithmTaskApp()
	:m_bIPCConnect(FALSE), m_bExecDisk(FALSE), m_bExecAlgoThrBusy(FALSE)
{
	//重新启动管理器支持
	m_dwRestartManagerSupportFlags = AFX_RESTART_MANAGER_SUPPORT_RESTART;

	//TODO:在此添加生成代码。
	//在InitInstance中放置所有重要的初始化任务。
	InitializeCriticalSectionAndSpinCount(&m_csCntFileSafe, 4000);
	InitializeCriticalSectionAndSpinCount(&m_csJudgeRepeatCount, 4000);
	InitializeCriticalSectionAndSpinCount(&theApp.m_SVICoordFile, 4000);
}

CVSAlgorithmTaskApp::~CVSAlgorithmTaskApp()
{
	DeleteCriticalSection(&m_csCntFileSafe);
	DeleteCriticalSection(&m_csJudgeRepeatCount);
	ExitVision();
	DeleteCriticalSection(&theApp.m_SVICoordFile);
}

//唯一的CVS AlgorithmTaskApp对象。

CVSAlgorithmTaskApp theApp;
//生成KYH2022.03.11 dump
static LONG CALLBACK TopLevelExceptionFilterCallBack(EXCEPTION_POINTERS* exceptionInfo);
LONG CALLBACK TopLevelExceptionFilterCallBack(EXCEPTION_POINTERS* exceptionInfo)
{
	MINIDUMP_EXCEPTION_INFORMATION dmpInfo = { 0 };
	dmpInfo.ThreadId = ::GetCurrentThreadId();
	dmpInfo.ExceptionPointers = exceptionInfo;
	dmpInfo.ClientPointers = FALSE;
	CTime CurrentTime;
	CurrentTime = CTime::GetCurrentTime();

	CString Path = _T("E:\\IMTC\\Dump");//您想要保存的路径
	CString Name;//您也可以更改想要保存的名称格式。
	Name.Format(_T("%s\\DumpFile_%02d%02d%02d_%02d%02d%02d.dmp"), Path, CurrentTime.GetYear(), CurrentTime.GetMonth(), CurrentTime.GetDay(), CurrentTime.GetHour(), CurrentTime.GetMinute(), CurrentTime.GetSecond());

	CreateDirectory(Path, NULL);

	HANDLE hFile = CreateFile(Name, GENERIC_WRITE, FILE_SHARE_WRITE, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL); // 创建转储
	BOOL bWrite = ::MiniDumpWriteDump(::GetCurrentProcess(), ::GetCurrentProcessId(), hFile, MiniDumpNormal, &dmpInfo, NULL, NULL);
	return 0L;
}

//初始化CVS AlgorithmTaskApp

BOOL CVSAlgorithmTaskApp::InitInstance()
{
	int iCnt = 0;
	LPWSTR* pStr = NULL;
	CString strParam;
	pStr = CommandLineToArgvW(GetCommandLine(), &iCnt);

	if (pStr == NULL || iCnt < 2)
	{
		AfxMessageBox(_T("Please Check Command Line Argument !!"));
		return 1;
	}
	strParam = pStr[1];
	theApp.m_Config.SetINIPath(strParam);

	// 初始化算法库
	Init_InspectLib(VS_ALGORITHM_TASK_INI_FILE);

	//防止程序重复运行-S
	//删除重复运行防止—B11PC1:Stage2响应操作
//HANDLE hMutex = CreateMutex(NULL, TRUE, _T("VsAlgorithmTask"));
//if(GetLastError() == ERROR_ALREADY_EXISTS)
//{
//	ReleaseMutex(hMutex);

//	CWnd *pWndPrev, *pWndChild;
//	pWndPrev = CWnd::FindWindow(NULL, _T("VsAlgorithmTask"));
//	if(pWndPrev)
//	{
//		pWndChild = pWndPrev->GetLastActivePopup();

//		if(pWndChild->IsIconic())
//			pWndPrev->ShowWindow(SW_RESTORE);

//		pWndChild->SetForegroundWindow();
//	}	
//	return FALSE;
//}
//ReleaseMutex(hMutex);
	//防止程序重复执行-E

	//应用程序清单使用ComCtl32.dll版本6或更高版本
	//如果指定使用,则必须在Windows XP上使用InitCommonControlsEx()。
	//如果不使用InitCommonControlsEx(),则无法创建窗口。
	INITCOMMONCONTROLSEX InitCtrls;
	InitCtrls.dwSize = sizeof(InitCtrls);
	//包含应用程序要使用的所有公共控件类
	//请设置此项。
	InitCtrls.dwICC = ICC_WIN95_CLASSES;
	InitCommonControlsEx(&InitCtrls);

	CWinApp::InitInstance();

	//KYH2022.03.11生成dump
	SetUnhandledExceptionFilter(TopLevelExceptionFilterCallBack);

	AfxEnableControlContainer();

	//在对话框中显示壳树视图或
	//如果包含shell列表视图控件,请创建shell管理器。
	CShellManager* pShellManager = new CShellManager;

	//添加190716YWS以删除Alg Task中所有过期的日志
	int uPeriodDay = GetPrivateProfileInt(_T("LogInfo"), _T("PeriodDay"), 30, VS_ALGORITHM_TASK_INI_FILE);

	//创建Log对象
	for (int i = 0; i < MAX_CAMERA_COUNT; i++)
	{
		CString strCam;
		strCam.Format(_T("CAM%02d"), i);
		m_pLogWriterCam[i] = new CAlgLogWriter(ALGORITHM_LOG_PATH, strCam.GetBuffer(0), uPeriodDay/*30*/, 0, true);
	}
	m_pLogWriterProc = new CAlgLogWriter(ALGORITHM_LOG_PATH, _T("PROCESS"), uPeriodDay/*30*/, 0, true);
	m_pLogWriterTact = new CAlgLogWriter(ALGORITHM_LOG_PATH, _T("TACT"), uPeriodDay/*30*/, 0, true);
	m_pLogWriterComm = new CAlgLogWriter(ALGORITHM_LOG_PATH, _T("COMMUNICATION"), uPeriodDay/*30*/, 0, true);

	CString strRegKey;

	//配置文件Load
	m_Config.Load();

	//创建Grab Step最大数量
	m_pGrab_Step = new STRU_INFO_GRAB_STEP[MAX_GRAB_STEP_COUNT];

	//连接Visual Station
	m_bExecIPC = TRUE;
	m_hEventIPCThreadAlive = NULL;
	m_hEventIPCThreadAlive = CreateEvent(NULL, TRUE, FALSE, NULL);
	ResetEvent(m_hEventIPCThreadAlive);
	m_pVSThread = AfxBeginThread(ThreadConnectVSServer, this);

	// Disk Check	
	m_hEventDiskThreadAlive = NULL;
	m_hEventDiskThreadAlive = CreateEvent(NULL, TRUE, FALSE, NULL);

	// Algo Thread Check Flag hjf	
	m_hEventDiskThreadAlive = NULL;
	m_hEventDiskThreadAlive = CreateEvent(NULL, TRUE, FALSE, NULL);

	m_pDiskInfo = new CDiskInfo();
	m_pDiskInfo->CreateDriveInfoVector();
	CString strDrive, strTemp;
	if (!theApp.m_Config.GetSimualtionMode())
	{
		int nCount = 0;
		while (1)
		{
			if (!AfxExtractSubString(strDrive, m_Config.GetUseDrive(), nCount++, ','))
				break;
			m_pDiskInfo->SetUseDisk(strDrive);
		}

		m_bExecDisk = TRUE;
		//m_pDiskCheckThread = AfxBeginThread(ThreadDiskCheck, this);

	}
	else
	{
		strDrive = m_Config.GetSimulationDrive();
		m_pDiskInfo->SetUseDisk(strDrive.Left(1));
		//strDrive = strDrive.Left(1) + _T("_Drive") + _T("\\");
	}

	strRegKey.Format(_T("BOE 11 Sequence Task"));

	LocalFree(pStr);

	//标准初始化
	//要减小不使用这些功能的最终可执行文件的大小,请执行以下操作:
	//下面不需要的特定初始化
	//必须删除例程
	//更改保存该设置的注册表项。
	//TODO:将此字符串与公司或组织的名称相同
	//需要修改为适当的内容。	
	SetRegistryKey(strRegKey);

	//初始化
	m_fnInitFunc();
	//检查算法运行超时
	m_hEventAlgoThreadTimeOutAlive = NULL;
	m_hEventAlgoThreadTimeOutAlive = CreateEvent(NULL, TRUE, FALSE, NULL);
	ResetEvent(m_hEventAlgoThreadTimeOutAlive);
	m_bExecAlgoThrBusy = TRUE;
	m_pAlgoCheckTimeOutThread = AfxBeginThread(AlgoThreadTimeOutCheck, this);
	//是否使用YHS 18.03.12-Merge Tool
	m_nInspStep = GetPrivateProfileInt(_T("MergeTool"), _T("USE_MERGE_TOOL"), 1, INIT_FILE_PATH) ? true : false;

	m_pDlg = new CVSAlgorithmTaskDlg();
	INT_PTR nResponse = m_pDlg->DoModal();

	if (nResponse == IDOK)
	{
		//TODO:单击此处的"确定",在对话框丢失时处理
		//放置代码。
	}
	else if (nResponse == IDCANCEL)
	{
		//TODO:单击此处的"取消",在对话框丢失时处理
		//放置代码。
	}

	//等待IPC结束
	m_bExecIPC = FALSE;
	::WaitForSingleObject(m_hEventIPCThreadAlive, INFINITE);
	ResetEvent(m_hEventIPCThreadAlive);
	if (m_bExecDisk)
	{
		m_bExecDisk = FALSE;
		::WaitForSingleObject(m_hEventDiskThreadAlive, INFINITE);
	}
	ResetEvent(m_hEventDiskThreadAlive);

	//等待检查线程超时结束 hjf
	if (m_bExecAlgoThrBusy)
	{
		m_bExecAlgoThrBusy = FALSE;
		::WaitForSingleObject(m_hEventAlgoThreadTimeOutAlive, 10);
	}
	ResetEvent(m_hEventAlgoThreadTimeOutAlive);

	m_pVSThread = NULL;
	m_pDiskCheckThread = NULL;

	for (int i = 0; i < MAX_CAMERA_COUNT; i++)
		SAFE_DELETE(m_pLogWriterCam[i]);
	SAFE_DELETE(m_pLogWriterProc);
	SAFE_DELETE(m_pLogWriterTact);
	SAFE_DELETE(m_pLogWriterComm);
	SAFE_DELETE(m_pDlg);
	SAFE_DELETE_ARR(m_pGrab_Step);
	SAFE_DELETE(m_pDiskInfo);

	//删除上面创建的shell管理器。
	if (pShellManager != NULL)
	{
		delete pShellManager;
	}

	//对话框已关闭,为了在不启动应用程序的消息泵的情况下结束应用程序,请将FALSE
	//返回
	return FALSE;
}

UINT CVSAlgorithmTaskApp::ThreadConnectVSServer(LPVOID pParam)
{
	BOOL bRet = FALSE;

	CVSAlgorithmTaskApp* pThis = (CVSAlgorithmTaskApp*)pParam;

	//连接Visual Station
	int					nRet;

	do
	{
		nRet = pThis->m_fnConectVisualStation();
		if (APP_OK != nRet)		Sleep(1000);
	} while (APP_OK != nRet && pThis->m_bExecIPC);

	printf("Internal Sequence Start \n");

	if (pThis->m_AlgorithmTask != NULL)
		pThis->m_AlgorithmTask->SendNotifyInitial();

	//VS状态检查
	while (pThis->m_bExecIPC)
	{
		//VS连接状态Get
		if (pThis->m_AlgorithmTask != NULL)
			//VS连接状态Get
			pThis->SetIPCState(pThis->m_AlgorithmTask->GetVSState());
		Sleep(1000);
	}

	pThis->SetIPCState(FALSE);
	//SAFE_DELETE(pInternalServer);

	SetEvent(pThis->m_hEventIPCThreadAlive);
	printf("Internal Sequence End \n");

	return 0;
}

int CVSAlgorithmTaskApp::m_fnConectVisualStation()
{
	if (m_AlgorithmTask != NULL)
	{
		return APP_OK;
	}
	TCHAR szAlgLogPath[100] = { 0, }, szSeqFileName[100] = { 0, }, szServerIP[100] = { 0, };
	int nTaskNo, nServerPort, uPeriodDay, uWriteTerm, uWriteLogLevel;

	//每个PC只能有一个Sequence Task
	nTaskNo = (m_Config.GetPCNum() * 100)
		+ (GetPrivateProfileInt(_T("VSServerInfo"), _T("TaskNo"), 21, VS_ALGORITHM_TASK_INI_FILE));
	nServerPort = GetPrivateProfileInt(_T("VSServerInfo"), _T("ServerPort"), 5000, VS_ALGORITHM_TASK_INI_FILE);
	uPeriodDay = GetPrivateProfileInt(_T("LogInfo"), _T("PeriodDay"), 30, VS_ALGORITHM_TASK_INI_FILE);
	uWriteTerm = GetPrivateProfileInt(_T("LogInfo"), _T("WriteTerm"), 0, VS_ALGORITHM_TASK_INI_FILE);
	uWriteLogLevel = GetPrivateProfileInt(_T("LogInfo"), _T("WriteLogLevel"), 3, VS_ALGORITHM_TASK_INI_FILE);
	//添加对当前解决方案运行驱动器的判断
	GetPrivateProfileString(_T("LogInfo"), _T("Path"), theApp.m_Config.GETCmdDRV() + _T(":\\VisualStation\\VSLOG\\VSPLC"), szAlgLogPath, sizeof(szAlgLogPath), VS_ALGORITHM_TASK_INI_FILE);
	GetPrivateProfileString(_T("LogInfo"), _T("LogName"), _T("CRUX"), szSeqFileName, sizeof(szSeqFileName), VS_ALGORITHM_TASK_INI_FILE);
	GetPrivateProfileString(_T("VSServerInfo"), _T("ServerIP"), _T("127.0.0.1"), szServerIP, sizeof(szServerIP), VS_ALGORITHM_TASK_INI_FILE);

	m_AlgorithmTask = new TaskInterfacer(nTaskNo, nServerPort, szServerIP, szAlgLogPath,
		szSeqFileName, false, uPeriodDay, uWriteTerm, uWriteLogLevel);
	int nRet = m_AlgorithmTask->Start();
	if (nRet != APP_OK)
	{
		SAFE_DELETE(m_AlgorithmTask);
	}
	return nRet;
}

//以后需要时更改为DLL结构
double CVSAlgorithmTaskApp::CallFocusValue(Mat matSrc, CRect rect)
{
	//如果画面缓冲区没有值
//if (matSrc == NULL) return 0;
	if (matSrc.empty()) return 0;

	//如果超过画面范围值
	if (rect.left < 0) return 0;
	if (rect.top < 0) return 0;
	if (rect.right > matSrc.cols) return 0;
	if (rect.bottom > matSrc.rows) return 0;

	//Rect设置异常
	if (rect.left == rect.right) return 0;
	if (rect.top == rect.bottom) return 0;

	Mat matSobel;
	Mat temp = Mat::zeros(matSrc.rows, matSrc.cols, matSrc.type());

	// GaussianBlur
	//Cv2.GaussianBlur(matSrc, matSobel, new OpenCvSharp.Size(3, 3), 1.5);
	//cv::imwrite("E:\\IMTC\\imageTest1.bmp", matSrc);
	cv::GaussianBlur(matSrc, matSobel, cvSize(3, 3), 1.5);
	//cv::imwrite("E:\\IMTC\\imageTest2.bmp", matSobel);
	// Sobel edge
	//Cv2.Sobel(matSobel, matSobel, MatType.CV_8UC1, 1, 1);
	//cv::Sobel(matSobel,matSobel,CV_8UC1,1,1,5);
	//cv::imwrite("E:\\IMTC\\imageTest3.bmp", matSobel);

	//////////////////////////////////////////////////////////////////////////
		//17.11.01-返回到标准偏差

		//获取mean&stdDev
	cv::Scalar m, s;
	cv::meanStdDev(matSobel, m, s);

	matSobel.release();

	return s[0];
	//////////////////////////////////////////////////////////////////////////

	long nSum = 0;

	for (int y = rect.top + 1; y < rect.bottom - 1; y++)
	{
		for (int x = rect.left + 1; x < rect.right - 1; x++)
		{
			long h1 = -matSobel.at<BYTE>(y - 1, x - 1)
				- 2 * matSobel.at<BYTE>(y - 1, x)
				- matSobel.at<BYTE>(y - 1, x + 1)
				+ matSobel.at<BYTE>(y + 1, x - 1)
				+ 2 * matSobel.at<BYTE>(y + 1, x)
				+ matSobel.at<BYTE>(y + 1, x + 1);

			long h2 = -matSobel.at<BYTE>(y - 1, x - 1)
				- 2 * matSobel.at<BYTE>(y, x - 1)
				- matSobel.at<BYTE>(y + 1, x - 1)
				+ matSobel.at<BYTE>(y - 1, x + 1)
				+ 2 * matSobel.at<BYTE>(y, x + 1)
				+ matSobel.at<BYTE>(y + 1, x + 1);

			nSum += (long)sqrt((double)(h1 * h1 + h2 * h2));

			temp.at<BYTE>(y, x) = (BYTE)sqrt((double)(h1 * h1 + h2 * h2));
		}
	}

	//	cv::imwrite("E:\\IMTC\\imageTest2.bmp", temp);

		//	设置多少Rect
	// 		for (int y = rect.top; y < rect.bottom; y++)
	// 		{
		//			//获取竖线的第一个数组地址
		//			//已固定的地址值(不必使用fixed)
	// 			byte* ptr = (byte*)matSobel.ptr(y) + rect.left;
	// 	
	// 			for (int x = rect.left; x < rect.right; x++)
	// 			{
		//			//累计GV值
	// 				nSum += *ptr++;
	// 			}
	// 		}

		//画面缓冲区关闭
	matSobel.release();

	//在Rect内,亮度平均值	
	return nSum / (float)(rect.Width() * rect.Height());
}

int CVSAlgorithmTaskApp::GetInsTypeIndex(int nImageNum, int nCameraNum, int nROINum, CString strAlgName)
{
	for (int i = 0; i < MAX_MEM_SIZE_E_ALGORITHM_NUMBER; i++)
	{
		if (strAlgName.CompareNoCase(m_pGrab_Step[nImageNum].stInfoCam[nCameraNum].ROI[nROINum].AlgorithmList[i].strAlgName) == 0)
			return i;
	}
	return -1;
}

bool CVSAlgorithmTaskApp::ReadAlgorithmParameter(TCHAR* strModelPath, TCHAR* strCornerEdgePath)
{
	//文件是否存在
	CFileFind find;
	BOOL bFindFile = FALSE;
	CString strMsg = _T("");

	CCPUTimer timerReadParameter;
	timerReadParameter.Start();

	//ReadPartitionBlockParameter 设置分区参数 hjf

	ReadPartitionBlockParameter(strModelPath);

	//////////////////////////////////////加载算法列表文件
	CString strAlgListXMLPath;
	//添加对当前解决方案运行驱动器的判断
	strAlgListXMLPath.Format(_T("%s:\\IMTC\\Text\\ALGORITHM_LIST.xml"), theApp.m_Config.GETCmdDRV());

	/**
	 * AI模型初始化.
	 *
	 * \param strModelPath
	 * \param strCornerEdgePath
	 * \return
	 */
	 //TCHAR tcConfig[100] = { 0, };
	 //GetPrivateProfileString(_T("AI Inspect Config"), _T("ConfigPath"), _T(""), tcConfig, sizeof(tcConfig), VS_ALGORITHM_TASK_INI_FILE);
	 ////std::string sConfig = TCHARToString(tcConfig);
	 //std::wstring temp1(tcConfig);
	 ////AI_Initialization(sConfig);
	 //const wchar_t* wCStr = temp1.c_str();
	 //std::string str(wCStr, wCStr + temp1.size());
	 //auto config = read_json_from_file(str.c_str());

	 // Initilize the AI Runtime
	 //stAIConfigInfo aiIniConfig(config["initConfig"]);
	 //GetAIRuntime()->InitRuntime(aiIniConfig);

	 //// Initilize AI models.
	 //GetAIRuntime()->CreateModle(config["modelInfo"]);
	 //theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, TRUE, FALSE, _T("[Initialize] Init AI Model Complete!!"));
	bFindFile = find.FindFile(strAlgListXMLPath);
	find.Close();

	if (!bFindFile)
	{
		strMsg.Format(_T("Not found algorithm list xml file. (%s)"), strAlgListXMLPath);
		AfxMessageBox(strMsg);
		return false;
	}

	//加载XML文件
	CMarkup xmlAlgList;
	if (!xmlAlgList.Load(strAlgListXMLPath))
	{
		strMsg.Format(_T("Model xml load fail. (%s)"), strAlgListXMLPath);
		AfxMessageBox(strMsg);
		return false;
	}

	xmlAlgList.FindElem();		// ALGORITHM_LIST
	xmlAlgList.IntoElem();		// inside ALGORITHM_LIST

	CString strAlgorithmName = _T("");
	CMarkup* xmlAlgorithmList = new CMarkup(xmlAlgList);
	//m_MapAlgList.clear();
	map<CString, UINT> MapAlgList;
	MapAlgList.clear();
	for (int nAlgListIndex = 0; nAlgListIndex < MAX_MEM_SIZE_E_ALGORITHM_NUMBER; nAlgListIndex++)
	{
		if (xmlAlgorithmList->FindElem(_T("ALGORITHM_%d"), nAlgListIndex))
		{
			strAlgorithmName = xmlAlgorithmList->GetAttrib(_T("Name"));
			MapAlgList[strAlgorithmName] = nAlgListIndex;
			xmlAlgorithmList->ResetMainPos();
		}
	}
	SetAlgoritmList(MapAlgList);
	SAFE_DELETE(xmlAlgorithmList);

	///////////////////////////////////////////////////////////////////////////////////////////////
	CString strDefItemListXMLPath;
	//添加对当前解决方案运行驱动器的判断
	strDefItemListXMLPath.Format(_T("%s:\\IMTC\\Text\\DEFITEM_LIST.xml"), theApp.m_Config.GETCmdDRV());

	bFindFile = find.FindFile(strDefItemListXMLPath);
	find.Close();

	if (!bFindFile)
	{
		strMsg.Format(_T("Not found defect item list xml file. (%s)"), strDefItemListXMLPath);
		AfxMessageBox(strMsg);
		return false;
	}

	//加载XML文件
	CMarkup xmlDefectItem;
	if (!xmlDefectItem.Load(strDefItemListXMLPath))
	{
		strMsg.Format(_T("Model xml load fail. (%s)"), strDefItemListXMLPath);
		AfxMessageBox(strMsg);
		return false;
	}

	xmlDefectItem.FindElem();		// DEF_ITEM
	xmlDefectItem.IntoElem();		// inside DEF_ITEM

	CString strDefItemName = _T("");
	CMarkup* xmlDefItemList = new CMarkup(xmlDefectItem);
	//m_MapDefItemList.clear();
	map<CString, UINT> MapDefItemList;
	MapDefItemList.clear();
	for (int nDefItemIndex = 0; nDefItemIndex < MAX_MEM_SIZE_E_DEFECT_NAME_COUNT; nDefItemIndex++)
	{
		if (xmlDefItemList->FindElem(_T("DefType_%d"), nDefItemIndex))
		{
			strDefItemName = xmlDefItemList->GetAttrib(_T("Name"));
			MapDefItemList[strDefItemName] = nDefItemIndex;
			xmlDefItemList->ResetMainPos();
		}
	}
	SetDefectItemList(MapDefItemList);
	SAFE_DELETE(xmlDefItemList);

	/////////////////////////////////////加载模型文件
	CString strModelXMLPath;
	//更改RMS对应的模型文件路径
//strModelXMLPath.Format(_T("%s\\%s\\Grab_List.xml"), MODEL_FILE_PATH, strModelID);
	strModelXMLPath.Format(_T("%s\\Grab_List.xml"), strModelPath);

	bFindFile = find.FindFile(strModelXMLPath);
	find.Close();
	if (!bFindFile)
	{
		strMsg.Format(_T("Not found model xml file. (%s)"), strModelXMLPath);
		AfxMessageBox(strMsg);
		return false;
	}

	//加载XML文件
	CMarkup xmlFile;
	if (!xmlFile.Load(strModelXMLPath))
	{
		strMsg.Format(_T("Model xml load fail. (%s)"), strModelXMLPath);
		AfxMessageBox(strMsg);
		return false;
	}

	xmlFile.FindElem();		// Recipe
	xmlFile.IntoElem();		// inside Recipe

	//传达LJH配方版本//////////////////////////////////////////////////////////////////////////////////////////////////////
	CMarkup* xmlList = new CMarkup(xmlFile);

	CString EQP = _T("");
	CString changeParamListDate = _T("");
	CString testAlgType = _T("");
	CString verMsg = _T("");

	if (xmlList->FindElem(_T("VERSION")))
	{
		CString verFullname;
		verFullname = xmlList->GetAttrib(_T("NAME"));

		//要分开保存的数组
		CString rcpVer[3] = { 0, };

		for (int j = 0; j < 3; j++)
			AfxExtractSubString(rcpVer[j], verFullname, j, '_');

		EQP.Format(_T("%s"), rcpVer[0]);
		changeParamListDate.Format(_T("%s"), rcpVer[1]);
		testAlgType.Format(_T("%s"), rcpVer[2]);
	}
	SAFE_DELETE(xmlList)

		if (testAlgType == "")
			verMsg.Format((_T("%s"), EQP) + _T("_") + (_T("%s"), changeParamListDate));
		else
			verMsg.Format((_T("%s"), EQP) + _T("_") + (_T("%s"), changeParamListDate) + _T("_") + (_T("%s"), testAlgType));

	//	theApp.m_AlgorithmTask->VS_Send_RcpVer_To_UI(verMsg);
		/////////////////////////////////////////////////////////////////////////////////////////////////////////

	try {
		//memset(m_pGrab_Step, 0, sizeof(STRU_INFO_GRAB_STEP) * MAX_GRAB_STEP_COUNT);
		STRU_INFO_GRAB_STEP* pGrabStepInfo = new STRU_INFO_GRAB_STEP[MAX_GRAB_STEP_COUNT];

		//关于Grab Step
		int nGrabNum = 0;
		//#pragma omp parallel default(shared)
		{
			//#pragma omp parallel for reduction(+:nGrabNum) schedule(dynamic)
			for (int nGrabCnt = 0; nGrabCnt < MAX_GRAB_STEP_COUNT; nGrabCnt++)
			{
				CMarkup* xmlList = new CMarkup(xmlFile);
				if (xmlList->FindElem(_T("GRAB_%d"), nGrabCnt))
				{
					nGrabNum++;
#pragma region >> Get Grab Step List
					STRU_INFO_GRAB_STEP* stInfoGrabStep = &pGrabStepInfo[nGrabCnt];
					stInfoGrabStep->bUse = xmlList->GetBoolAttrib(_T("USE"));
					//17.07.03 Step修改为无条件加载信息
				// 				if (stInfoGrabStep->bUse)
				// 				{
					stInfoGrabStep->eImgClassify = _ttoi(xmlList->GetAttrib(_T("AlgNo")));
					_tcscpy(stInfoGrabStep->strGrabStepName, xmlList->GetAttrib(_T("NAME")));

					///////////////////////////////////////////////////////////////////////////////////////
					CMarkup xmlPatternFile, xmlBlockPatternFile;

					CString strPatternXMLPath, strBlockPatternXMLPath;//strBlockPatternXMLPath 按画面分区文件路径

					//strPatternXMLPath.Format(_T("%s\\%s\\Grab_List\\%s.xml"), MODEL_FILE_PATH, strModelID, stInfoGrabStep->strGrabStepName);
					strPatternXMLPath.Format(_T("%s\\Grab_List\\%s.xml"), strModelPath, stInfoGrabStep->strGrabStepName);
					//按画面分区文件路径 hjf
					strBlockPatternXMLPath.Format(_T("%s\\BlockJudgeParams\\%s.xml"), strModelPath, stInfoGrabStep->strGrabStepName);

					CFileFind findPatternFile, findBlockPatternFile;
					BOOL bFindPatternFile = findPatternFile.FindFile(strPatternXMLPath);
					BOOL bFindBlockPatternFile = findBlockPatternFile.FindFile(strBlockPatternXMLPath);
					findBlockPatternFile.Close();
					findPatternFile.Close();
					if (!bFindPatternFile || !bFindBlockPatternFile)
					{
						strMsg.Format(_T("Not found pattern xml file. (%s)"), strPatternXMLPath);
						AfxMessageBox(strMsg);
					}
					else
					{
						//if (!xmlBlockPatternFile.Load(strBlockPatternXMLPath))
						//{
						//	strMsg.Format(_T("Pattern xml load fail. (%s)"), strBlockPatternXMLPath);
						//	AfxMessageBox(strMsg);
						//}
						//else 
						//{
						//	xmlBlockPatternFile.FindElem();// 获取到Recipe
						//	xmlBlockPatternFile.IntoElem();		// inside Recipe
						//	
						//	/*
						//	<BLOCK_0>
						//		<ALGORITHM_0>
						//			<DEF_ITEM_0>
						//				<JUDGEMENT_0>
						//					
						//				</JUDGEMENT_0>
						//			</DEF_ITEM_0>
						//		</ALGORITHM_0>
						//	</BLOCK_0>
						//	*/
						//}
						if (!xmlPatternFile.Load(strPatternXMLPath) || !xmlBlockPatternFile.Load(strBlockPatternXMLPath))
						{
							strMsg.Format(_T("Pattern xml load fail. (%s)"), strPatternXMLPath);
							AfxMessageBox(strMsg);
						}
						else
						{
							xmlPatternFile.FindElem();		// Recipe
							xmlPatternFile.IntoElem();		// inside Recipe

							xmlBlockPatternFile.FindElem();		// Recipe
							xmlBlockPatternFile.IntoElem();		// inside Recipe


							//#pragma omp parallel shared(nGrabCnt)
							{
								//特定于相机的信息
#pragma omp parallel for schedule(dynamic)		//由于当前相机数量为1-2个,效率低下
								for (int nCamCnt = 0; nCamCnt < MAX_CAMERA_COUNT; nCamCnt++)
								{
#pragma region >> Get Parameter Each Camera
									CMarkup* xmlPattern = new CMarkup(xmlPatternFile);
									CMarkup* xmlBlockPattern = new CMarkup(xmlBlockPatternFile);
									if (xmlPattern->FindElem(_T("CAM_%d"), nCamCnt))
									{
										stInfoGrabStep->nCamCnt++;
										STRU_INFO_CAMERA* stInfoCam = &stInfoGrabStep->stInfoCam[nCamCnt];
										stInfoCam->bUse = xmlPattern->GetBoolAttrib(_T("USE"));
										if (stInfoCam->bUse)
										{
											xmlPattern->IntoElem();
											if (xmlPattern->FindElem(_T("AD")))		///17.04.11添加AD参数
											{
												stInfoCam->bUseAD = xmlPattern->GetBoolAttrib(_T("USE"));
												if (stInfoCam->bUseAD)
												{
													xmlPattern->IntoElem();
													for (int nADPrmCnt = 0; nADPrmCnt < MAX_MEM_SIZE_AD_PARA_TOTAL_COUNT; nADPrmCnt++)
													{
														if (xmlPattern->FindElem(_T("PARAM_%d"), nADPrmCnt))
														{
															stInfoCam->dADPara[nADPrmCnt] = _ttof(xmlPattern->GetAttrib(_T("VALUE")));
															xmlPattern->ResetMainPos();
														}
													}
													xmlPattern->OutOfElem();
												}
												xmlPattern->ResetMainPos();
											}
											//xmlPattern->OutOfElem();
											for (int nNonROICnt = 0; nNonROICnt < MAX_MEM_SIZE_E_INSPECT_AREA; nNonROICnt++)
											{

												if (xmlPattern->FindElem(_T("NON_%d"), nNonROICnt))
												{
													stInfoCam->nNonROICnt++;
													INSP_AREA* stInfoArea = &stInfoCam->NonROI[nNonROICnt];
													stInfoArea->bUseROI = xmlPattern->GetBoolAttrib(_T("USE"));
													if (stInfoArea->bUseROI)
													{
														COPY_CSTR2TCH(stInfoArea->strROIName, xmlPattern->GetAttrib(_T("NAME")), sizeof(stInfoArea->strROIName));
														stInfoArea->rectROI.left = _ttoi(xmlPattern->GetAttrib(_T("START_X")));
														stInfoArea->rectROI.top = _ttoi(xmlPattern->GetAttrib(_T("START_Y")));
														stInfoArea->rectROI.right = _ttoi(xmlPattern->GetAttrib(_T("END_X")));
														stInfoArea->rectROI.bottom = _ttoi(xmlPattern->GetAttrib(_T("END_Y")));
													}
													xmlPattern->ResetMainPos();
												}
											}
											for (int nRndROICnt = 0; nRndROICnt < MAX_MEM_SIZE_E_INSPECT_AREA; nRndROICnt++)
											{
												if (xmlPattern->FindElem(_T("RND_%d"), nRndROICnt))
												{
													stInfoCam->nRndROICnt++;
													INSP_AREA* stInfoArea = &stInfoCam->RndROI[nRndROICnt];
													stInfoArea->bUseROI = xmlPattern->GetBoolAttrib(_T("USE"));
													if (stInfoArea->bUseROI)
													{
														COPY_CSTR2TCH(stInfoArea->strROIName, xmlPattern->GetAttrib(_T("NAME")), sizeof(stInfoArea->strROIName));
														stInfoArea->rectROI.left = _ttoi(xmlPattern->GetAttrib(_T("START_X")));
														stInfoArea->rectROI.top = _ttoi(xmlPattern->GetAttrib(_T("START_Y")));
														stInfoArea->rectROI.right = _ttoi(xmlPattern->GetAttrib(_T("END_X")));
														stInfoArea->rectROI.bottom = _ttoi(xmlPattern->GetAttrib(_T("END_Y")));
													}
													xmlPattern->ResetMainPos();
												}
											}
											//2019.02.20 for Hole ROI
											// 2024.07 modify for polmark
											int polmarkCount = 0;
											int holeCount = 0;
											for (int nHoleROICnt = 0; nHoleROICnt < MAX_MEM_SIZE_E_INSPECT_AREA; nHoleROICnt++)
											{
												if (xmlPattern->FindElem(_T("HOLE_%d"), nHoleROICnt))
												{
													CString roiName = xmlPattern->GetAttrib(_T("NAME"));
													bool isPolMark = roiName.GetLength() > 4 && roiName.Left(4).CompareNoCase(_T("pol_")) == 0;
													
													INSP_AREA* stInfoArea = isPolMark ? &stInfoCam->PolMarkROI[polmarkCount] : &stInfoCam->HoleROI[holeCount];
													stInfoArea->bUseROI = xmlPattern->GetBoolAttrib(_T("USE"));
													if (stInfoArea->bUseROI)
													{
														COPY_CSTR2TCH(stInfoArea->strROIName, roiName, sizeof(stInfoArea->strROIName));
														stInfoArea->rectROI.left = _ttoi(xmlPattern->GetAttrib(_T("START_X")));
														stInfoArea->rectROI.top = _ttoi(xmlPattern->GetAttrib(_T("START_Y")));
														stInfoArea->rectROI.right = _ttoi(xmlPattern->GetAttrib(_T("END_X")));
														stInfoArea->rectROI.bottom = _ttoi(xmlPattern->GetAttrib(_T("END_Y")));
													}

													if (isPolMark) {
														polmarkCount++;
													}
													else {
														stInfoCam->nHoleROICnt++;
													}
													xmlPattern->ResetMainPos();
												}
											}
											stInfoCam->nHoleROICnt = holeCount;
											stInfoCam->nPolMarkROICnt = polmarkCount;

											for (int nROICnt = 0; nROICnt < 1/*MAX_MEM_SIZE_ROI_COUNT*/; nROICnt++)
											{
												if (xmlPattern->FindElem(_T("ROI_%d"), nROICnt))
												{
													stInfoCam->nROICnt++;
													STRU_INFO_ROI* stInfoROI = &stInfoCam->ROI[nROICnt];
													stInfoROI->bUseROI = xmlPattern->GetBoolAttrib(_T("USE"));
													if (stInfoROI->bUseROI)
													{
														_tcscpy(stInfoROI->strROIName, xmlPattern->GetAttrib(_T("NAME")));
														stInfoROI->rectROI.left = _ttoi(xmlPattern->GetAttrib(_T("START_X")));
														stInfoROI->rectROI.top = _ttoi(xmlPattern->GetAttrib(_T("START_Y")));
														stInfoROI->rectROI.right = _ttoi(xmlPattern->GetAttrib(_T("END_X")));
														stInfoROI->rectROI.bottom = _ttoi(xmlPattern->GetAttrib(_T("END_Y")));

														xmlPattern->IntoElem();
														//算法列表特定信息
	////////////////////////////////////////////////////////分区参数导入 S
														for (UINT stBlockDefectItemIndex = 0; stBlockDefectItemIndex < GetnBlockCountX() * GetnBlockCountY(); stBlockDefectItemIndex++) {

															for (int nAlgCnt = 0; nAlgCnt < MAX_MEM_SIZE_E_ALGORITHM_NUMBER; nAlgCnt++)
															{
																if (xmlPattern->FindElem(_T("ALGORITHM_%d"), nAlgCnt))
																{
																	///模式-相机-星形算法的条目可能各不相同(在ex:CAM1中,仅在XML中填写POINT算法并显示GUI)
																																//不按顺序加载,而是先获取相应的算法名称,查找索引并将其写入参数表(请参见ALGORITHM_LIST.xml)
																	CString strAlgorithmName = xmlPattern->GetAttrib(_T("NAME"));
																	int nAlgorithmIndex = theApp.GetAlgorithmIndex(strAlgorithmName);
																	if (nAlgorithmIndex < 0)
																	{
																		AfxMessageBox(_T("Not Found Algorithm : ") + strAlgorithmName + _T("\r\n(") + strAlgListXMLPath + _T(")"));
																	}
																	STRU_PARAM_ALG* stInfoAlg = &stInfoROI->AlgorithmList[nAlgorithmIndex];
																	stInfoAlg->bAlgorithmUse = xmlPattern->GetBoolAttrib(_T("USE"));
																	if (stInfoAlg->bAlgorithmUse)
																	{
																		COPY_CSTR2TCH(stInfoAlg->strAlgName, strAlgorithmName, sizeof(stInfoAlg->strAlgName));
																		xmlPattern->IntoElem();

																		//算法参数
																		for (int nAlgPrmCnt = 0; nAlgPrmCnt < MAX_MEM_SIZE_ALG_PARA_TOTAL_COUNT; nAlgPrmCnt++)
																		{
																			if (xmlPattern->FindElem(_T("PARAM_%d"), nAlgPrmCnt))
																			{
																				stInfoAlg->dPara[nAlgPrmCnt] = _ttof(xmlPattern->GetAttrib(_T("VALUE")));
																				xmlPattern->ResetMainPos();
																			}
																		}
																		xmlPattern->OutOfElem();
																		//获取分区行列
																		stInfoAlg->nBlockCountX = GetnBlockCountX();
																		stInfoAlg->nBlockCountY = GetnBlockCountY();
																		//#pragma region >> Set Parameter for Block
																																				//CMarkup* xmlBlockPattern = new CMarkup(xmlBlockPatternFile);//分区参数读取
																																				/*for (UINT stBlockDefectItemIndex = 0; stBlockDefectItemIndex < GetnBlockCountX() * GetnBlockCountY(); stBlockDefectItemIndex++)
																																				{*/
																		if (xmlBlockPattern->FindElem(_T("BLOCK_%d"), stBlockDefectItemIndex))
																		{

																			stPanelBlockJudgeInfo* stInfoDefItem = &stInfoAlg->stBlockDefectItem[stBlockDefectItemIndex];
																			stInfoDefItem->bBlockUse = xmlBlockPattern->GetBoolAttrib(_T("USE"));
																			stInfoDefItem->nBlockNum = stBlockDefectItemIndex;
																			//当前分区块启用 填充判定参数
																			if (stInfoDefItem->bBlockUse)//<BLOCK_%d>
																			{
																				xmlBlockPattern->IntoElem();//进入<ALGORITHM_%d>
																				//找到对应的算法节点
																				if (xmlBlockPattern->FindElem(_T("ALGORITHM_%d"), nAlgCnt))
																				{

																					xmlBlockPattern->IntoElem();//进入<DEF_ITEM_%d>

																					for (int nDefItemCnt = 0; nDefItemCnt < MAX_MEM_SIZE_E_DEFECT_NAME_COUNT; nDefItemCnt++)
																					{
																						if (xmlBlockPattern->FindElem(_T("DEF_ITEM_%d"), nDefItemCnt))
																						{
																							stInfoAlg->nDefectItemCount++;

																							CString strDefItemName = xmlBlockPattern->GetAttrib(_T("NAME"));
																							int nDefItemIndex = theApp.GetDefectTypeIndex(strDefItemName);
																							if (nDefItemIndex < 0)
																							{
																								AfxMessageBox(_T("Not Found Def Item : ") + strDefItemName + _T("\r\n(") + strDefItemListXMLPath + _T(")"));
																							}

																							STRU_DEFECT_ITEM* stBlockInfoDefItem = &stInfoDefItem->stDefectItem[nDefItemIndex];
																							stBlockInfoDefItem->bDefectItemUse = xmlBlockPattern->GetBoolAttrib(_T("USE"));
																							if (stBlockInfoDefItem->bDefectItemUse)
																							{
																								COPY_CSTR2TCH(stBlockInfoDefItem->strItemName, strDefItemName, sizeof(stBlockInfoDefItem->strItemName));

																								xmlBlockPattern->IntoElem();//进入<JUDGEMENT_%d>
																								for (int nJudgeCnt = 0; nJudgeCnt < MAX_MEM_SIZE_E_DEFECT_NAME_COUNT; nJudgeCnt++)
																								{
																									if (xmlBlockPattern->FindElem(_T("JUDGEMENT_%d"), nJudgeCnt))
																									{
																										STRU_JUDGEMENT* stInfoJudge = &stBlockInfoDefItem->Judgment[nJudgeCnt];
																										stInfoJudge->bUse = xmlBlockPattern->GetBoolAttrib(_T("USE"));
																										stInfoJudge->dValue = _ttof(xmlBlockPattern->GetAttrib(_T("VALUE")));
																										CString strCond = xmlBlockPattern->GetAttrib(_T("COND"));
																										if (strCond.CompareNoCase(_T("=")) == 0)			stInfoJudge->nSign = 0;
																										else if (strCond.CompareNoCase(_T("<>")) == 0)		stInfoJudge->nSign = 1;
																										else if (strCond.CompareNoCase(_T(">")) == 0)		stInfoJudge->nSign = 2;
																										else if (strCond.CompareNoCase(_T("<")) == 0)		stInfoJudge->nSign = 3;
																										else if (strCond.CompareNoCase(_T(">=")) == 0)		stInfoJudge->nSign = 4;
																										else if (strCond.CompareNoCase(_T("<=")) == 0)		stInfoJudge->nSign = 5;
																										xmlBlockPattern->ResetMainPos();
																									}
																								}
																								xmlBlockPattern->OutOfElem();
																							}
																							xmlBlockPattern->ResetMainPos();
																						}
																					}
																					xmlBlockPattern->OutOfElem();
																					xmlBlockPattern->ResetMainPos();
																				}
																				xmlBlockPattern->OutOfElem();
																			}
																			xmlBlockPattern->ResetMainPos();
																		}

																	}

																	xmlPattern->ResetMainPos();

																}
															}
															//xmlPattern->OutOfElem();
														}
														////////////////////////////////////////////////////////分区参数导入 E
														xmlPattern->OutOfElem();

													}
													xmlPattern->ResetMainPos();
												}
												//xmlPattern->OutOfElem();
											}
										}
										// 读自动圆角ROI
										CString strAutoRndRoiPath;
										strAutoRndRoiPath.Format(_T("%s\\CornerEdge\\%s_ROIList.txt"), strCornerEdgePath, stInfoGrabStep->strGrabStepName);
										FILE* cornerRoiListFile = NULL;
										fopen_s(&cornerRoiListFile, (CStringA)strAutoRndRoiPath, ("r"));
										if (cornerRoiListFile != NULL) {
											int roiIdx = 0;
											while (!feof(cornerRoiListFile)) {
												INSP_AREA* rndRoi = &stInfoCam->AutoRndROI[roiIdx];
												fscanf_s(cornerRoiListFile, "%d,%d,%d,%d\n",
													&rndRoi->rectROI.left,
													&rndRoi->rectROI.top,
													&rndRoi->rectROI.right,
													&rndRoi->rectROI.bottom);
											}
											stInfoCam->nAutoRndROICnt++;
											fclose(cornerRoiListFile);
											cornerRoiListFile = NULL;
										}
									}
#pragma endregion
									SAFE_DELETE(xmlPattern)
										SAFE_DELETE(xmlBlockPattern)

								}
							}
						}
					}

					//17.10.24[Round]-加载模型文件Start						
					{
						//最多30个曲线区域
						for (int i = 0; i < MAX_MEM_SIZE_E_INSPECT_AREA; i++)
						{
							//临时文件存储路径
							CString strTemp;

							//更改RMS对应的模型文件路径

							//17.11.09[Round]-仅当不是Apply时读取文件
							strTemp.Format(_T("%s\\CornerEdge\\%s_%02d.EdgePT"), strCornerEdgePath, stInfoGrabStep->strGrabStepName, i);

							//需要确认文件是否存在
							CFileFind findEdgeFile;
							BOOL bFindEdgeFile = FALSE;

							bFindEdgeFile = findEdgeFile.FindFile(strTemp);
							findEdgeFile.Close();

							//如果有文件
							if (bFindEdgeFile)
							{

								FILE* out = NULL;
								fopen_s(&out, (CStringA)strTemp, ("r"));

								if (out != NULL)
								{

									cv::Point2i ptTemp;
									int j = 0;

									switch (theApp.m_Config.GetEqpType())
									{
									case EQP_AVI:
									case EQP_SVI:

										//读取近索引
										fscanf_s(out, "CornerIndex%d\n", &pGrabStepInfo[nGrabCnt].tRoundSet[i].nCornerMinLength);

										//检查转角是否存在于Cell区域中
										fscanf_s(out, "CornerInside%d,%d,%d,%d\n",
											&pGrabStepInfo[nGrabCnt].tRoundSet[i].nCornerInside[0],
											&pGrabStepInfo[nGrabCnt].tRoundSet[i].nCornerInside[1],
											&pGrabStepInfo[nGrabCnt].tRoundSet[i].nCornerInside[2],
											&pGrabStepInfo[nGrabCnt].tRoundSet[i].nCornerInside[3]);

										while (true)
										{
											fscanf_s(out, "%d,%d\n", &ptTemp.x, &ptTemp.y);

											//fscanf_s失败或坐标时退出
											if (pGrabStepInfo[nGrabCnt].tRoundSet[i].nContourCount > 0 &&
												pGrabStepInfo[nGrabCnt].tRoundSet[i].ptContours[j - 1].x == ptTemp.x &&
												pGrabStepInfo[nGrabCnt].tRoundSet[i].ptContours[j - 1].y == ptTemp.y)
												break;

											//添加坐标
									//我尝试将其用作矢量,但由于SetGrabStepInfomemcopy的部分,所以将其更改为数组
											pGrabStepInfo[nGrabCnt].tRoundSet[i].ptContours[j].x = ptTemp.x;
											pGrabStepInfo[nGrabCnt].tRoundSet[i].ptContours[j].y = ptTemp.y;
											pGrabStepInfo[nGrabCnt].tRoundSet[i].nContourCount++;

											j++;
										}
										break;
									case EQP_APP:
										//读取近索引
										fscanf_s(out, "CornerIndex%d\n", &pGrabStepInfo[nGrabCnt].tRoundSet[i].nContourCount);

										while (j != pGrabStepInfo[nGrabCnt].tRoundSet[i].nContourCount)
										{
											fscanf_s(out, "%d,%d\n", &ptTemp.x, &ptTemp.y);
											//添加坐标
									//我尝试将其用作矢量,但由于SetGrabStepInfomemcopy的部分,所以将其更改为数组
											pGrabStepInfo[nGrabCnt].tRoundSet[i].ptContours[j].x = ptTemp.x;
											pGrabStepInfo[nGrabCnt].tRoundSet[i].ptContours[j].y = ptTemp.y;
											j++;
										}
										break;
									}
									fclose(out);
									out = NULL;
								}
							}
						}
					}
					//17.10.24[Round]-加载模型文件End
					//[CHole]-加载模型文件Start						
					{
						//最多30个曲线区域
						for (int i = 0; i < MAX_MEM_SIZE_E_INSPECT_AREA; i++)
						{
							//临时文件存储路径
							CString strTemp2;

							//更改RMS对应的模型文件路径

							//17.11.09【Camera Hole】-仅当不是Apply时,读取文件
							strTemp2.Format(_T("%s\\CameraHole\\%s_%02d.EdgePT"), strCornerEdgePath, stInfoGrabStep->strGrabStepName, i);

							//需要确认文件是否存在
							CFileFind findEdgeFile2;
							BOOL bFindEdgeFile2 = FALSE;

							bFindEdgeFile2 = findEdgeFile2.FindFile(strTemp2);
							findEdgeFile2.Close();

							//如果有文件
							if (bFindEdgeFile2)
							{

								FILE* out = NULL;
								fopen_s(&out, (CStringA)strTemp2, ("r"));

								if (out != NULL)
								{

									cv::Point2i ptTemp;
									int j = 0;

									switch (theApp.m_Config.GetEqpType())
									{
									case EQP_AVI:
									case EQP_SVI:

										//读取近索引
										fscanf_s(out, "CameraHoleIndex%d\n", &pGrabStepInfo[nGrabCnt].tCHoleSet[i].nCornerMinLength);

										//检查转角是否存在于Cell区域中
										fscanf_s(out, "CameraHoleInside%d,%d,%d,%d\n",
											&pGrabStepInfo[nGrabCnt].tCHoleSet[i].nCornerInside[0],
											&pGrabStepInfo[nGrabCnt].tCHoleSet[i].nCornerInside[1],
											&pGrabStepInfo[nGrabCnt].tCHoleSet[i].nCornerInside[2],
											&pGrabStepInfo[nGrabCnt].tCHoleSet[i].nCornerInside[3]);

										while (true)
										{
											fscanf_s(out, "%d,%d\n", &ptTemp.x, &ptTemp.y);

											//fscanf_s失败或坐标时退出
											if (pGrabStepInfo[nGrabCnt].tCHoleSet[i].nContourCount > 0 &&
												pGrabStepInfo[nGrabCnt].tCHoleSet[i].ptContours[j - 1].x == ptTemp.x &&
												pGrabStepInfo[nGrabCnt].tCHoleSet[i].ptContours[j - 1].y == ptTemp.y)
												break;

											//添加坐标
									//我尝试将其用作矢量,但由于SetGrabStepInfomemcopy的部分,所以将其更改为数组
											pGrabStepInfo[nGrabCnt].tCHoleSet[i].ptContours[j].x = ptTemp.x;
											pGrabStepInfo[nGrabCnt].tCHoleSet[i].ptContours[j].y = ptTemp.y;
											pGrabStepInfo[nGrabCnt].tCHoleSet[i].nContourCount++;

											j++;
										}
										break;
									case EQP_APP:
										//读取近索引
										fscanf_s(out, "CornerIndex%d\n", &pGrabStepInfo[nGrabCnt].tCHoleSet[i].nContourCount);

										while (j != pGrabStepInfo[nGrabCnt].tCHoleSet[i].nContourCount)
										{
											fscanf_s(out, "%d,%d\n", &ptTemp.x, &ptTemp.y);
											//添加坐标
									//我尝试将其用作矢量,但由于SetGrabStepInfomemcopy的部分,所以将其更改为数组
											pGrabStepInfo[nGrabCnt].tCHoleSet[i].ptContours[j].x = ptTemp.x;
											pGrabStepInfo[nGrabCnt].tCHoleSet[i].ptContours[j].y = ptTemp.y;
											j++;
										}
										break;
									}
									fclose(out);
									out = NULL;
								}
							}
						}
					}
					//[CHole]-加载模型文件End
	// 				}
#pragma endregion
				}
				SAFE_DELETE(xmlList)
			}
		}
		SetGrabStepInfo(pGrabStepInfo);
		//SetBlockDefectFilteringParam()
		SAFE_DELETE_ARR(pGrabStepInfo);
		SetGrabStepCount(nGrabNum);
	}
	catch (...) {
		theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, TRUE, TRUE, _T("Failed Read Algorithm Parameter!!!"));
		return false;
	}

	timerReadParameter.End();
	theApp.WriteLog(eLOGTACT, eLOGLEVEL_DETAIL, FALSE, FALSE, _T("Read algorithm parameter tact time %.2f"), timerReadParameter.GetTime() / 1000.);

	return true;
}

bool CVSAlgorithmTaskApp::ReadPadInspParameter(TCHAR* strModelPath)
{
	// Delete Pad Area Info	
	DeletePadAreaInfo();

	CCPUTimer timerReadParameter;
	timerReadParameter.Start();

	try {

		CString strModelXMLPath;
		//文件路径
		strModelXMLPath.Format(_T("%s\\PAD_INSP_INFO"), strModelPath);

		//Pad Info目录数		
		CString strFileName, strDirName;
		CString strPadInspDir[E_PAD_AREA_COUNT] = { _T("PadArea"), _T("fiducial") , _T("AlignMark"),  _T("PadRoi")	  ,  _T("PadRoi") };
		CString strPadInspFile[E_PAD_AREA_COUNT] = { _T("PadArea"), _T("fiducial") , _T("AlignMark"),  _T("PadRoiInsp"),  _T("PadRoiNone") };

		STRU_INFO_PAD** struInfoPad = new STRU_INFO_PAD * [E_PAD_AREA_COUNT];
		for (int ncnt = 0; ncnt < E_PAD_AREA_COUNT; ncnt++) struInfoPad[ncnt] = new STRU_INFO_PAD[MAX_CAMERA_COUNT];

#ifdef _DEBUG
#else
#pragma omp parallel for
#endif
		for (int nPadInfoCnt = 0; nPadInfoCnt < (sizeof(strPadInspDir) / sizeof(*strPadInspDir)); nPadInfoCnt++)
		{
#ifdef _DEBUG
#else
#pragma omp parallel for
#endif	
			for (int nCamCnt = 0; nCamCnt < MAX_CAMERA_COUNT; nCamCnt++)
			{
				CString strFileName, strImgName;

				CString strPadInspPath;

				strFileName.Format(_T("%s_CAM%02d.xml"), strPadInspFile[nPadInfoCnt], nCamCnt);

				strPadInspPath.Format(_T("%s\\%s\\%s"), strModelXMLPath, strPadInspDir[nPadInfoCnt], strFileName);

				struInfoPad[nPadInfoCnt][nCamCnt]._malloc(0);

				CFileFind find;

				BOOL bFindFile = FALSE;

				bFindFile = find.FindFile(strPadInspPath); find.Close(); if (!bFindFile) { continue; }

				CMarkup xmlFile; if (!xmlFile.Load(strPadInspPath)) { continue; }

				CMarkup* xmlList = new CMarkup(xmlFile);

				xmlList->FindElem(_T("AreaData"));

				int nMaxCnt = _ttoi(xmlList->GetAttrib(_T("RoiCount")));

				xmlList->IntoElem();

				struInfoPad[nPadInfoCnt][nCamCnt]._malloc(nMaxCnt);

				for (int nRoiCnt = 0; nRoiCnt < nMaxCnt; nRoiCnt++)
				{
					if (xmlList->FindElem(_T("ROI_%d"), nRoiCnt + 1))
					{
						STRU_PAD_AREA* tPadArea = (STRU_PAD_AREA*)&struInfoPad[nPadInfoCnt][nCamCnt].tPadInfo[nRoiCnt];

						//Area Coord Load
						cv::Rect rect = cv::Rect(Point(_ttoi(xmlList->GetAttrib(_T("START_X"))), _ttoi(xmlList->GetAttrib(_T("START_Y"))))
							, Point(_ttoi(xmlList->GetAttrib(_T("END_X"))), _ttoi(xmlList->GetAttrib(_T("END_Y")))));
						tPadArea->cvRect = rect;

						//Image Load
						int nAreaNo = _ttoi(xmlList->GetAttrib(_T("NO")));
						strImgName.Format(_T("%s_img_%02d_CAM%02d.bmp"), strPadInspFile[nPadInfoCnt], nAreaNo, nCamCnt);
						strPadInspPath.Format(_T("%s\\%s\\%s"), strModelXMLPath, strPadInspDir[nPadInfoCnt], strImgName);
						char* pStr = NULL; pStr = CSTR2PCH(strPadInspPath);
						tPadArea->ipImg = cvLoadImage(pStr, IMREAD_UNCHANGED);
						//Polygon Coord Load
						CString str = xmlList->GetAttrib(_T("POINT_X")) + xmlList->GetAttrib(_T("POINT_Y"));

						if (str != "")
						{
							tPadArea->Point_malloc(1);
							tPadArea->cvPoint[0].x = _ttoi(xmlList->GetAttrib(_T("POINT_X")));
							tPadArea->cvPoint[0].y = _ttoi(xmlList->GetAttrib(_T("POINT_Y")));
							xmlList->IntoElem();
						}
						else
						{
							int nPntNum = _ttoi(xmlList->GetAttrib(_T("PolygonCount")));
							tPadArea->Point_malloc(nPntNum);
							xmlList->IntoElem();

							for (int nPntCnt = 0; nPntCnt < nPntNum; nPntCnt++)
							{
								if (xmlList->FindElem(_T("POINT_%d"), nPntCnt + 1))
								{
									tPadArea->cvPoint[nPntCnt].x = _ttoi(xmlList->GetAttrib(_T("POINT_X")));
									tPadArea->cvPoint[nPntCnt].y = _ttoi(xmlList->GetAttrib(_T("POINT_Y")));
								}
							}
						}
						xmlList->ResetMainPos();
					}
					xmlList->OutOfElem();
				}
				SAFE_DELETE(xmlList);
			}
		}
		SetPadAreaInfo(struInfoPad);

	}
	catch (Exception ex) {
		theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, TRUE, TRUE, _T("Failed Read Pad Inspect Parameter!!! %s"), ex.msg);
		return false;
	}

	timerReadParameter.End();
	theApp.WriteLog(eLOGTACT, eLOGLEVEL_DETAIL, FALSE, FALSE, _T("Read Pad Inspect Parameter tact time %.2f"), timerReadParameter.GetTime() / 1000.);

	return true;
}

BOOL CVSAlgorithmTaskApp::m_fnInitFunc()
{
	theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, TRUE, FALSE, _T("---------------Initialize Start!!-----------------"));

	if (!(InspPanel.InitVision()))		//初始化InspCam数据(生成inspThrd)
	{
		theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, TRUE, FALSE, _T("[Initialize] Fail (1)!!"));
		return FALSE;
	}

	//读取相同错误的重复Count
	m_fnReadRepeatDefectInfo();

	theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, TRUE, FALSE, _T("[Initialize] Init Complete!! Thread Create!!"));

	//选择要保存的驱动器-当前禁用
//SearchDriveDisk();

	return TRUE;
}

//添加日志记录
void CVSAlgorithmTaskApp::WriteLog(const ENUM_KIND_OF_LOG eLogID, const ENUM_LOG_LEVEL eLogLevel, const BOOL bSendGUITask, const BOOL bTraceList, TCHAR* pszLog, ...)
{
	//日志级别
	//setlocale();
	setlocale(LC_ALL, ""); //使用当前计算机首选项中的语言信息 hjf
	if (eLogLevel < ALG_LOG_LEVEL)
		return;

	//在日志记录的临时缓冲区中,参考参数的内容记录日志内容。
	va_list vaList;
	va_start(vaList, pszLog);
	TCHAR* cBuffer = NULL;
	TCHAR* cBufferTemp = NULL;		// for UI
	int len = 0;
	int nBufSize = 0;

	if (pszLog != NULL)
	{
		len = _vscwprintf(pszLog, vaList) + 1;
		nBufSize = len * sizeof(TCHAR);

		cBuffer = new TCHAR[nBufSize];
		memset(cBuffer, 0, nBufSize);

		if (cBuffer)
			vswprintf(cBuffer, pszLog, (va_list)vaList);
	}
	va_end(vaList);

	//将日志发送到GUI Task
	if (bSendGUITask)
		if (GetIPCState())
		{
			ST_SENDUI_COMMON logInfo;
			logInfo.eModuleType = EModuleType::ALGO;
			logInfo.eLogLevel = ELogLevel::INFO_;
			logInfo.nStationNum = theApp.m_Config.GetPCNum();
			memcpy(logInfo.byteMsgBuf, cBuffer, nBufSize);

			LogSendToUI::getInstance()->SendCommonLog(logInfo);
			//LogSendToUI::getInstance()->SendCommonLog(EModuleType::ALGO, ELogLevel::INFO_,
			//	theApp.m_Config.GetPCNum(),
			//	cBuffer);
			//m_AlgorithmTask->VS_Send_Log_To_UI(cBuffer, nBufSize);


		}
			


	//在Alg Task UI中显示Trace日志
	if (bTraceList)
	{
		if (m_pDlg != NULL)
		{
			cBufferTemp = new TCHAR[nBufSize];
			memset(cBufferTemp, 0, nBufSize);
			memcpy(cBufferTemp, cBuffer, nBufSize);
			::PostMessage(m_pDlg->m_hWnd, WM_PRINT_UI_LOG_MSG_UNICODE, nBufSize, (LPARAM)cBufferTemp);	// Print Log GUI
		}
	}

	switch (eLogID)
	{
	case eLOGCAM0:	case eLOGCAM1:	case eLOGCAM2:	case eLOGCAM3:
		m_pLogWriterCam[eLogID]->m_fnWriteLog(cBuffer);
		break;
	case eLOGPROC:
		m_pLogWriterProc->m_fnWriteLog(cBuffer);
		break;
	case eLOGTACT:
		m_pLogWriterTact->m_fnWriteLog(cBuffer);
		break;
	case eLOGCOMM:
		m_pLogWriterComm->m_fnWriteLog(cBuffer);
		break;
	default:
		break;
	}
	SAFE_DELETE(cBuffer);
}

CString CVSAlgorithmTaskApp::GetCurStepFileName(TCHAR* strFindFolder, TCHAR* strFindFileName)
{
	CFileFind finder;

	CString strWildcard = _T("");
	CString strRet = _T("");

	strWildcard.Format(_T("%s\\*%s.*"), strFindFolder, strFindFileName);

	BOOL bWorking = finder.FindFile(strWildcard);

	if (bWorking)
	{
		finder.FindNextFile();

		if (finder.IsDots())
			return _T("");

		strRet = finder.GetFilePath();
	}
	else
	{
		strRet = strWildcard;
	}
	finder.Close();

	return strRet;
}

//返回值:在当前设置的Ratio值中应更改的值(-2~2至+2)
int CVSAlgorithmTaskApp::CheckImageRatio(UINT nCurRatio, int nDstWidth, int nDstHeight, int nSrcWidth, int nSrcHeight)
{
	int nRet = 0;

	for (int nSeqMode = 2; nSeqMode >= 0; nSeqMode--)	// 0 : Non / 1 : 4-Shot / 2 : 9-Shot
	{
		if (nDstWidth >= nSrcWidth * (nSeqMode + 1) && nDstHeight >= nSrcHeight * (nSeqMode + 1))	//确认缩放比例
		{
			//如果当前设置的方法的图像百分比与实际图像缓冲区的百分比不同
			if (nCurRatio != (nSeqMode + 1))
			{
				nRet = (nSeqMode + 1) - nCurRatio;
			}
			break;
		}
	}
	return nRet;
}

UINT CVSAlgorithmTaskApp::ThreadDiskCheck(LPVOID pParam)
{
	CVSAlgorithmTaskApp* pThis = (CVSAlgorithmTaskApp*)pParam;

	CString strDrive, strTemp;
	CString strUseDrive;
	int		nSleepCnt = 0;
	const int nSleepInterval = 50;

	do
	{
		//为了在Task结束时立即结束...
		if (nSleepCnt * nSleepInterval >= DRIVE_CHECK_INTERVAL || nSleepCnt == 0)
		{
			nSleepCnt = 0;
			pThis->m_pDiskInfo->RenewalDiskInfo();
			strDrive = theApp.m_Config.GetINIDrive();
			strTemp = strDrive.Left(1) + _T(":\\");
			strUseDrive = pThis->m_pDiskInfo->GetAvailableDrivePath(strTemp);
			if (strUseDrive != strTemp)
			{
				strTemp = strUseDrive.Left(1) + _T("_Drive") + _T("\\");
				theApp.m_Config.OpenFile(INIT_FILE_PATH);
				theApp.m_Config.Write(_T("Diskinformation"), _T("Last Used Drive"), strTemp);
				theApp.m_Config.SetINIDrive(strTemp);
			}
		}

		Sleep(10);
		nSleepCnt++;
	} while (pThis->m_bExecDisk);

	SetEvent(pThis->m_hEventDiskThreadAlive);
	printf("Disk Thread End \n");

	return 0;
}

//检查算法线程是否超时，若超时则退出 释放资源 复位标志位 hjf
UINT CVSAlgorithmTaskApp::AlgoThreadTimeOutCheck(LPVOID pParam)
{
	CVSAlgorithmTaskApp* pThis = (CVSAlgorithmTaskApp*)pParam;

	do
	{
		//开始检查算法线程超时
		//pThis->InspPanel.CheckInsThread();

		//1000ms检查一次
		Sleep(1000);
	} while (pThis->m_bExecAlgoThrBusy);
	SetEvent(pThis->m_hEventAlgoThreadTimeOutAlive);
	return 0;
}

void CVSAlgorithmTaskApp::CheckDrive()
{
	CString strDrive, strTemp;
	CString strUseDrive;

	m_pDiskInfo->RenewalDiskInfo();
	strDrive = theApp.m_Config.GetINIDrive();
	strTemp = strDrive.Left(1) + _T(":\\");
	strUseDrive = m_pDiskInfo->GetAvailableDrivePath(strTemp);
	if (strUseDrive != strTemp)
	{
		strTemp = strUseDrive.Left(1) + _T("_Drive") + _T("\\");
		//theApp.m_Config.SetCurrentDrive( strTemp );
		theApp.m_Config.OpenFile(INIT_FILE_PATH);
		theApp.m_Config.Write(_T("Diskinformation"), _T("Last Used Drive"), strTemp);
		theApp.m_Config.SetINIDrive(strTemp);
	}
}

bool CVSAlgorithmTaskApp::ReadJudgeParameter(TCHAR* strModelPath)
{
	///////////////////////////////////////////////////////////////////////////////////////////////
	CString strMsg = _T("");
	CFileFind find;
	BOOL bFindFile = FALSE;
	CString strDefItemListXMLPath;
	//添加对当前解决方案运行驱动器的判断
	strDefItemListXMLPath.Format(_T("%s:\\IMTC\\Text\\DEFITEM_LIST.xml"), theApp.m_Config.GETCmdDRV());
	//strDefItemListXMLPath.Format(_T("%s\\ReportFiltering.rule"), strModelPath);
	bFindFile = find.FindFile(strDefItemListXMLPath);
	find.Close();

	if (!bFindFile)
	{
		strMsg.Format(_T("Not found defect item list xml file. (%s)"), strDefItemListXMLPath);
		AfxMessageBox(strMsg);
		return false;
	}

	//加载XML文件
	CMarkup xmlDefectItem;
	if (!xmlDefectItem.Load(strDefItemListXMLPath))
	{
		strMsg.Format(_T("Model xml load fail. (%s)"), strDefItemListXMLPath);
		AfxMessageBox(strMsg);
		return false;
	}

	xmlDefectItem.FindElem();		// DEF_ITEM
	xmlDefectItem.IntoElem();		// inside DEF_ITEM

	CString strDefSysName = _T(""), strDefCode = _T("");
	CMarkup* xmlDefItemList = new CMarkup(xmlDefectItem);
	stDefClassification* stDefClass = new stDefClassification[MAX_MEM_SIZE_E_DEFECT_NAME_COUNT];

	for (int nDefItemIndex = 0; nDefItemIndex < MAX_MEM_SIZE_E_DEFECT_NAME_COUNT; nDefItemIndex++)
	{
		if (xmlDefItemList->FindElem(_T("DefType_%d"), nDefItemIndex))
		{
			strDefSysName = xmlDefItemList->GetAttrib(_T("SysName"));
			//memcpy(stDefClass[nDefItemIndex].strDefectName, strDefSysName.GetBuffer(0), sizeof(stDefClass[nDefItemIndex].strDefectName) - sizeof(TCHAR));
			//_tcscat(stDefClass[nDefItemIndex].strDefectName, _T("\0"));
			COPY_CSTR2TCH(stDefClass[nDefItemIndex].strDefectName, strDefSysName, sizeof(stDefClass[nDefItemIndex].strDefectName));
			strDefCode = xmlDefItemList->GetAttrib(_T("DefCode"));
			// 			memcpy(stDefClass[nDefItemIndex].strDefectCode, strDefCode.GetBuffer(0), sizeof(stDefClass[nDefItemIndex].strDefectCode) - sizeof(TCHAR));
			// 			_tcscat(stDefClass[nDefItemIndex].strDefectCode, _T("\0"));
			COPY_CSTR2TCH(stDefClass[nDefItemIndex].strDefectCode, strDefCode, sizeof(stDefClass[nDefItemIndex].strDefectCode));
			xmlDefItemList->ResetMainPos();
		}
	}
	SetDefectClassify(stDefClass);
	SAFE_DELETE_ARR(stDefClass);
	SAFE_DELETE(xmlDefItemList);

	/////////////////////////////////////////////////面板判定标准Load
	CString strModelPanelJudgePath = _T("");
	//更改RMS对应的模型文件路径
//strModelPanelJudgePath.Format(_T("%s\\%s\\PanelJudge.rule"), MODEL_FILE_PATH, strModelID);
	strModelPanelJudgePath.Format(_T("%s\\PanelJudge.rule"), strModelPath);

	BOOL bRet = FALSE;
	CString strLine = _T(""), strTemp = _T(""), strItem = _T("");
	CStdioFile fileReader;

	std::vector<stPanelJudgePriority> vPanelJudgeTemp;
	vPanelJudgeTemp.clear();

	try {
		bRet = fileReader.Open(strModelPanelJudgePath, CFile::modeRead | CFile::typeText);
		if (bRet)
		{
			int nCount = 0;
			fileReader.ReadString(strLine);		// 丢弃Header信息
			while (fileReader.ReadString(strLine))
			{
				stPanelJudgePriority stPanelJudge;
				memset(&stPanelJudge, 0, sizeof(stPanelJudgePriority));

				AfxExtractSubString(strTemp, strLine, 0, ',');
				COPY_CSTR2TCH(stPanelJudge.strGrade, strTemp, sizeof(stPanelJudge.strGrade));

				for (nCount = 1; nCount <= E_PANEL_DEFECT_TREND_COUNT; nCount++)
				{
					int nIndex = 0;
					AfxExtractSubString(strTemp, strLine, nCount, ',');

					AfxExtractSubString(strItem, strTemp, nIndex++, '/');
					m_fnGetJudgeInfo(stPanelJudge.stJudgeInfo, strItem, nCount);
					AfxExtractSubString(strItem, strTemp, nIndex++, '/');
					m_fnGetJudgeInfo(stPanelJudge.stFilterInfo, strItem, nCount);

				}

				for (nCount = E_PANEL_DEFECT_TREND_COUNT + 1; nCount <= E_PANEL_DEFECT_TREND_COUNT; nCount++)
				{
					AfxExtractSubString(strTemp, strLine, nCount, ',');
					if (strTemp != _T(""))
						m_fnGetJudgeInfo(stPanelJudge.stJudgeInfo, strTemp, nCount);
					else
						m_fnGetJudgeInfo(stPanelJudge.stJudgeInfo, _T("0+"), nCount);
				}

				vPanelJudgeTemp.push_back(stPanelJudge);
			}
			fileReader.Close();
		}
	}
	catch (...) {
		bRet = FALSE;
	}
	theApp.SetPanelJudgeInfo(vPanelJudgeTemp);

	if (!bRet)
	{
		//日志输出
		theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, TRUE, TRUE, _T("Failed Read Judge Parameter File!!!"));
		if (!fileReader)
			fileReader.Close();
		return false;
	}

	return true;
}

bool CVSAlgorithmTaskApp::ReadUserDefinedFilter(TCHAR* strModelPath)
{
	CString strModelUserFilterPath = _T("");
	//	strModelReportFilterPath.Format(_T("%s\\%s\\ReportFiltering.rule"), MODEL_FILE_PATH, strModelID);
	strModelUserFilterPath.Format(_T("%s\\UserDefinedFilter.rule"), strModelPath);
	BOOL bRet = FALSE;
	CString strLine = _T(""), strTemp = _T(""), strItem = _T("");
	CStdioFile fileReader;

	std::vector<stUserDefinedFilter> vUserDefinedFilter;
	vUserDefinedFilter.clear();

	try {
		bRet = fileReader.Open(strModelUserFilterPath, CFile::modeRead | CFile::typeText);
		if (bRet)
		{

			fileReader.ReadString(strLine);
			while (fileReader.ReadString(strLine))
			{
				int nCount = 0;
				stUserDefinedFilter stUserFilter;
				memset(&stUserFilter, 0, sizeof(stUserDefinedFilter));
				stUserFilter.bUse = TRUE;
				AfxExtractSubString(strTemp, strLine, nCount++, ',');
				AfxExtractSubString(strTemp, strLine, nCount++, ',');
				AfxExtractSubString(strTemp, strLine, nCount++, ',');
				_tcscpy(stUserFilter.strGrade, strTemp);
				AfxExtractSubString(strTemp, strLine, nCount++, ',');
				AfxExtractSubString(strTemp, strLine, nCount++, ',');
				stUserFilter.nFilterItemsCount = _ttoi(strTemp);
				AfxExtractSubString(strTemp, strLine, nCount++, ',');
				AfxExtractSubString(strItem, strTemp, 0, '+');
				for (int i = 0; i < stUserFilter.nFilterItemsCount; i++)
				{
					AfxExtractSubString(strItem, strTemp, i, '+');
					stUserFilter.nFilterItems[i] = _ttoi(strItem);
				}
				AfxExtractSubString(strTemp, strLine, nCount++, ',');
				m_fnGetJudgeInfo(&stUserFilter.stFilterInfo, strTemp, 1);

				vUserDefinedFilter.push_back(stUserFilter);
			}
			fileReader.Close();
		}
	}
	catch (...) {
		bRet = FALSE;
	}
	theApp.SetUserDefinedFilter(vUserDefinedFilter);
	std::vector<stUserDefinedFilter> dd;
	dd = theApp.GetUserDefinedFilter();
	if (!bRet)
	{
		//日志输出
		theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, TRUE, TRUE, _T("Failed Read Judge Parameter File!!!"));
		if (!fileReader)
			fileReader.Close();
		return false;
	}
	return true;
}

bool CVSAlgorithmTaskApp::ReadDefectClassify(TCHAR* strModelPath)
{
	//基于父报告过滤的Load
	CString strDefectClassifyFilePath = _T("");
	//strDefectRankFilePath.Format(_T("%s\\%s\\DefectRank.rule"), MODEL_FILE_PATH, strModelID);
	////临时 修复APPLY路径
	//strDefectClassifyFilePath.Format(_T("E:\\IMTC\\DATA\\MODEL\\APPLY\\InspRecipe\\17-08-01_01-01-01.1\\DefectClassify.rule"));
	strDefectClassifyFilePath.Format(_T("%s\\DefectClassify.rule"), strModelPath);
	BOOL bRet = FALSE;
	CString strLine = _T(""), strTemp = _T("");
	CStdioFile fileReader;
	int DefectRank[E_DEFECT_JUDGEMENT_COUNT] = { 0, };
	int DefectGroup[E_DEFECT_JUDGEMENT_COUNT] = { 0, };

	try {
		bRet = fileReader.Open(strDefectClassifyFilePath, CFile::modeRead | CFile::typeText);
		if (bRet)
		{
			int nCount = 0;
			fileReader.ReadString(strLine);		// 丢弃Header信息

			while (fileReader.ReadString(strLine))
			{
				AfxExtractSubString(strTemp, strLine, 0, ',');

				if (strTemp.CompareNoCase(_T("Rank")) == 0)
				{
					//Defect Rank设置
					for (nCount = 0; nCount < E_DEFECT_JUDGEMENT_COUNT; nCount++)
					{
						AfxExtractSubString(strTemp, strLine, nCount + 1, ',');
						DefectRank[nCount] = _ttoi(strTemp);
					}
				}
				else if (strTemp.CompareNoCase(_T("Group")) == 0)
				{
					//Defect Group设置					
					for (nCount = 0; nCount < E_DEFECT_JUDGEMENT_COUNT; nCount++)
					{
						AfxExtractSubString(strTemp, strLine, nCount + 1, ',');
						DefectGroup[nCount] = _ttoi(strTemp);
					}
				}
			}

			fileReader.Close();
		}
	}
	catch (...) {
		theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, TRUE, TRUE, _T("Failed Read Defect Classify Rule !!! Set Defect Rank / Group Default"));
		for (int i = 0; i < E_DEFECT_JUDGEMENT_COUNT; i++)
			DefectRank[i] = 1;
	}
	theApp.SetDefectRank(DefectRank);
	theApp.SetDefectGroup(DefectGroup);

	if (!bRet)
	{
		if (!fileReader)
			fileReader.Close();
		return false;
	}

	return true;

}

// 读取缺陷合并规则表
bool CVSAlgorithmTaskApp::ReadMergeRules(TCHAR* strModelPath)
{
	std::wstring_convert<std::codecvt_utf8_utf16<TCHAR>, TCHAR> converter;
	std::wstring wstr = strModelPath;
	std::string parentPath = converter.to_bytes(wstr);

	// 默认只启用前3个merge recipe
	int LOGIC_CNT = 3;
	m_vMergeLogics.resize(LOGIC_CNT);
	for (int i = 0; i < LOGIC_CNT; i++) {
		std::string filepath = parentPath + std::string("/merge_recipe") + std::to_string(i) + std::string(".csv");
		if (!ReadSingleMergeRule(filepath, m_vMergeLogics[i])) {
			return false;
		}
	}

	return true;
}

bool CVSAlgorithmTaskApp::ReadSingleMergeRule(std::string fileName, std::vector<std::vector<std::string>>& vMergeLogic)
{
	std::vector<std::vector<std::string>> data;

	std::ifstream file(fileName);
	if (!file.is_open()) {
		return false;
	}

	vMergeLogic.clear();
	std::string line;
	while (std::getline(file, line)) {
		std::vector<std::string> row;
		std::stringstream ss(line);
		std::string cell;

		while (std::getline(ss, cell, ',')) {
			row.push_back(cell);
		}

		vMergeLogic.push_back(row);
	}

	file.close();
	return true;
}

bool CVSAlgorithmTaskApp::ReadPolMarkTemplates(TCHAR* strModelPath)
{
	std::wstring_convert<std::codecvt_utf8_utf16<TCHAR>, TCHAR> converter;
	std::wstring wstr = strModelPath;
	std::string parentPath = converter.to_bytes(wstr);
	
	theApp.m_polNumTemplates.clear();
	theApp.m_polSignTemplates.clear();

	std::string templateDir = parentPath + std::string("/PolMark");
	std::vector<cv::String> files;
	cv::glob(templateDir, files);
	for (const auto& file : files) {
		cv::Mat templateImg = cv::imread(file, cv::IMREAD_GRAYSCALE);
		// 图像名
		std::size_t found_slash = file.find_last_of("/\\");
		std::string filename = file.substr(found_slash + 1);
		// 去掉后缀
		std::size_t found_dot = filename.find_last_of('.');
		std::string name = filename.substr(0, found_dot);

		if (name.find("num_") != std::string::npos) {
			theApp.m_polNumTemplates[name] = templateImg;
		}
		else if (name.find("sign_") != std::string::npos) {
			theApp.m_polSignTemplates[name] = templateImg;
		}
	}
	return true;
}

//分区参数加载 hjf
bool CVSAlgorithmTaskApp::ReadPartitionBlockParameter(TCHAR* strModelPath) {
	/*


	*/
	//临时设置成3*3的块大小 hjf
	SetBlockCountX(1);
	SetBlockCountY(1);
	theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, TRUE, TRUE, _T("Read Partition Block Parameter Complete!"));
	return true;
}

void CVSAlgorithmTaskApp::m_fnGetJudgeInfo(stPanelJudgeInfo* pStJudgeInfo, CString strVal, int nCount)
{
	if (strVal.GetLength() == 0)		strVal = _T("0-");		// Default

	//统一Algorithm Parameter和Format以实现GUI集成
	if (strVal.Right(1).CompareNoCase(_T("=")) == 0)		pStJudgeInfo[nCount - 1].nSign = 0;	// 未使用
	else if (strVal.Right(1).CompareNoCase(_T("!")) == 0)		pStJudgeInfo[nCount - 1].nSign = 1;
	//else if (strVal.Right(1).CompareNoCase(_T(">")) == 0)		pStJudgeInfo[nCount-1].nSign = 2;	// 未使用
	//else if (strVal.Right(1).CompareNoCase(_T("<")) == 0)		pStJudgeInfo[nCount-1].nSign = 3;	// 未使用
	else if (strVal.Right(1).CompareNoCase(_T("+")) == 0)		pStJudgeInfo[nCount - 1].nSign = 4;
	else if (strVal.Right(1).CompareNoCase(_T("-")) == 0)		pStJudgeInfo[nCount - 1].nSign = 5;
	else
	{
		strVal += _T("-");
		pStJudgeInfo[nCount - 1].nSign = 5;
	}
	pStJudgeInfo[nCount - 1].nRefVal = _ttoi(strVal.Left(strVal.GetLength() - 1));
}

// 2018.10.01 MDJ File Read
void CVSAlgorithmTaskApp::ReadRepeatDefectInfo()
{
	m_fnReadRepeatDefectInfo();
}

bool CVSAlgorithmTaskApp::m_fnReadRepeatDefectInfo()
{
	m_listRepeatDefInfo[ePIXEL].clear();
	m_listRepeatDefInfo[eWORK].clear();

	m_fnReadRepeatFile(REPEAT_DEFECT_PIXEL_INFO_PATH, &m_listRepeatDefInfo[ePIXEL]);
	m_fnReadRepeatFile(REPEAT_DEFECT_WORK_INFO_PATH, &m_listRepeatDefInfo[eWORK]);

	return true;

}

bool CVSAlgorithmTaskApp::WriteResultFile(CString strPanelID, CString strFilePath, CString strFileName, CString strColumn, TCHAR* strResult)
{
	//检查路径
	DWORD result;
	if (((result = GetFileAttributes(strFilePath)) == -1) || !(result & FILE_ATTRIBUTE_DIRECTORY)) {
		CreateDirectory(strFilePath, NULL);
	}

	CString strResultData;
	SYSTEMTIME	time;
	GetLocalTime(&time);

	CString strDate;
	CString strTime;
	strTime.Format(_T("%02u:%02u:%02u"), time.wHour, time.wMinute, time.wSecond);
	strDate.Format(_T("%04u%02u%02u_"), time.wYear, time.wMonth, time.wDay);

	strDate.Append(strFileName);

	CString strFullPath;
	strFullPath.Format(_T("%s\\%s"), strFilePath, strDate);

	CStdioFile stdFile;

	char cSavePath[256];
	memset(cSavePath, NULL, 256);
	WideCharToMultiByte(CP_ACP, NULL, strFullPath, -1, cSavePath, 256, NULL, NULL);

	int nFileCheck = access(cSavePath, 0);

	if (!stdFile.Open(strFullPath, CFile::modeWrite | CFile::modeCreate | CFile::modeNoTruncate | CFile::typeText | CFile::shareDenyWrite, NULL))
	{
		return false;
	}

	stdFile.SeekToEnd(); //移动到文件指针的末尾

	strResultData.Format(_T("\n%s,%s"), strTime, strPanelID);

	CString strValue;
	strValue.Format(_T(",%s"), strResult);
	strResultData.Append(strValue);

	CString strStandardColumn;
	strStandardColumn.Format(_T("TIME,"));
	strStandardColumn.Append(strColumn);

	if (nFileCheck == 0)			//如果有文件
	{
		stdFile.WriteString(strResultData);
	}
	else						//如果没有文件
		stdFile.WriteString(strStandardColumn);

	stdFile.Close();

	return true;
}

BOOL CVSAlgorithmTaskApp::m_fnReadRepeatFile(CString strFilePath, std::list<RepeatDefectInfo>* pList)
{
	CString strLine = _T(""), strTemp = _T("");
	CStdioFile fileReader;
	RepeatDefectInfo stRepeatDefInfo;
	BOOL bRet = FALSE;

	try {
		bRet = fileReader.Open(strFilePath, CFile::modeRead | CFile::typeText);
		if (bRet)
		{
			int nCount = 0;
			fileReader.ReadString(strLine);		// 丢弃Header信息

			while (fileReader.ReadString(strLine))
			{
				/// Defect Code
				AfxExtractSubString(strTemp, strLine, 0, ',');
				stRepeatDefInfo.eDefType = (ENUM_DEFECT_JUDGEMENT)(_ttoi(strTemp));

				///CCD不良信息
				AfxExtractSubString(strTemp, strLine, 1, ',');
				stRepeatDefInfo.ptCenterPos.x = _ttoi(strTemp);

				AfxExtractSubString(strTemp, strLine, 2, ',');
				stRepeatDefInfo.ptCenterPos.y = _ttoi(strTemp);

				AfxExtractSubString(strTemp, strLine, 3, ',');
				stRepeatDefInfo.nRepeatCount = _ttoi(strTemp);

				pList->push_back(stRepeatDefInfo);
			}

			fileReader.Close();
		}
	}
	catch (...) {
		bRet = FALSE;
		pList->clear();
		theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, FALSE, FALSE, _T("Failed Read Repeat Defect Info !!!"));
	}

	if (!bRet)		//如果文件不存在或无法打开
	{
		theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, FALSE, FALSE, _T("Not Found Repeat Defect Info !!! (%s)"), strFilePath);
		if (!fileReader)
			fileReader.Close();
		return false;
	}
	return bRet;
}

BOOL CVSAlgorithmTaskApp::GetMergeToolUse()
{
	m_nInspStep = GetPrivateProfileInt(_T("MergeTool"), _T("USE_MERGE_TOOL"), 1, INIT_FILE_PATH) ? true : false;
	return m_nInspStep;
};
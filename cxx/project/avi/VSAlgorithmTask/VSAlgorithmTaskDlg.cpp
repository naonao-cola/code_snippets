

//

#include "stdafx.h"
#include "VSAlgorithmTask.h"
#include "VSAlgorithmTaskDlg.h"
#include "afxdialogex.h"
#include "Define.h"
#include "version.h"

#include "DllInterface.h"	// for test
#include "TESTAlgorithm.h"
#ifdef _DEBUG
#define new DEBUG_NEW
#endif

CVSAlgorithmTaskDlg::CVSAlgorithmTaskDlg(CWnd* pParent )
	: CDialogEx(CVSAlgorithmTaskDlg::IDD, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CVSAlgorithmTaskDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_LIST_LOG, m_ctrlListLog);
}

BEGIN_MESSAGE_MAP(CVSAlgorithmTaskDlg, CDialogEx)
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_WM_SYSCOMMAND()
	ON_WM_DESTROY()
	ON_WM_TIMER()
	ON_MESSAGE(WM_TRAYICON_MSG,	TrayIconMessage)
	ON_WM_DRAWITEM()
	ON_BN_CLICKED(IDC_BTN_TEST, &CVSAlgorithmTaskDlg::OnBnClickedBtnTest)
	ON_MESSAGE(WM_PRINT_UI_LOG_MSG_UNICODE,		m_fnPrintUiMsgUnicode)
	ON_MESSAGE(WM_PRINT_UI_LOG_MSG_MULTI_BYTE,	m_fnPrintUiMsgMultiByte)
	ON_BN_CLICKED(IDC_BTN_TEST2, &CVSAlgorithmTaskDlg::OnBnClickedBtnTest2)
	ON_BN_CLICKED(IDC_BTN_CNT_CLEAR, &CVSAlgorithmTaskDlg::OnBnClickedBtnCntClear)
	ON_BN_CLICKED(IDC_BUTTON2, &CVSAlgorithmTaskDlg::OnBnClickedButton2)
END_MESSAGE_MAP()

BOOL CVSAlgorithmTaskDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

		//设置此对话框的图标。如果应用程序的主窗口不是对话框:
		//框架将自动执行此操作。
		SetIcon(m_hIcon, TRUE);			//设置大图标。
		SetIcon(m_hIcon, FALSE);		//设置小图标。

		//TODO:在此添加其他初始化操作。
		//启动Auto Mode最小化
	ShowWindow(SW_MINIMIZE);

		//Icon更改
	m_fnTrayIconRefresh();
	m_bTrayStatus = FALSE;
	TraySetting();	

		//Task状态更新Timer
	SetTimer(eTIMER_UpdateTaskStatus, 1000, NULL);
	CString strVersion;
	strVersion.Format(_T("VSAlgorithmTask"));

	SetWindowText(strVersion);

		return TRUE;  //如果未在控件中设置焦点,则返回TRUE。
}

//在对话框中添加最小化按钮时绘制图标的步骤
//需要以下代码对于使用文档/视图模型的MFC应用程序:
//框架会自动执行此操作。

void CVSAlgorithmTaskDlg::OnPaint()
{
	if (IsIconic())
	{
				CPaintDC dc(this); //用于绘图的设备上下文。

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

				//在客户端矩形中居中对齐图标。
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

				//绘制图标。
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

//为了在用户关闭最小化窗口时显示光标,请在系统中
//调用此函数
HCURSOR CVSAlgorithmTaskDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}

void CVSAlgorithmTaskDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
		//TODO:在此添加消息处理器代码并/或调用默认值。
	BOOL bHide = FALSE;
	BOOL bClose = FALSE;

	if(nID == SC_CLOSE	)
	{
		bHide = TRUE;

		if((GetKeyState(VK_SHIFT)<0)&&(AfxMessageBox(_T("Exit?"),MB_YESNO|MB_ICONQUESTION)==IDYES))
		{
			bClose = TRUE;
		}

//		bClose = TRUE;

	}
	else if (nID == SC_MINIMIZE)
	{
		ShowWindow(SW_HIDE);
	}

	if (bClose)
	{
		if (m_bTrayStatus)
		{
			NOTIFYICONDATA nid;
			nid.cbSize = sizeof(nid);
			nid.hWnd = m_hWnd;
			nid.uID = IDR_MAINFRAME;
			// Delete Icon in Status Bar
			Shell_NotifyIcon(NIM_DELETE, &nid);
		}
		AfxGetApp()->LoadIcon(IDI_ICON1);	

		CDialog::OnSysCommand(nID, lParam);
	}
	else if (bHide)
	{
		ShowWindow(SW_HIDE);
		return;

	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}

}

//添加托盘图标
void CVSAlgorithmTaskDlg::TraySetting(void)
{
	NOTIFYICONDATA nid;

	nid.cbSize = sizeof(nid);
	nid.hWnd = m_hWnd;
	nid.uID = IDR_MAINFRAME;
	nid.uFlags = NIF_MESSAGE | NIF_ICON | NIF_TIP;
	nid.uCallbackMessage = WM_TRAYICON_MSG;
	nid.hIcon = AfxGetApp()->LoadIcon(IDI_ICON1);

	TCHAR strTitle[256];
	GetWindowText(strTitle, sizeof(strTitle));
	lstrcpy(nid.szTip, strTitle);
	Shell_NotifyIcon(NIM_ADD, &nid);
	SendMessage(WM_SETICON, (WPARAM)TRUE, (LPARAM)nid.hIcon);
	SendMessage(WM_SETICON, (WPARAM)FALSE, (LPARAM)nid.hIcon);
	m_bTrayStatus = TRUE;
}

BOOL CVSAlgorithmTaskDlg::PreTranslateMessage(MSG* pMsg)
{
		//TODO:在此处添加特殊代码和/或调用基类。

	if (pMsg->message == WM_COMMAND && pMsg->wParam == 2)
	{
		NOTIFYICONDATA nid;
		nid.cbSize = sizeof(nid);
		nid.hWnd = m_hWnd;
		nid.uID = IDR_MAINFRAME;
		// Delete Icon in Status Bar
		Shell_NotifyIcon(NIM_DELETE, &nid);

		AfxGetApp()->LoadIcon(IDI_ICON1);
	}

	if(pMsg->wParam == VK_RETURN || pMsg->wParam == VK_ESCAPE)
		return TRUE;

	return CDialogEx::PreTranslateMessage(pMsg);

}

//添加托盘图标
LRESULT CVSAlgorithmTaskDlg::TrayIconMessage(WPARAM wParam, LPARAM lParam)
{
	// Tray Icon Click -> Dialog pop up
	if (lParam == WM_LBUTTONDBLCLK)
	{
		ShowWindow(SW_SHOW);
	}
	return 0L;
}

void CVSAlgorithmTaskDlg::OnDestroy()
{
	CDialogEx::OnDestroy();

		//TODO:在此添加消息处理器代码。
		//禁用资源
	NOTIFYICONDATA nid;
	nid.cbSize = sizeof(nid);
	nid.hWnd = m_hWnd;
	nid.uID = IDI_ICON1;

	Shell_NotifyIcon(NIM_DELETE,&nid);
}

void CVSAlgorithmTaskDlg::OnTimer(UINT_PTR nIDEvent)
{
		//TODO:在此添加消息处理器代码并/或调用默认值。
	HWND hButton = NULL;

	switch (nIDEvent)
	{
	case eTIMER_UpdateTaskStatus:
		if (theApp.GetIPCState())
			theApp.m_AlgorithmTask->SetParentWnd(m_hWnd);
		hButton	= ::GetDlgItem(m_hWnd, IDC_BTN_IPC_CONNECT);
		::InvalidateRect(hButton, NULL, FALSE);
		break;
	}	

	CDialogEx::OnTimer(nIDEvent);
}

void CVSAlgorithmTaskDlg::OnDrawItem(int nIDCtl, LPDRAWITEMSTRUCT lpDrawItemStruct)
{
		//TODO:在此添加消息处理器代码并/或调用默认值。
		//nIDCtl=IDC值
	switch(nIDCtl)
	{
		case IDC_BTN_IPC_CONNECT:
		{
			if (theApp.GetIPCState())
			{
				m_fnChangeBtnColor(lpDrawItemStruct, LIGHT_GREEN, DARK_GREEN);
			}
			else
			{
				m_fnChangeBtnColor(lpDrawItemStruct, LIGHT_RED, DARK_RED);
			}
		}
		break;
	}

	CDialogEx::OnDrawItem(nIDCtl, lpDrawItemStruct);
}

void CVSAlgorithmTaskDlg::m_fnChangeBtnColor(LPDRAWITEMSTRUCT lpDrawItemStruct, COLORREF colorBtn, COLORREF colorText)
{
	CDC dc;
	RECT rect;
		dc.Attach(lpDrawItemStruct->hDC);					//获取按钮的dc
		rect = lpDrawItemStruct->rcItem;					//获取按钮区域
		dc.Draw3dRect(&rect,WHITE,BLACK);					//绘制按钮的轮廓

		dc.FillSolidRect(&rect, colorBtn);					//按钮颜色
		dc.SetBkColor(colorBtn);							//text的背景颜色
	dc.SetTextColor(colorText);							//texttort

		UINT state = lpDrawItemStruct->itemState;			//获取按钮状态
	if((state &ODS_SELECTED))
	{
		dc.DrawEdge(&rect,EDGE_SUNKEN,BF_RECT);
	}
	else
	{
		dc.DrawEdge(&rect,EDGE_RAISED,BF_RECT);
	}

		TCHAR buffer[MAX_PATH];											//获取按钮文本的临时缓冲区
		ZeroMemory(buffer,MAX_PATH);									//初始化缓冲区
		::GetWindowText(lpDrawItemStruct->hwndItem,buffer,MAX_PATH);	//获取按钮的text
		dc.DrawText(buffer,&rect,DT_CENTER|DT_VCENTER|DT_SINGLELINE);	//插入按钮的text
		dc.Detach();													//松开按钮的dc
}

void ImageSave(cv::Mat& MatSrcBuffer, TCHAR* strPath, ...);

void CVSAlgorithmTaskDlg::OnBnClickedBtnTest()
{
		//TODO:在此添加控制通知处理器代码。
	double nRatio = *theApp.GetAlignParameter(0) + E_PARA_SVI_CROP_RATIO;
	int nRet = 0;
	byte  byteParam[1000]		= {0,};
	byte* pSendParam			= byteParam;

	BYTE* pByteOutput[4] = {NULL, };
	for (int i=0; i<4; i++)
		pByteOutput[i] = new BYTE[6576 * 4384];

	cv::Mat MatPSImage = cv::imread("E:\\IMTC\\Backup\\00_DUST_CAM00.bmp", IMREAD_UNCHANGED);
	ImageSave(MatPSImage, _T("E:\\IMTC\\PSTEST\\PS_IMAGE.bmp"));

	BYTE* pByteInput = MatPSImage.data;

	int cnt1 = 0, cnt2 = 0, cnt3 = 0, cnt4 = 0;

	for(UINT32 j=0;j<4384*2;j++)
	{
		int p = 0;
		for(UINT32 i=0; i<6576*2; i++)
		{
			if		(i%2 == 0 && j%2 == 0)
				pByteOutput[1][cnt1++] = pByteInput[j*6576*2+i];
			else if (i%2 == 1 && j%2 == 0)
				pByteOutput[0][cnt2++] = pByteInput[j*6576*2+i];
			else if (i%2 == 0 && j%2 == 1)
				pByteOutput[3][cnt3++] = pByteInput[j*6576*2+i];
			else if (i%2 == 1 && j%2 == 1)
				pByteOutput[2][cnt4++] = pByteInput[j*6576*2+i];
		}
	}

	cv::Mat MatOrg[4];
	for (int i=0; i<4; i++)
	{
		MatOrg[i] = cv::Mat(4384, 6576, CV_8UC1, pByteOutput[i]);
		ImageSave(MatOrg[i], _T("E:\\IMTC\\PSTEST\\%d.bmp"), i);
	}	

	for (int i=0; i<4; i++)
		SAFE_DELETE_ARR(pByteOutput[i]);

//旋转测试
// 	for (int nImageNo=0; nImageNo<10; nImageNo++)

//		//文件名为Alg。更改为在Task中确定-多摄像头响应

// 		char* pStr = NULL;

	return;

// 	CString strTemp = _T("");

	theApp.ReadAlgorithmParameter(_T("bbbb"));
	return;
}

int CVSAlgorithmTaskDlg::m_fnTrayIconRefresh()
{
	try
	{
		HWND  hWnd;
		CRect re;
		DWORD dwWidth, dwHeight, x, y;

		// find a handle of a tray
		hWnd = ::FindWindow( _T("Shell_TrayWnd"), NULL );

		if( hWnd != NULL )
			hWnd = ::FindWindowEx( hWnd, 0, _T("TrayNotifyWnd"), NULL );

		if( hWnd != NULL )
			hWnd = ::FindWindowEx( hWnd, 0, _T("SysPager"), NULL );

		if( hWnd != NULL )
			hWnd = ::FindWindowEx( hWnd, 0, _T("ToolbarWindow32"), NULL );
		// get client rectangle (needed for width and height of tray)
		if( hWnd!=NULL )
		{
			::GetClientRect( hWnd, &re );

			// get size of small icons
			dwWidth = (DWORD)GetSystemMetrics(SM_CXSMICON);   //  sm_cxsmicon);
			dwHeight = (DWORD)GetSystemMetrics(SM_CYSMICON);   //  sm_cysmicon);

			// initial y position of mouse - half of height of icon
			y = dwHeight >> 1;
			while( y < (DWORD)re.bottom )  // while y < height of tray
			{
				x = dwHeight>>1;              // initial x position of mouse - half of width of icon
				while( x < (DWORD)re.right ) // while x < width of tray
				{
					::SendMessage( hWnd, WM_MOUSEMOVE, 0, (y<<16)|x); // simulate moving mouse over an icon
					x = x + dwWidth; // add width of icon to x position
				}
				y = y + dwHeight; // add height of icon to y position
			}
		}

		return APP_OK;
	}
	catch (...)
	{
		return APP_NG;
	}
}

LRESULT CVSAlgorithmTaskDlg::m_fnPrintUiMsgUnicode(WPARAM wParam, LPARAM lParam)
{
	CString strMsg = _T("");
	SYSTEMTIME systime;
	::GetLocalTime(&systime);

	strMsg.Format(L"%02d:%02d:%02d.%03d - %s\n", systime.wHour, systime.wMinute, systime.wSecond, 
		systime.wMilliseconds, (LPCTSTR)lParam);

	if(m_ctrlListLog.GetCount() > MAX_GRID_LOG	)
		m_ctrlListLog.DeleteString(MAX_GRID_LOG - 1);

	m_ctrlListLog.InsertString(0,strMsg);
	m_ctrlListLog.SetCurSel(0);

	//free((VOID*)lParam);
	delete[] (wchar_t*)lParam;

	return 0;
}

LRESULT CVSAlgorithmTaskDlg::m_fnPrintUiMsgMultiByte(WPARAM wParam, LPARAM lParam)
{
	wchar_t* pStr;

		//返回多字节大小计算长度
	int strSize = MultiByteToWideChar(CP_ACP, 0, (LPCSTR)lParam, -1, NULL, NULL);
		//wchar_t内存分配

	pStr = new WCHAR[strSize];
	memset(pStr,0,sizeof(WCHAR) * strSize);

		//转换格式
	MultiByteToWideChar(CP_ACP, 0, (LPCSTR)lParam, (int)strlen((LPCSTR)lParam)+1, pStr, strSize);

	CString strMsg = _T("");
	SYSTEMTIME systime;	

	::GetLocalTime(&systime);

	strMsg.Format(L"%02d:%02d:%02d.%03d - %s\n", systime.wHour, systime.wMinute, systime.wSecond, 
		systime.wMilliseconds, pStr);

	if(m_ctrlListLog.GetCount() > MAX_GRID_LOG	)
		m_ctrlListLog.DeleteString(MAX_GRID_LOG - 1);

	m_ctrlListLog.InsertString(0,strMsg);
	m_ctrlListLog.SetCurSel(0);

	//free((VOID*)lParam);
	delete[] (char*)lParam;
	delete[] pStr;

	return 0;
}

void CVSAlgorithmTaskDlg::OnBnClickedBtnTest2()
{
		//TODO:在此添加控制通知处理器代码。
	BYTE* pByteOutput[4] = {NULL, };
	for (int i=0; i<4; i++)
		pByteOutput[i] = new BYTE[6576 * 4384];

	CFileDialog* pDlg = NULL;

		pDlg = new CFileDialog( TRUE, _T("bmp"), NULL, OFN_HIDEREADONLY | OFN_OVERWRITEPROMPT, _T("BMP FILES (*.BMP)|*.bmp|All Files (*.*)|*.*||") );

	CString strPath;
	if(pDlg->DoModal() == IDOK)
	{
		strPath.Format(_T("%s"), pDlg->GetPathName() );
	}
	else
	{
		SAFE_DELETE(pDlg);
		return;
	}

	cv::Mat MatPSImage = cv::imread(CSTR2PCH(strPath), IMREAD_UNCHANGED);
	ImageSave(MatPSImage, _T("E:\\IMTC\\PSTEST\\PS_IMAGE.bmp"));

	BYTE* pByteInput = MatPSImage.data;

	int cnt1 = 0, cnt2 = 0, cnt3 = 0, cnt4 = 0;

	for(UINT32 j=0;j<4384*2;j++)
	{
		int p = 0;
		for(UINT32 i=0; i<6576*2; i++)
		{
			if		(i%2 == 0 && j%2 == 0)
				pByteOutput[1][cnt1++] = pByteInput[j*6576*2+i];
			else if (i%2 == 1 && j%2 == 0)
				pByteOutput[0][cnt2++] = pByteInput[j*6576*2+i];
			else if (i%2 == 0 && j%2 == 1)
				pByteOutput[3][cnt3++] = pByteInput[j*6576*2+i];
			else if (i%2 == 1 && j%2 == 1)
				pByteOutput[2][cnt4++] = pByteInput[j*6576*2+i];
		}
	}

	cv::Mat MatOrg[4];
	for (int i=0; i<4; i++)
	{
		MatOrg[i] = cv::Mat(4384, 6576, CV_8UC1, pByteOutput[i]);
		ImageSave(MatOrg[i], _T("E:\\IMTC\\PSTEST\\%d.bmp"), i);
	}	

	for (int i=0; i<4; i++)
		SAFE_DELETE_ARR(pByteOutput[i]);
}

void CVSAlgorithmTaskDlg::OnBnClickedBtnCntClear()
{
		//TODO:在此添加控制通知处理器代码。
	CString strLogPath = _T(""), strSection = _T(""), strKey = _T("");
	strLogPath.Format(_T("%s\\CountingStageAD_PC%02d.INI"), DEFECT_INFO_PATH, theApp.m_Config.GetPCNum());

	TRY 
	{
				//当前Count读取/增加后写入
		EnterCriticalSection(&theApp.m_csCntFileSafe);		

		for (int nStageNo = 1; nStageNo <= MAX_STAGE_COUNT; nStageNo++)
		{
			strSection.Format(_T("Stage_%d_%d"), nStageNo, theApp.m_Config.GetPCNum());
			WritePrivateProfileString(strSection, _T("AD"), _T("0"), strLogPath);

			for (int nImageNum = 0; nImageNum < theApp.GetGrabStepCount(); nImageNum++)
			{
				if (theApp.GetImageClassify(nImageNum) != E_IMAGE_CLASSIFY_AVI_DUST)
				{	
					strKey.Format(_T("%s_GV"), theApp.GetGrabStepName(nImageNum));
					WritePrivateProfileString(strSection, strKey, _T("0"), strLogPath);
				}
			}
		}

		LeaveCriticalSection(&theApp.m_csCntFileSafe);
	}
	CATCH (CException, e)
	{
		e->Delete();
		theApp.WriteLog(eLOGPROC, eLOGLEVEL_DETAIL, TRUE, TRUE, _T("Exception m_fnCountingStageAD()"));
	}
	END_CATCH
}

void CVSAlgorithmTaskDlg::OnBnClickedButton2()
{

		//TODO:瞳沼泽里的某崇下裂灰烬
	//CDialog dlg(IDD_DIALOG1);
	//dlg.DoModal();
		//TODO:瞳沼泽里的某崇下裂灰烬
	TESTAlgorithm* pDlg = new TESTAlgorithm();
	pDlg->Create(IDD_DIALOG1, this);
	pDlg->ShowWindow(SW_SHOW);
}

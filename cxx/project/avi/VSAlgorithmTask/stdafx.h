
//stdafx.h:经常使用但不经常更改
//标准系统嵌入文件和与项目相关的嵌入文件
//包含的文件

#pragma once

#ifndef _SECURE_ATL
#define _SECURE_ATL 1
#endif

#ifndef VC_EXTRALEAN
#define VC_EXTRALEAN            //很少使用的内容将从Windows标头中排除。
#endif

#include "targetver.h"

#define _ATL_CSTRING_EXPLICIT_CONSTRUCTORS      //某些CString构造函数被显式声明。

//取消隐藏MFC的公共部分和可忽略的警告消息。
#define _AFX_ALL_WARNINGS

#include <afxwin.h>         //MFC的核心和标准组件。
#include <afxext.h>         //MFC扩展。

#include <afxdisp.h>        //MFC自动化类。

#ifndef _AFX_NO_OLE_SUPPORT
#include <afxdtctl.h>           //Internet Explorer 4公共控件的MFC支持。
#endif
#ifndef _AFX_NO_AFXCMN_SUPPORT
#include <afxcmn.h>             //Windows公共控件的MFC支持。
#endif // _AFX_NO_AFXCMN_SUPPORT

#include <afxcontrolbars.h>     //MFC中的功能区和控制栏支持
#include <afx.h>

#ifdef _UNICODE
#if defined _M_IX86
#pragma comment(linker,"/manifestdependency:\"type='win32' name='Microsoft.Windows.Common-Controls' version='6.0.0.0' processorArchitecture='x86' publicKeyToken='6595b64144ccf1df' language='*'\"")
#elif defined _M_X64
#pragma comment(linker,"/manifestdependency:\"type='win32' name='Microsoft.Windows.Common-Controls' version='6.0.0.0' processorArchitecture='amd64' publicKeyToken='6595b64144ccf1df' language='*'\"")
#else
#pragma comment(linker,"/manifestdependency:\"type='win32' name='Microsoft.Windows.Common-Controls' version='6.0.0.0' processorArchitecture='*' publicKeyToken='6595b64144ccf1df' language='*'\"")
#endif
#endif

#include "ErrorProcess.h"
#include "Define.h"

#ifdef X64
#pragma comment(lib,"ClientSockDll_X64.lib")
#else
#pragma comment(lib,"ClientSockDll.lib")
#endif

//生成KYH2022.03.11 dump
#include <DbgHelp.h>
#pragma comment (lib, "DbgHelp")

//////////////////////////////////////////////////////////////////////////
// OpenCV 3.1
//////////////////////////////////////////////////////////////////////////
#include <opencv\cv.h>
#include <opencv\highgui.h>
#include <opencv2\opencv.hpp>
#include <opencv2\core\cuda.hpp>
#include <opencv2\highgui\highgui.hpp>

//#include <opencv2\cudacodec.hpp>

using namespace cv;
using namespace cv::ml;
using namespace cv::cuda;

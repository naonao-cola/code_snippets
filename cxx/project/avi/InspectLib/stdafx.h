//stdafx.h:经常使用但不经常更改
//标准系统嵌入文件和与项目相关的嵌入文件
//包含的文件
//

#pragma once

#include "targetver.h"

#define WIN32_LEAN_AND_MEAN             //很少使用的内容将从Windows标头中排除。
#define _ATL_CSTRING_EXPLICIT_CONSTRUCTORS      //某些CString构造函数被显式声明。

#ifndef VC_EXTRALEAN
#define VC_EXTRALEAN            //很少使用的内容将从Windows标头中排除。
#endif

#include <afx.h>
#include <afxwin.h>         //MFC的核心和标准组件。
#include <afxext.h>         //MFC扩展。
#ifndef _AFX_NO_OLE_SUPPORT
#include <afxdtctl.h>           //Internet Explorer 4公共控件的MFC支持。
#endif
#ifndef _AFX_NO_AFXCMN_SUPPORT
#include <afxcmn.h>                     //Windows公共控件的MFC支持。
#endif // _AFX_NO_AFXCMN_SUPPORT

#include <iostream>
//Windows头文件:
#include <windows.h>

//TODO:此处引用程序所需的其他标题。

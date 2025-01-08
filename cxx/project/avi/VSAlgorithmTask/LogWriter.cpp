#include "StdAfx.h"
#include "logwriter.h"

const int OKAY									= 1;			
const int NG									= 0;

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

/*
*	Module name	:	Class构造函数
*	Parameter	:	Log文件路径
*				文件名
*				存档日期
*				记录时间单位
*	Return		:	无
*	Function	:	Class构造函数,在创建Class时负责初始化功能。
*	Create		:	2006.02.27
*	Version		:	1.0
*/
CAlgLogWriter::CAlgLogWriter(CString pszLogFilePath, TCHAR* pszLogFileName, UINT uPeriodDay, UINT uWriteTerm, bool bEnable)
{
		//添加2015.10.12 Release Mode运行相关的初始化语法
	memset(m_szLogFilePath, 0, _tcslen(m_szLogFilePath));
	memset(m_szLogFileName, 0, _tcslen(m_szLogFileName));
		//用输入的值初始化成员变量。
	_tcsncpy_s(m_szLogFilePath, pszLogFilePath, _tcslen(pszLogFilePath));
	_tcsncpy_s(m_szLogFileName, pszLogFileName, _tcslen(pszLogFileName));
	m_uPeriodDay		= uPeriodDay;
	m_uWriteTerm		= uWriteTerm;
	m_bEnable			= bEnable;

		//检查日志文件路径的最后一个字符是否为"\"。
	if(m_szLogFilePath[_tcslen(m_szLogFilePath)-1] != '\\')
	{
		_stprintf_s(m_szLogFilePath, _T("%s\\"), m_szLogFilePath);
	}

		//初始化最后一个记录的文件名。
	memset(m_szLastWriteFileName,0,MAX_LOG_FILE_NAME);
	m_ulFileNumber		= 0;
	m_hLastFileHandle	= NULL;

		//初始化Critical Section,设置考虑Multi CPU的Spin Count。
	::InitializeCriticalSectionAndSpinCount(&m_csLogWriter, 2000);

		//生成文件路径。
	if(OKAY != m_fnMakePath())
	{
				//错误处理
		_tprintf_s(_T("CAlgLogWriter::CAlgLogWriter -> Path make error\n"));
	}

		//要求删除过期的文件。
	if(OKAY != m_fnDeletePeriodOverFile())
	{
		_tprintf_s(_T("CAlgLogWriter::m_fnConnectFileHandle -> PeriodOverFile delete fail..\n"));
	}

		//添加190716YWS以删除Alg Task中所有过期的日志
	if (OKAY != m_fnPeriodOverAllDeleteFile())
	{
		_tprintf_s(_T("CAlgLogWriter::m_fnConnectFileHandle -> PeriodOverFile delete fail..\n"));
	}
}

/*
*	Module name	:	Class销毁者
*	Parameter	:	无
*	Return		:	无
*	Function	:	消灭者
*	Create		:	2006.02.27
*	Version		:	1.0
*/
CAlgLogWriter::~CAlgLogWriter(void)
{
		//关闭上次记录的文件句柄。
	if(NULL != m_hLastFileHandle)
	{
		fclose(m_hLastFileHandle);
	}

		//删除关键部分
	::DeleteCriticalSection(&m_csLogWriter);
}

/*
*	Module name	:	m_fnWriteLog
*	Parameter	:以	Null结尾的字符串Pointer
*	Return		:	正常结束:1
*				以上结束:除1以外的其他值
*	Function	:记录	Log。
*	Create		:	2006.02.27
*	Version		:	1.1
*	-Version 1.1(金龙泰),2006.11.10
*		-根据Logwriter动作Flag添加功能
*/
int CAlgLogWriter::m_fnWriteLog(const TCHAR* pszLog)
{

		SYSTEMTIME systime;						//了解系统时间的变量
	::GetLocalTime(&systime);

		//如果日志记录功能操作flag为false
	if(!m_bEnable)
	{
				//在屏幕上显示并结束当前Log。
		_tprintf_s(_T("[%02d:%02d:%02d.%03d] %s\n"), systime.wHour, systime.wMinute, systime.wSecond, 
			systime.wMilliseconds, pszLog);
	}
		//否则,如果Log记录Flag为true
	else
	{
				EnterCriticalSection(&m_csLogWriter);	//进入关键部分

				//根据当前时间请求连接Log文件句柄。在函数内,为m_hLastFileHandle设置值。
		if(OKAY != m_fnConnectFileHandle())
		{
						//错误处理
			_tprintf_s(_T("CAlgLogWriter::m_fnWriteLog -> m_fnConnectFileHandle() process Fail..\n"));
		}

				//在字符串的开头插入当前时间,分钟和秒(1/1000)。
				//将当前字符串写入关联的文件。

		if(NULL != m_hLastFileHandle)
		{
			_tprintf_s(_T("[%02d:%02d:%02d.%03d] %s\n"), systime.wHour, systime.wMinute, systime.wSecond, 
				systime.wMilliseconds, pszLog);
			_ftprintf_s(m_hLastFileHandle,_T("[%02d:%02d:%02d.%03d] %s\n"), systime.wHour, systime.wMinute, systime.wSecond, 
				systime.wMilliseconds, pszLog);

			fclose(m_hLastFileHandle);
			m_hLastFileHandle = NULL;
		}
		else 
		{
			_tprintf_s(_T("CAlgLogWriter::m_fnWriteLog -> Log File Handle is NULL\n"));
			_tprintf_s(_T("[%2d:%2d:%2d.%3d] %s\n"), systime.wHour, systime.wMinute, systime.wSecond, 
				systime.wMilliseconds, pszLog);
		}
				LeaveCriticalSection(&m_csLogWriter);	//逃脱Critical Section
	}

		//正常结束
	return	1;
}

/*
*	Module name	:	m_fnConnectFileHandle
*	Parameter	:	无
*	Return		:	正常结束:1
*				以上结束:除1以外的其他值
*	Function	:	根据当前时间创建Log文件并连接文件句柄。
*	Create		:	2006.02.27
*	Version		:	1.0
*/
int CAlgLogWriter::m_fnConnectFileHandle()
{
	SYSTEMTIME	systime;
	TCHAR		szDIRFileName[MAX_LOG_FILE_PATH];
	int			nHour = -1;
	ULONG		ulNewFileNumber;

		//通过组合默认文件名,当前年,月,日和时间来确定实际要记录的文件名。
		//选择时间的标准是以0点为基准加上Write term,然后选择包含该时间段。
		//如果WriteTerm为0或24,则不在文件名中写入时间。
	::GetLocalTime(&systime);
	if(24 <= m_uWriteTerm || 0 >= m_uWriteTerm)
	{
		ulNewFileNumber	= systime.wYear*10000 + systime.wMonth*100 + systime.wDay;
	}
	else
	{
		nHour			= (systime.wHour / m_uWriteTerm) * m_uWriteTerm;
		ulNewFileNumber	= systime.wYear*1000000 + systime.wMonth*10000 + systime.wDay*100 + nHour;	
	}

		//如果上次记录的文件句柄存在,则关闭该文件。
	if(NULL != m_hLastFileHandle)
	{
		fclose(m_hLastFileHandle);
		m_hLastFileHandle = NULL;
	}

		////如果要记录的文件名与上次成功的文件名不相同
	//if(m_ulFileNumber != ulNewFileNumber)
	//{
		//	//请求删除过期的文件。
	//	if(OKAY != m_fnDeletePeriodOverFile())
	//	{
	//		printf("CAlgLogWriter::m_fnConnectFileHandle -> PeriodOverFile delete fail..\n");
	//	}
	//}

		//确定要记录的文件名。如果确定了nHour值,则在文件名中包括时间。
	if(0 <= nHour)
	{
		_stprintf_s(m_szLastWriteFileName, _T("%s_%04d%02d%02d%02d.Log"), m_szLogFileName, systime.wYear, systime.wMonth, systime.wDay, nHour);
	}
	else
	{
		_stprintf_s(m_szLastWriteFileName, _T("%s_%04d%02d%02d.Log"), m_szLogFileName, systime.wYear, systime.wMonth, systime.wDay);
	}

		//确定包含Path的完整名称。
	_stprintf_s(szDIRFileName, _T("%s%s"), m_szLogFilePath, m_szLastWriteFileName);

		//以附加权限打开已确定的File并保存Handle。
	_tfopen_s(&m_hLastFileHandle, szDIRFileName, _T("a"));
	if(NULL == m_hLastFileHandle)
	{
		_tprintf_s(_T("CAlgLogWriter::m_fnConnectFileHandle -> Logfile Create fail..\n"));
		return -1;
	}

		//更新最近记录的文件名的日期提取文件。
	m_ulFileNumber = ulNewFileNumber;

		//正常结束
	return OKAY;
}

/*
*	Module name	:	m_fnMakePath
*	Parameter	:	无
*	Return		:	正常结束:1
*				以上结束:除1以外的其他值
*	Function	:创建	文件的路径。
*	Create		:	2006.02.28
*	Version		:	1.0
*/
int CAlgLogWriter::m_fnMakePath()
{
	int	nRet;
//	int nSize;
//	WCHAR *wszLogPath;

		//检查日志文件路径的最后一个字符是否为"\"。
	if(m_szLogFilePath[_tcslen(m_szLogFilePath)-1] != '\\')
	{
		_stprintf_s(m_szLogFilePath, _T("%s\\"), m_szLogFilePath);
	}

		//以Unicode方式更改日志文件路径的字符串。作为函数参数写入
	//nSize		= MultiByteToWideChar(CP_ACP, 0, m_szLogFilePath, -1, NULL, NULL);
	//wszLogPath	= new WCHAR[nSize];
	//MultiByteToWideChar(CP_ACP, 0, m_szLogFilePath, (int)strlen(m_szLogFilePath) + 1, wszLogPath, nSize);

		//根据更改为Unicode的日志文件路径生成Path。
	nRet = SHCreateDirectoryEx(NULL, m_szLogFilePath, NULL);

	if(ERROR_SUCCESS != nRet)
	{
		//printf("CAlgLogWriter::m_fnMakePath -> Path make fail..\n");
		nRet = OKAY;
	}
	else
	{
		nRet = OKAY;
	}
//	delete []wszLogPath;

		//返回结果值。
	return nRet;
}

/*
*	Module name	:	m_fnDeletePeriodOverFile
*	Parameter	:	无
*	Return		:	正常结束:1
*				以上结束:除1以外的其他值
*	Function	:删除	过期的文件。
*	Create		:	2006.02.28
*	Version		:	1.0
*/
int CAlgLogWriter::m_fnDeletePeriodOverFile()
{
	HANDLE				hSearch;
	WIN32_FIND_DATA		FileData;
	TCHAR				cLogFileName[MAX_LOG_FILE_NAME];
	TCHAR				cDeleteLogFileName[MAX_LOG_FILE_NAME];
	bool				bExistFIle;
	SYSTEMTIME			SystemTime;
	FILETIME			TempFileTime, SystemFileTime;
	UINT				uDifferenceDate;
	int					nErrorCode = OKAY;

		//如果保存时间为0,则必须拥有所有文件,因此正常结束。
	if(0 >= m_uPeriodDay)
	{
		//printf("CAlgLogWriter::m_fnDeletePeriodOverFile -> Period is 0.\n"); 
		return nErrorCode;
	}

		//在保存文件路径中的文件名中,获取包含日志默认文件名的第一个文件的属性。
	_stprintf_s(cLogFileName, _T("%s%s*.log"), m_szLogFilePath, m_szLogFileName);
	hSearch = FindFirstFile(cLogFileName, &FileData);

		//如果文件查找结果显示没有文件,则正常结束
	if (INVALID_HANDLE_VALUE == hSearch) 
	{ 
		//printf("CAlgLogWriter::m_fnDeletePeriodOverFile -> No files found.\n"); 
				//文件查找结束
		FindClose(hSearch);
		return nErrorCode;
	}
		//将"有文件"标志设置为true。
	bExistFIle = true;

		//获取当前系统时间。
	::GetLocalTime(&SystemTime);
	SystemTimeToFileTime(&SystemTime,&SystemFileTime);

		//重复。在文件存在标志为false之前
	while(bExistFIle)
	{
				//求当前时间内文件创建日期的差异。
		TempFileTime.dwHighDateTime = SystemFileTime.dwHighDateTime - FileData.ftLastWriteTime.dwHighDateTime;
		TempFileTime.dwLowDateTime	= SystemFileTime.dwLowDateTime - FileData.ftLastWriteTime.dwLowDateTime;

		uDifferenceDate				= int((TempFileTime.dwHighDateTime*4294967296 + TempFileTime.dwLowDateTime)/864000000000);

				//如果日期差异超过了保存期限,则删除文件。
		if(uDifferenceDate >= m_uPeriodDay)
		{
			_stprintf_s(cDeleteLogFileName, _T("%s%s"), m_szLogFilePath, FileData.cFileName);
			DeleteFile(cDeleteLogFileName);
		}

				//如果在查找下一个文件时出现错误
		if(!FindNextFile(hSearch, &FileData))
		{
			nErrorCode = GetLastError();
						//如果没有以下文件
			if(ERROR_NO_MORE_FILES == nErrorCode)
			{
							//正常结束
				bExistFIle = false;
				nErrorCode = OKAY;
			}
			else
			{
								//无法获取以下文件
				_tprintf_s(_T("CAlgLogWriter::m_fnDeletePeriodOverFile -> Couldn't find next file.\n"));
				bExistFIle = false;
			}
		}
	}

		//文件查找结束
	FindClose(hSearch);

	return nErrorCode;
}

//在Alg Task中添加190716YWS以删除所有过期的日志
int CAlgLogWriter::m_fnPeriodOverAllDeleteFile()
{
	HANDLE				hSearch;
	WIN32_FIND_DATA		FileData;
	TCHAR				cLogFileName[MAX_LOG_FILE_NAME];
	TCHAR				cDeleteLogFileName[MAX_LOG_FILE_NAME];
	bool				bExistFIle;
	SYSTEMTIME			SystemTime;
	FILETIME			TempFileTime, SystemFileTime;
	UINT				uDifferenceDate;
	int					nErrorCode = OKAY;

	if (0 >= m_uPeriodDay)
	{
		return nErrorCode;
	}

		//如果保存时间为0,则必须拥有所有文件,因此正常结束。
	if (0 >= m_uPeriodDay)
	{
		//printf("CAlgLogWriter::m_fnDeletePeriodOverFile -> Period is 0.\n"); 
		return nErrorCode;
	}

		//在保存文件路径中的文件名中,获取包含日志默认文件名的第一个文件的属性。
	_stprintf_s(cLogFileName, _T("%s*.*"), m_szLogFilePath);
	hSearch = FindFirstFile(cLogFileName, &FileData);

		//如果文件查找结果显示没有文件,则正常结束
	if (INVALID_HANDLE_VALUE == hSearch)
	{
		//printf("CAlgLogWriter::m_fnDeletePeriodOverFile -> No files found.\n"); 
				//文件查找结束
		FindClose(hSearch);
		return nErrorCode;
	}
		//将"有文件"标志设置为true。
	bExistFIle = true;

		//获取当前系统时间。
	::GetLocalTime(&SystemTime);
	SystemTimeToFileTime(&SystemTime, &SystemFileTime);

		//重复。在文件存在标志为false之前
	while (bExistFIle)
	{
				//求当前时间内文件创建日期的差异。
		TempFileTime.dwHighDateTime = SystemFileTime.dwHighDateTime - FileData.ftLastWriteTime.dwHighDateTime;
		TempFileTime.dwLowDateTime = SystemFileTime.dwLowDateTime - FileData.ftLastWriteTime.dwLowDateTime;

		uDifferenceDate = int((TempFileTime.dwHighDateTime * 4294967296 + TempFileTime.dwLowDateTime) / 864000000000);

				//如果日期差异超过了保存期限,则删除文件。
		if (uDifferenceDate >= m_uPeriodDay)
		{
			_stprintf_s(cDeleteLogFileName, _T("%s%s"), m_szLogFilePath, FileData.cFileName);
			DeleteFile(cDeleteLogFileName);
		}

				//如果在查找下一个文件时出现错误
		if (!FindNextFile(hSearch, &FileData))
		{
			nErrorCode = GetLastError();
						//如果没有以下文件
			if (ERROR_NO_MORE_FILES == nErrorCode)
			{
							//正常结束
				bExistFIle = false;
				nErrorCode = OKAY;
			}
			else
			{
								//无法获取以下文件
				_tprintf_s(_T("CAlgLogWriter::m_fnDeletePeriodOverFile -> Couldn't find next file.\n"));
				bExistFIle = false;
			}
		}
	}

		//文件查找结束
	FindClose(hSearch);

	return nErrorCode;
}


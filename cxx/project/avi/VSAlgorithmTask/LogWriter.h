#pragma once

#define	MAX_LOG_BUFFER			1024						//日志记录时使用的缓冲区大小
#define MAX_LOG_FILE_PATH		200							//Log文件存储路径的最大长度
#define MAX_LOG_FILE_NAME		500							//Log文件名的最大长度

/*
*	Module name	:	CAlgLogWriter
*	Function	:提供	日志记录功能。
*	Create		:	2006.02.21
*	Version		:	1.1
*	-Version 1.1(金龙泰),2006.11.10
*		-Logwriter动作添加Flag
*/
class CAlgLogWriter
{
public:
	CAlgLogWriter(CString pszLogFilePath, TCHAR* pszLogFileName, UINT uPeriodDay, UINT uWriteTerm, bool bEnable = true);
	~CAlgLogWriter(void);

		/*成员函数*/
public:
		int		m_fnWriteLog(const TCHAR* pszLog);			//记录Log。记录Log。

private:
		int		m_fnConnectFileHandle();					//根据当前时间创建Log文件并连接文件句柄。
		int		m_fnMakePath();								//创建文件的路径。
		int		m_fnDeletePeriodOverFile();					//删除过期的文件。
		int		m_fnPeriodOverAllDeleteFile();				//删除所有过期的日志文件。190716 YWS

		/*成员变量*/
private:
		TCHAR	m_szLogFilePath[MAX_LOG_FILE_PATH];			//日志文件路径。
		TCHAR	m_szLogFileName[MAX_LOG_FILE_NAME];			//Log具有默认文件名。名字后面有时间。
		UINT	m_uPeriodDay;								//Log文件存档日期
		UINT	m_uWriteTerm;								//Log文件记录时间,决定以多少小时为单位划分文件。(最多24小时。)
		TCHAR	m_szLastWriteFileName[MAX_LOG_FILE_NAME];	//最后记录成功的File name(用于按时间段记录和Log Period处理)
		FILE*	m_hLastFileHandle;							//最后,拥有记录成功的File Handle。
		CRITICAL_SECTION	m_csLogWriter;					//日志记录所需的紧急部分
		ULONG	m_ulFileNumber;								//最后,成功记录的File名称中只有时区具有数字格式。用于快速比较文件名
	bool	m_bEnable;
};


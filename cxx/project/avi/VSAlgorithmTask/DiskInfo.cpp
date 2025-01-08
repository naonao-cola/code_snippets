#include "StdAfx.h"
#include "DiskInfo.h"
#include "VSAlgorithmTask.h"

CDiskInfo::CDiskInfo(void)
{
}

CDiskInfo::~CDiskInfo(void)
{
}

void CDiskInfo::CreateDriveInfoVector()
{  
	DWORD   dwDiskInfo;
	int		nCount = 0;
	CString strDrive;	
		BOOL	bExistRamDrive = FALSE;		//确认Ram Drive是否存在
		char pTmp[256];      //保存local drive名称的列表
	memset(pTmp, '\0', 256);
	//CString	ExDrive1 = _T("C:\\");
	//CString	ExDrive2 = _T("D:\\");
	//CString	ExDrive3 = _T("E:\\");

	dwDiskInfo = GetLogicalDrives();

	tDriveInfoParam tTempDriveInfo;

	wchar_t szDrive[MAX_PATH] = {0, };
	GetLogicalDriveStrings(MAX_PATH, szDrive);
		//可以解析szDrive获得驱动器列表。
	for(int i=0 ; i<256 ; i++) 
	{
		if(szDrive[i] == '\0' && szDrive[i+1] == '\0')
			break;

		strDrive.Empty();
		while(szDrive[i] != '\0')
			strDrive += szDrive[i++];

		switch(GetDriveType(strDrive)) 
		{
		case DRIVE_CDROM:
			break;
		case DRIVE_REMOVABLE:
			break;
		case DRIVE_FIXED:
 			tTempDriveInfo.Drive_Name = strDrive;
			tTempDriveInfo.nDriveLimitSize = theApp.m_Config.GetDriveLimitSize();
			m_tDriveInfoParam.push_back(tTempDriveInfo);
			nCount++;			
			break;
		default :
			break;
		}
	}
}

void CDiskInfo::RenewalDiskInfo()
{
	float  TolalSize, TotalFree;
	float  dPrecent = 0.0;
	CString	Drive_name = _T("");

	ULARGE_INTEGER  lTotalBytes;
	ULARGE_INTEGER  lTotalFree;

	for(int i = 0 ; i < m_tDriveInfoParam.size() ; i++)
	{
				if( GetDriveType( m_tDriveInfoParam[i].Drive_Name ) == DRIVE_FIXED ) //仅计算固定驱动器
		{
			GetDiskFreeSpaceEx(m_tDriveInfoParam[i].Drive_Name , &m_lFreeBytesAvailable, &lTotalBytes, &lTotalFree);

			TolalSize = ((float)lTotalBytes.u.HighPart *4 + ((float)lTotalBytes.u.LowPart/(1024*1024*1024)));
			TotalFree = ((float)lTotalFree.u.HighPart *4 + ((float)lTotalFree.u.LowPart/(1024*1024*1024)));

			dPrecent = (float)(TotalFree * 1.0 / TolalSize * 100);
			m_tDriveInfoParam[i].Free_PerCent = dPrecent;

		}
	}

}

CString CDiskInfo::GetAvailableDrivePath(CString strCurrentDrive)
{	
	int	n_dCheck = 0;	
		//2015.09.01添加驱动器特定的容量限制功能和完整的序列修改	
	int nLastDriveIndex = 0;
		//使用到当前驱动器的最大容量
	for (int nDriveIndex=0 ; nDriveIndex< m_tDriveInfoParam.size(); nDriveIndex++)
	{
		if ( m_tDriveInfoParam[nDriveIndex].Drive_Name == strCurrentDrive)
		{			
			if(m_tDriveInfoParam[nDriveIndex].bUseFlag == true && m_tDriveInfoParam[nDriveIndex].nDriveLimitSize > 100 - m_tDriveInfoParam[nDriveIndex].Free_PerCent)
				return strCurrentDrive;
			nLastDriveIndex = nDriveIndex;
		}
	}

		//除当前使用的驱动器外,按顺序扫描以下驱动器:
	//for (int i=0 ; i< DRIVE_PATH_COUNT ; i++)
	for (int i=0 ; i< m_tDriveInfoParam.size() - 1; i++)
	{
				//驱动器顺序访问
		if (++nLastDriveIndex > m_tDriveInfoParam.size() -1)	nLastDriveIndex = 0;

		if(m_tDriveInfoParam[nLastDriveIndex].bUseFlag == true && m_tDriveInfoParam[nLastDriveIndex].nDriveLimitSize > 100 - m_tDriveInfoParam[nLastDriveIndex].Free_PerCent)
		{
			n_dCheck = 1;
			break;	
		}
	}
	//Default
	if(n_dCheck == 0)
	{
		AfxMessageBox(_T("Please Check Free Size on Hard Disk!!!"));
		theApp.WriteLog(eLOGPROC, eLOGLEVEL_SIMPLE, TRUE, FALSE, _T("Please Check Free Size on Hard Disk!!!"));
	}
	return m_tDriveInfoParam[nLastDriveIndex].Drive_Name;
}

void CDiskInfo::SetUseDisk(CString strDrive)
{
	CString strTemp;
	strTemp.Format(_T("%s:\\"),strDrive);
	//strDrive = strDrive + _T("\:\\");
	for (int nDriveIndex=0 ; nDriveIndex< m_tDriveInfoParam.size(); nDriveIndex++)
	{
		if ( m_tDriveInfoParam[nDriveIndex].Drive_Name == strTemp )
			m_tDriveInfoParam[nDriveIndex].bUseFlag = true;
	}
}
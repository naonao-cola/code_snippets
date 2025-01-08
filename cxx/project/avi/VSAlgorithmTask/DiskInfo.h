#pragma once

#include <vector>

struct tDriveInfoParam
{
	CString				Drive_Name;
	float				Free_PerCent;
	bool				bUseFlag;
	int					nDriveLimitSize;

	tDriveInfoParam()
	{		
		Free_PerCent = 0.0;	 
		Drive_Name = _T("");	
		bUseFlag = false;
		nDriveLimitSize = 80;
	}
};

class CDiskInfo
{
public:
	CDiskInfo(void);
	~CDiskInfo(void);

	void CreateDriveInfoVector();
	void RenewalDiskInfo();

	void		SetUseDisk(CString strDrive);
	int			GetNumOfDisk()								{	return (int)m_tDriveInfoParam.size()				;};
	CString		GetDriveName(int nDiskNum)					{	return m_tDriveInfoParam[nDiskNum].Drive_Name;		;};
	CString		GetAvailableDrivePath(CString strCurrentDrive);

private:
	// Disk Check
	std::vector<tDriveInfoParam>	m_tDriveInfoParam;
	ULARGE_INTEGER  m_lFreeBytesAvailable;
	ULARGE_INTEGER  m_lTotalBytes;
	ULARGE_INTEGER  m_lTotalFree;	

	// Disk Check Thread

};


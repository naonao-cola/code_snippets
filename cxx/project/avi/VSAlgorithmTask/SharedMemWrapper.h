#pragma once
#pragma pack(push)
#pragma pack(1)		// for Padding bit
#include <afxstr.h>
#include <wtypes.h>
//#include <windows.h>

#define MAX_IMAGE_COUNT	30		//共享内存已存储最大数量

#define SAFE_DELETE(p)	if(p){ delete (p); (p)=0; }			///<delete语句重定义,仅在分配内存时有效
#define SAFE_DELETE_ARR(p) if(p){ delete [](p); (p)=0;}		//重定义/<delete[]语句,仅当分配了内存时才起作用

struct SMemImageInfo
{
	SMemImageInfo();		
	SMemImageInfo &operator = (const SMemImageInfo & rhs);

	int		nImageWidth;
	int		nImageHeight;
	int		nImageBitrate;
	int		nImageBandWidth;
	int		nImageCount;
	//BYTE    byteReserved[151173120];	    // AVI:14208*10640=151173120    APP: 11700*700 = 81900000    SVI:3830*2553*3=29333970
};

class CSharedMemWrapper
{
public:
	CSharedMemWrapper();
	CSharedMemWrapper(int nImageSizeX, int nImageSizeY, int nImageBitrate, int nImageBandWidth, CString strDrv_CamNo, int nImageCount);
	virtual ~CSharedMemWrapper();

	// Global Function
	BOOL			CreateSharedMem(int nImageSizeX, int nImageSizeY, int nImageBitrate, int nImageBandWidth, CString strDrv_CamNo, int nImageCount);
	BOOL			OpenSharedMem(CString strDrv_CamNo);
	BOOL			IsInitial();

	BYTE*			GetImgAddress(int nImageCount);
	int				GetImgWidth();
	int				GetImgHeight();
	int				GetImgBitrate();
	int				GetImgBandWidth();
	double			GetImgBufSize();

	BOOL			SetImageInfo(SMemImageInfo *pData);
	SMemImageInfo*	GetImageInfo();

	void			DeleteMemory();

private:
	HANDLE			m_hAsignedMemory;
	BYTE*			m_pSharedMemory;

	SMemImageInfo*	m_pImageInfo;
	unsigned char*	m_pImageBuffer[MAX_IMAGE_COUNT];

	BYTE*			GetGlobalMemoryPtr(INT64 &nStartPos, INT64 nReadSize);

};

#pragma pack(pop)
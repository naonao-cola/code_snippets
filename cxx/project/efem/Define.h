#pragma once
#include <filesystem> // C++17
#include <time.h>
#include <windows.h>

// #include "../framework/BaseAlgo.h"
namespace fs = std::filesystem;

#define SAVE_DEBUG_IMG
#define SAVE_DEBUG_FEATRUE

#define START_TIMER \
    double start_time = clock();

#define END_TIMER \
    double end_time = clock(); \
    double duration = (end_time - start_time) / CLOCKS_PER_SEC; \
    LOGI("[{}]\t[{}]msc.", __FUNCTION__, duration);



#define PI 3.1415926
const int MAX_MEM_SIZE_E_DEFECT_JUDGMENT_COUNT = 160;
const int MAX_JUDGE_NUM = 10;
#define IMAGE_MAX_GV		256


#define BLOB_DRAW_ROTATED_BOX				0x000010
#define BLOB_DRAW_BOUNDING_BOX				0x000020
#define BLOB_DRAW_BLOBS						0x000080
#define BLOB_DRAW_BLOBS_CONTOUR				0x000100



enum ENUM_ERROR_CODE
{
	E_ERROR_CODE_ALIGN_IMAGE_OVER = -1,
	E_ERROR_CODE_ALIGN_WARNING_PARA,
	E_ERROR_CODE_TRUE,
	E_ERROR_CODE_ALIGN_CAN_NOT_CALC,
	E_ERROR_CODE_ALIGN_NO_DATA
};

enum ENUM_ALIGN_TYPE
{
	E_ALIGN_TYPE_LEFT = 0,
	E_ALIGN_TYPE_TOP,
	E_ALIGN_TYPE_RIGHT,
	E_ALIGN_TYPE_BOTTOM,
	E_ALIGN_TYPE_END
};

struct STRU_JUDGEMENT
{
	bool bUse;			
	int nSign;			
	double dValue;
	std::string name;


	struct STRU_JUDGEMENT()
	{
		memset(this, 0, sizeof(STRU_JUDGEMENT));
	}
};

struct STRU_DEFECT_ITEM
{
	BOOL bDefectItemUse;   
	std::string strItemName;

	STRU_JUDGEMENT Judgment[2 * MAX_MEM_SIZE_E_DEFECT_JUDGMENT_COUNT]; 

	STRU_DEFECT_ITEM()
	{
		memset(this, 0, sizeof(STRU_DEFECT_ITEM));
	}
};


enum ENUM_BOLB_FEATURE
{
	E_FEATURE_AREA = 0,	
	E_FEATURE_BOX_AREA,	
	E_FEATURE_BOX_RATIO,
	E_FEATURE_SUM_GV,	
	E_FEATURE_MIN_GV,	
	E_FEATURE_MAX_GV,	
	E_FEATURE_MEAN_GV,	
	E_FEATURE_DIFF_GV,	
	E_FEATURE_BK_GV,	
	E_FEATURE_STD_DEV,	
	E_FEATURE_SEMU,		
	E_FEATURE_COMPACTNESS,	
	E_FEATURE_MIN_GV_RATIO,	
	E_FEATURE_MAX_GV_RATIO,	
	E_FEATURE_DIFF_GV_RATIO,
	E_FEATURE_PERIMETER,	
	E_FEATURE_ROUNDNESS,	
	E_FEATURE_ELONGATION,	
	E_FEATURE_BOX_X,	
	E_FEATURE_BOX_Y,	

	E_FEATURE_MIN_BOX_AREA,
	E_FEATURE_MINOR_AXIS,	
	E_FEATURE_MAJOR_AXIS,	
	E_FEATURE_AXIS_RATIO,	
	E_FEATURE_MIN_BOX_RATIO,

	E_FEATURE_CENTER_X,	
	E_FEATURE_CENTER_Y,	

	E_FEATURE_MEAN_DELTAE,	

	E_FEATURE_EDGE_DISTANCE,

	E_FEATURE_GV_UP_COUNT_0,	
	E_FEATURE_GV_UP_COUNT_1,	
	E_FEATURE_GV_UP_COUNT_2,	
	E_FEATURE_GV_DOWN_COUNT_0,	
	E_FEATURE_GV_DOWN_COUNT_1,	
	E_FEATURE_GV_DOWN_COUNT_2,	

	E_FEATURE_IS_EDGE_C,	
	E_FEATURE_IS_EDGE_V,	
	E_FEATURE_IS_EDGE_H,	
	E_FEATURE_IS_EDGE_CENTER,	

	

	E_FEATURE_COUNT
};


struct stBLOB_FEATURE
{
	bool				bFiltering;				

	cv::Rect			rectBox;				
	long				nArea;					
	long				nBoxArea;				
	float				fBoxRatio;				
	cv::Point			ptCenter;				
	long				nSumGV;					
	long				nMinGV;					
	long				nMaxGV;					
	float				fMeanGV;				
	float				fDiffGV;				
	float				fBKGV;					
	float				fStdDev;				
	float				fSEMU;					
	float				fCompactness;			//���ն�
	float				nMinGVRatio;			
	float				nMaxGVRatio;			
	float				fDiffGVRatio;			
	float				fPerimeter;				//�����ܳ�
	float				fRoundness;				//Բ��
	float				fElongation;			//����̶�
	float				fMinBoxArea;			
	float				fMinorAxis;				
	float				fMajorAxis;				
	float				fAxisRatio;				
	float				fAngle;					
	float				fMinBoxRatio;			

	long				fDefectMeanGV;			
	int                 nDistanceFromEdge;      

	float				fMeanDelataE;			

	cv::Rect               FeaRectROI;

	__int64				nHist[256];	

	cv::Size			BoxSize;				

	std::vector <cv::Point>	ptIndexs;				
	std::vector <cv::Point>	ptContours;				

	stBLOB_FEATURE() :
		nArea(0)
	{
		bFiltering				= false;

		rectBox					= cv::Rect(0, 0, 0, 0);
		nArea					= 0;
		nBoxArea				= 0;
		fBoxRatio				= 0.0f;
		nSumGV					= 0;
		nMinGV					= 0;
		nMaxGV					= 0;
		fMeanGV					= 0.0f;
		fDiffGV					= 0.0f;
		fBKGV					= 0.0f;
		fStdDev					= 0.0f;
		fSEMU					= 0.0f;
		fCompactness			= 0.0f;
		nMinGVRatio				= 0.0f;
		nMaxGVRatio				= 0.0f;
		fDiffGVRatio			= 0.0f;
		fPerimeter				= 0.0f;
		fRoundness				= 0.0f;
		fElongation				= 0.0f;
		fMinBoxArea				= 0.0f;
		fMinorAxis				= 0.0f;
		fMajorAxis				= 0.0f;
		fAxisRatio				= 0.0f;
		fAngle					= 0.0f;
		fMinBoxRatio			= 0.0f;

		fMeanDelataE			= 0.0f;

		fDefectMeanGV			= 0;
		nDistanceFromEdge		= 0;
		FeaRectROI				= cv::Rect(0, 0, 0, 0);
		memset(nHist, 0, sizeof(__int64) * 256);

		ptCenter = cv::Size(0, 0);
		BoxSize = cv::Size(0, 0);

		std::vector <cv::Point>().swap(ptIndexs);
		std::vector <cv::Point>().swap(ptContours);
	}
};

enum ENUM_SIGN_OF_INEQUALITY
{
	E_SIGN_EQUAL = 0,			// == 
	E_SIGN_NOT_EQUAL,			// != 
	E_SIGN_GREATER,				// >  
	E_SIGN_LESS,				// <  
	E_SIGN_GREATER_OR_EQUAL,	// >= 
	E_SIGN_LESS_OR_EQUAL,		// <= 
	E_SIGN_GREATER_OR_EQUAL_OR,	// >||
	E_SIGN_LESS_OR_EQUAL_OR		// ||<
};

enum ENUM_EDGE_POSITION_DIRECTION
{
	E_POSITION_BR = 0,			//���½�
	E_POSITION_BL,				//���½�
	E_POSITION_TR,				//���Ͻ�
	E_POSITION_TL,				//���Ͻ�

	E_POSITION_B,				//�±�
	E_POSITION_T,				//�ϱ�
	E_POSITION_R,				//�ұ�
	E_POSITION_L				//���
};

// void WriteBlobResultInfo(std::vector<stBLOB_FEATURE> BlobResultTotal){
// 	std::string fpath = fs::current_path().string() + "\\debugImg\\";
//     if (!fs::exists(fpath)) {
//         if (!fs::create_directories(fpath)) {
//             std::cerr << "Error creating directory: " << fpath << std::endl;
//             std::string fpath1 = fs::current_path().string() + "\\Unkonw";
//             fs::create_directories(fpath1);
//         }
//     }
// 	std::string filePath = fpath + "BlobFeature.txt";
// 	std::ofstream file(filePath);
// 	file << "这是一个示例文本文件。\n";
// 	file.close();
// }
#pragma once

#include "Define.h"

class CBlob_Sub
{
public:
	CBlob_Sub(void);
	~CBlob_Sub(void);

		//只提取最大的Blob,然后提取最大Blob的Rect。
	BOOL FindBiggestBlob(cv::Mat& src, cv::Mat& dst);
	
		//提取最大Blob的RotateRect
	BOOL GetBiggestBlobMinAreaRect(cv::Mat& src, cv::RotatedRect& rtRect);

};


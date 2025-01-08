
#include "Define.h"

EXTERNC AFX_API_EXPORT BOOL  GetCrossPointByTwoStraightLines(float a1, float b1, float c1, float a2, float b2, float c2, float &fx, float &fy );

EXTERNC AFX_API_EXPORT BOOL GetDistance_PointToLine( float x0, float y0, float a1, float b1, float c1, float& fDis );

////////////Edge相关

// Basic Sobel : 3 x 3 Mask
EXTERNC AFX_API_EXPORT BOOL	 Edge_BasicSobel(Mat& Src, Mat& Des, int nKernalSize = 3, double dScale=1, double dDelta=0, int nBorderType =BORDER_DEFAULT);

//Basic Thinning:寻找厚边缘的中心线。注意:大画面可能速度慢。
EXTERNC AFX_API_EXPORT void	 Edge_Thinning(Mat& src, Mat& dst);

EXTERNC AFX_API_EXPORT BOOL	GuidedEdgeEnhance(const Mat Src, Mat& O_Des);

EXTERNC AFX_API_EXPORT BOOL DetailEnhancement(Mat& src, Mat& dst, float factor);

EXTERNC AFX_API_EXPORT void Threshold_MultiOtsu(Mat src, Mat& dst);

//返回最大的Blob
EXTERNC AFX_API_EXPORT void FindBiggestBlob(cv::Mat& src, cv::Mat& dst);

//返回最大的Contour
EXTERNC AFX_API_EXPORT void FindBiggestContour(cv::Mat src, vector<vector<cv::Point>> &ptBiggest);

EXTERNC AFX_API_EXPORT void SelectBiggestBlob(cv::Mat& src, cv::Mat& dst, int nSelectInx);

//用最小乘法获取斜率
EXTERNC AFX_API_EXPORT long MethodOfLeastSquares(vector<cv::Point> pt, double &nA);

EXTERNC AFX_API_EXPORT void MinimumFilter(cv::Mat src, cv::Mat& dst, int nMaskSize);

EXTERNC AFX_API_EXPORT void MaximumFilter(cv::Mat src, cv::Mat& dst, int nMaskSize);


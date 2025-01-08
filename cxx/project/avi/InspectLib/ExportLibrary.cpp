#include "stdafx.h"
#include "ExportLibrary.h"
#include "BasicLibrary.h"
#include "DomainFilter.h"
#ifdef _DEBUG
#define new DEBUG_NEW
#endif

using namespace cv;
// a1x + b1y + c1 = 0;
// a2x + b2y + c2 = 0;
EXTERNC AFX_API_EXPORT BOOL  GetCrossPointByTwoStraightLines(float a1, float b1, float c1, float a2, float b2, float c2, float& fx, float& fy)
{
	//两条直线的交点!
	//解联立方程,求x,y。
	double det = b2 * a1 - b1 * a2;

	if (det == 0.f) return FALSE;

	fx = (float)((c2 * b1 - c1 * b2) / det);
	fy = (float)((a2 * c1 - a1 * c2) / det);

	return TRUE;
}

//点:(x0,y0)
//直线方程a1x+b1y+c1=0时。
EXTERNC AFX_API_EXPORT BOOL GetDistance_PointToLine(float x0, float y0, float a1, float b1, float c1, float& fDis)
{
	double det = sqrt(a1 * a1 + b1 * b1);
	if (det == 0.f) return FALSE;

	fDis = (float)((fabs(a1 * x0 + b1 * y0 + c1)) / det);

	return TRUE;
}

EXTERNC AFX_API_EXPORT BOOL	Edge_BasicSobel(Mat& Src, Mat& Des, int nKernalSize, double dScale, double dDelta, int nBorderType)
{
	Mat gray = Src.clone();
	//cvtColor(m_OrgMat, gray, CV_BGR2GRAY);
	Mat grad_X, grad_Y;
	Mat abs_grad_X, abs_grad_Y;
	int nDepth = CV_16S;

	Sobel(gray, grad_X, nDepth, 1, 0, nKernalSize, dScale, dDelta, nBorderType);
	convertScaleAbs(grad_X, abs_grad_X);

	Sobel(gray, grad_Y, nDepth, 0, 1, nKernalSize, dScale, dDelta, nBorderType);
	convertScaleAbs(grad_Y, abs_grad_Y);

	cv::addWeighted(abs_grad_X, 0.5, abs_grad_Y, 0.5, 0, Des);

	return TRUE;
}

/*/////////////////////////////////////////////////////////////////////////////////////////////
// Function name: Thinning()
//功能:执行细线化算法
//输入:src(输入画面),dst(结果画面)
//输出:none
// Final update: 2015.10.28
//参考论文:与Thinning相关的ZHANG
/////////////////////////////////////////////////////////////////////////////////////////////*/
EXTERNC AFX_API_EXPORT void Edge_Thinning(cv::Mat& src, cv::Mat& dst)
{
	int x, y, p, q, xp, yp, xm, ym, cc, cd;
	int np1, sp1, hv;
	int cnt = 0, chk = 0, flag = 0;
	int row = src.rows;
	int col = src.cols;

	unsigned char* m_BufImage;
	unsigned char* m_ResultImg;
	m_BufImage = (unsigned char*)malloc(sizeof(unsigned char) * row * col);
	m_ResultImg = (unsigned char*)malloc(sizeof(unsigned char) * row * col);

	for (y = 0; y < row; y++) {
		for (x = 0; x < col; x++) {
			*(m_ResultImg + (col * y) + x) = src.at<uchar>(y, x);
		}
	}

	cv::Mat Test(row, col, CV_8UC1, (BYTE*)m_ResultImg);

	do {

		for (y = 1; y < row - 1; y++) {
			for (x = 1; x < col - 1; x++) {
				*(m_BufImage + (col * y) + x) = 0;
			}
		}

		if (chk == 0) flag = 0;
		chk = cnt % 2;
		cnt++;

		for (y = 1; y < row - 1; y++) {
			ym = y - 1;
			yp = y + 1;
			for (x = 1; x < col - 1; x++) {
				if (*(m_ResultImg + (col * y) + x) == 0) continue;

				np1 = 0;
				for (q = y - 1; q <= y + 1; q++) {
					for (p = x - 1; p <= x + 1; p++) {
						if (*(m_ResultImg + (col * q) + p) != 0) np1++;
					}
				}

				//第一条件。非零像素计数
				//数据少意味着完全外围或噪音部分=>数据处理没有意义。
				//数据多是完全中央部分的意思=>不需要进行数据处理。
				if (np1 < 3 || np1 > 7) {
					*(m_BufImage + (col * y) + x) = 255;
					continue;
				}

				xm = x - 1;
				xp = x + 1;
				sp1 = 0;

				//p9,p2,p3		=>匹配	(xm,ym),(x,ym),(xp,ym)
				//p8,p1,p4		=>匹配(xm,y),(x,y),(xp,y)
				//p7,p6,p5		=>匹配(xm,yp),(x,yp),(xp,yp)

				//查找周围像素的0-1模式
				if (*(m_ResultImg + (col * ym) + x) == 0 && *(m_ResultImg + (col * ym) + xp) != 0) sp1++;		//p2和P3
				if (*(m_ResultImg + (col * ym) + xp) == 0 && *(m_ResultImg + (col * y) + xp) != 0) sp1++;		//p3和p4
				if (*(m_ResultImg + (col * y) + xp) == 0 && *(m_ResultImg + (col * yp) + xp) != 0) sp1++;		//p4和p5
				if (*(m_ResultImg + (col * yp) + xp) == 0 && *(m_ResultImg + (col * yp) + x) != 0) sp1++;		//p5和p6
				if (*(m_ResultImg + (col * yp) + x) == 0 && *(m_ResultImg + (col * yp) + xm) != 0) sp1++;		//p6和p7
				if (*(m_ResultImg + (col * yp) + xm) == 0 && *(m_ResultImg + (col * y) + xm) != 0) sp1++;		//p7和p8
				if (*(m_ResultImg + (col * y) + xm) == 0 && *(m_ResultImg + (col * ym) + xm) != 0) sp1++;		//p8和p9
				if (*(m_ResultImg + (col * ym) + xm) == 0 && *(m_ResultImg + (col * ym) + x) != 0) sp1++;		//p9和p2

				//如果只有一个0-1模式,则不应使用。
				if (sp1 != 1) {
					*(m_BufImage + (col * y) + x) = 255;
					continue;
				}

				if (chk == 0) {

					// East  p2, p4, p6
					cc = *(m_ResultImg + (col * ym) + x) * *(m_ResultImg + (col * y) + xp);
					cc = cc * *(m_ResultImg + (col * yp) + x);

					// south  p4, p6, p8
					cd = *(m_ResultImg + (col * y) + xp) * *(m_ResultImg + (col * yp) + x);
					cd = cd * *(m_ResultImg + (col * y) + xm);
				}
				else {
					// p2, p4, p8  North
					cc = *(m_ResultImg + (col * ym) + x) * *(m_ResultImg + (col * y) + xp);
					cc = cc * *(m_ResultImg + (col * y) + xm);

					// p2, p6, p8 West
					cd = *(m_ResultImg + (col * ym) + x) * *(m_ResultImg + (col * yp) + x);
					cd = cd * *(m_ResultImg + (col * y) + xm);
				}

				//条件3
				//case 1)如果South or East成员的乘积为0,则删除
				//case 2)如果North或West成员的乘积为0,则删除
				if (cc != 0 || cd != 0) {
					*(m_BufImage + (col * y) + x) = 255;
					continue;
				}
				flag = 1;
			}
		}

		for (y = 1; y < row - 1; y++) {
			for (x = 1; x < col - 1; x++) {
				*(m_ResultImg + (col * y) + x) = *(m_BufImage + (col * y) + x);
			}
		}
	} while (!(chk == 1 && flag == 0));

	//进行以获得薄曲线
	for (y = 1; y < row - 1; y++) {
		yp = y + 1;
		ym = y - 1;
		for (x = 1; x < col - 1; x++) {
			if (*(m_ResultImg + (col * y) + x) == 0) continue;

			xm = x - 1;
			xp = x + 1;
			sp1 = 0;
			//对于存在数据的部分,查找周围像素的0-1模式
			if (*(m_ResultImg + (col * ym) + x) == 0 && *(m_ResultImg + (col * ym) + xp) != 0) sp1++;
			if (*(m_ResultImg + (col * ym) + xp) == 0 && *(m_ResultImg + (col * y) + xp) != 0) sp1++;
			if (*(m_ResultImg + (col * y) + xp) == 0 && *(m_ResultImg + (col * yp) + xp) != 0) sp1++;
			if (*(m_ResultImg + (col * yp) + xp) == 0 && *(m_ResultImg + (col * yp) + x) != 0) sp1++;
			if (*(m_ResultImg + (col * yp) + x) == 0 && *(m_ResultImg + (col * yp) + xm) != 0) sp1++;
			if (*(m_ResultImg + (col * yp) + xm) == 0 && *(m_ResultImg + (col * y) + xm) != 0) sp1++;
			if (*(m_ResultImg + (col * y) + xm) == 0 && *(m_ResultImg + (col * ym) + xm) != 0) sp1++;
			if (*(m_ResultImg + (col * ym) + xm) == 0 && *(m_ResultImg + (col * ym) + x) != 0) sp1++;

			hv = 0;
			if (sp1 == 2) {
				if ((*(m_ResultImg + (col * ym) + x) & *(m_ResultImg + (col * y) + xp)) != 0) hv++;
				else if ((*(m_ResultImg + (col * y) + xp) & *(m_ResultImg + (col * yp) + x)) != 0) hv++;
				else if ((*(m_ResultImg + (col * yp) + x) & *(m_ResultImg + (col * y) + xm)) != 0) hv++;
				else if ((*(m_ResultImg + (col * y) + xm) & *(m_ResultImg + (col * ym) + x)) != 0) hv++;

				if (hv == 1) *(m_ResultImg + (col * y) + x) = 0;
			}
			else if (sp1 == 3) {
				if ((*(m_ResultImg + (col * ym) + x) & *(m_ResultImg + (col * y) + xm) & *(m_ResultImg + (col * y) + xp)) != 0) hv++;
				else if ((*(m_ResultImg + (col * yp) + x) & *(m_ResultImg + (col * y) + xm) & *(m_ResultImg + (col * y) + xp)) != 0) hv++;
				else if ((*(m_ResultImg + (col * ym) + x) & *(m_ResultImg + (col * yp) + x) & *(m_ResultImg + (col * y) + xm)) != 0) hv++;
				else if ((*(m_ResultImg + (col * ym) + x) & *(m_ResultImg + (col * yp) + x) & *(m_ResultImg + (col * y) + xp)) != 0) hv++;

				if (hv == 1) *(m_ResultImg + (col * y) + x) = 0;
			}
		}
	}

	dst = cv::Mat::zeros(src.size(), CV_8UC1);

	for (y = 0; y < row; y++) {
		for (x = 0; x < col; x++) {
			dst.at<uchar>(y, x) = *(m_ResultImg + (col * y) + x);
		}
	}

	free(m_BufImage);
	free(m_ResultImg);
}

//GuidedLine Edge Enhancemnet
EXTERNC AFX_API_EXPORT BOOL	GuidedEdgeEnhance(const Mat Src, Mat& O_Des)
{
	Mat I = Src.clone();
	I.convertTo(I, CV_32F, 1.0 / 255.0);

	Mat Test = I.clone();
	cv::Mat p = I;
	int radius = 21;
	double eps = 0.044 * 0.044;
	cv::Mat q = guidedFilter(I, p, radius, eps);
	cv::Mat enhance = (I - q) * 5 + q;
	double min, max;
	cv::minMaxLoc(enhance, &min, &max);
	Mat guided;
	enhance.convertTo(O_Des, CV_8U, 255.0 / (max - min), -255.0 * min / (max - min));
	return TRUE;
}

//Detail Enhancement
EXTERNC AFX_API_EXPORT BOOL DetailEnhancement(Mat& src, Mat& dst, float factor)
{
	float sigma_s = 10;
	float sigma_r = 0.55f;

	Mat src2 = src.clone();
	Mat src3;
	cv::cvtColor(src2, src3, CV_GRAY2RGB);

	Mat dst_ = Mat::zeros(src.size(), CV_8UC3);

	int h = src.size().height;
	int w = src.size().width;
	//float factor = 1.4f;

	Mat img = Mat(src.size(), CV_32FC3);
	src3.convertTo(img, CV_32FC3, 1.0 / 255.0);

	Mat res = Mat(h, w, CV_32FC1);
	dst_.convertTo(res, CV_32FC3, 1.0 / 255.0);

	Mat result = Mat(img.size(), CV_32FC3);
	Mat lab = Mat(img.size(), CV_32FC3);
	vector <Mat> lab_channel;

	cv::cvtColor(img, lab, COLOR_BGR2Lab);
	split(lab, lab_channel);

	Mat L = Mat(img.size(), CV_32FC1);
	lab_channel[0].convertTo(L, CV_32FC1, 1.0 / 255.0);

	DomainFilter obj;
	obj.filter(L, res, sigma_s, sigma_r, 1);

	Mat detail = Mat(h, w, CV_32FC1);
	detail = L - res;
	cv::multiply(detail, factor, detail);
	L = res + detail;
	L.convertTo(lab_channel[0], CV_32FC1, 255);
	merge(lab_channel, lab);
	cv::cvtColor(lab, result, COLOR_Lab2BGR);
	result.convertTo(dst_, CV_8UC3, 255);
	cv::cvtColor(dst_, dst, CV_BGR2GRAY);

	return TRUE;
}

EXTERNC AFX_API_EXPORT void Threshold_MultiOtsu(Mat src, Mat& dst)
{
	if (src.channels() > 1)
		return;

	int* hist = new int[256];
	for (int n = 0; n < 256; n++)
		hist[n] = 0;
	int totalCount = 0;

	// hist
	for (int i = 0; i < src.cols; i++) {
		for (int j = 0; j < src.rows; j++) {
			if (src.at<uchar>(j, i) > 0) {
				unsigned char val = src.at<uchar>(j, i);
				hist[val]++;
				totalCount++;
			}
		}
	}

	double W0K, W1K, W2K, M0, M1, M2, currVarB, optimalThresh1, optimalThresh2, maxBetweenVar, M0K, M1K, M2K, MT;

	optimalThresh1 = 0.0;
	optimalThresh2 = 0.0;
	W0K = 0.0;
	W1K = 0.0;
	M0K = 0.0;
	M1K = 0.0;
	MT = 0.0;
	maxBetweenVar = 0.0;

	for (int k = 0; k <= 255; k++) {
		MT += k * (hist[k] / (double)totalCount);
	}

	for (int t1 = 0; t1 <= 255; t1++) {
		W0K += hist[t1] / (double)totalCount; //Pi
		M0K += t1 * (hist[t1] / (double)totalCount); //i * Pi
		M0 = M0K / W0K; //(i * Pi)/Pi

		W1K = 0;
		M1K = 0;
		for (int t2 = t1 + 1; t2 <= 255; t2++) {
			W1K += hist[t2] / (double)totalCount; //Pi
			M1K += t2 * (hist[t2] / (double)totalCount); //i * Pi
			M1 = M1K / W1K; //(i * Pi)/Pi

			W2K = 1 - (W0K + W1K);
			M2K = MT - (M0K + M1K);

			if (W2K <= 0) break;
			M2 = M2K / W2K;
			currVarB = W0K * (M0 - MT) * (M0 - MT) + W1K * (M1 - MT) * (M1 - MT) + W2K * (M2 - MT) * (M2 - MT);

			if (maxBetweenVar < currVarB) {
				maxBetweenVar = currVarB;
				optimalThresh1 = t1;
				optimalThresh2 = t2;
			}
		}
	}

	for (int j = 0; j < src.cols; j++)
	{
		for (int i = 0; i < src.rows; i++)
		{
			uchar val = src.at<uchar>(i, j);

			if (val <= optimalThresh1)
				dst.at<uchar>(i, j) = 0;
			else if (val >= optimalThresh2)
				dst.at<uchar>(i, j) = 128;
			else
				dst.at<uchar>(i, j) = 255;
		}
	}

	delete[] hist;
}

//返回最大的Blob
EXTERNC AFX_API_EXPORT void FindBiggestBlob(cv::Mat& src, cv::Mat& dst)
{
	Mat prc;
	src.copyTo(prc);

	dst = Mat::zeros(prc.size(), prc.type());

	int largest_area = 0;
	int largest_contour_index = 0;

	vector< vector<Point> > contours; // Vector for storing contour
	vector<Vec4i> hierarchy;

	findContours(prc, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE); // Find the contours in the image
	cv::Rect bounding_rect;
	for (int i = 0; i < contours.size(); i++) {// iterate through each contour. 
		double a = contourArea(contours[i], false);  //  Find the area of contour
		if (a > largest_area) {
			largest_area = (int)a;
			largest_contour_index = i;              //Store the index of largest contour
			bounding_rect = boundingRect(contours[i]); // Find the bounding rectangle for biggest contour
		}
	}

	drawContours(dst, contours, largest_contour_index, Scalar(255), CV_FILLED, 8, hierarchy); // Draw the largest contour using previously stored index.
}

EXTERNC AFX_API_EXPORT void FindBiggestContour(cv::Mat src, vector<vector<cv::Point>>& ptBiggest)
{
	Mat prc;
	src.copyTo(prc);

	int largest_area = 0;
	int largest_contour_index = 0;

	vector< vector<Point> > contours; // Vector for storing contour
	vector<Vec4i> hierarchy;

	findContours(prc, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE); // Find the contours in the image
	cv::Rect bounding_rect;
	for (int i = 0; i < contours.size(); i++) {// iterate through each contour. 
		double a = contourArea(contours[i], false);  //  Find the area of contour
		if (a > largest_area) {
			largest_area = (int)a;
			largest_contour_index = i;              //Store the index of largest contour
			bounding_rect = boundingRect(contours[i]); // Find the bounding rectangle for biggest contour
		}
	}

	if (contours.size() < 1)
		return;

	ptBiggest.push_back(contours[largest_contour_index]);
}

//返回nSelectInx第二个那么大的Blob
EXTERNC AFX_API_EXPORT void SelectBiggestBlob(cv::Mat& src, cv::Mat& dst, int nSelectInx)
{
	dst = Mat::zeros(src.size(), src.type());
	int largest_area = 0;
	int largest_contour_index = 0;

	vector< vector<Point> > contours; // Vector for storing contour
	vector<Vec4i> hierarchy;
	vector<double> dArea;
	findContours(src, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE); // Find the contours in the image
	cv::Rect bounding_rect;
	for (int i = 0; i < contours.size(); i++) {// iterate through each contour. 
		double a = contourArea(contours[i], false);  //  Find the area of contour
		dArea.push_back(a);
	}
	sort(dArea.begin(), dArea.end());

	for (int i = 0; i < contours.size(); i++) {// iterate through each contour. 
		double a = contourArea(contours[i], false);  //  Find the area of contour
		int nSelect = nSelectInx;
		if (dArea.size() <= nSelect || nSelect <= 0)
			nSelect = 1;

		if (dArea[dArea.size() - nSelect] == a)
		{
			largest_contour_index = i;
		}
	}

	drawContours(dst, contours, largest_contour_index, Scalar(255), CV_FILLED, 8, hierarchy); // Draw the largest contour using previously stored index.

}

//用最小自乘法求直线方程的斜率
EXTERNC AFX_API_EXPORT long MethodOfLeastSquares(vector<cv::Point> pt, double& nA)
{
	if (pt.size() < 10)
		return E_ERROR_CODE_FALSE;
	double nA_Up = 0;
	double nA_Down = 0;
	double dSigmaX = 0;
	double dSigmaX_2 = 0;
	double dSigmaY = 0;
	double dSigmaXY = 0;
	double dN = (double)pt.size();

	//Y=Ax+B的斜率计算式可以为0。
	//X=换成Ay+B计算,以后+90度
	for (int i = 0; i < dN; i++)
	{
		dSigmaX += pt[i].y;
		dSigmaX_2 += pt[i].y * pt[i].y;
		dSigmaY += pt[i].x;
		dSigmaXY += pt[i].x * pt[i].y;
	}

	nA_Up = dN * dSigmaXY - dSigmaX * dSigmaY;
	nA_Down = dN * dSigmaX_2 - dSigmaX * dSigmaX;
	vector<cv::Point>().swap(pt);
	if (nA_Down == 0)
		nA = 0;
	else
		nA = nA_Up / nA_Down;

	return E_ERROR_CODE_TRUE;
}

EXTERNC AFX_API_EXPORT void MinimumFilter(cv::Mat src, cv::Mat& dst, int nMaskSize)
{
	Mat prc;
	src.copyTo(prc);

	if (nMaskSize % 2 == 0)
		nMaskSize++;

	int nCols = src.cols;
	int nRows = src.rows;
	int nStep = src.step;

	int nMaskX_ST, nMaskY_ST, nMask_ST;

	nMask_ST = nMaskSize / 2 * -1;
	int nMin;

	uchar* ucSrcdata;
	uchar* ucDstdata;

	for (int nY = 0; nY < nRows; nY++)
	{
		if (nY + nMask_ST < 0 || nY + abs(nMask_ST) > nRows - 1)
			continue;
		ucDstdata = prc.ptr(nY);
		for (int nX = 0; nX < nCols; nX++)
		{
			if (nX + nMask_ST < 0 || nX + nMask_ST > nCols - 1)
				continue;

			nMin = 255;

			nMaskY_ST = nMask_ST;
			for (int nMaskY = 0; nMaskY < nMaskSize; nMaskY++)
			{
				int nY_ = nY + nMaskY_ST;
				ucSrcdata = src.ptr(nY_);

				nMaskX_ST = nMask_ST;
				for (int nMaskX = 0; nMaskX < nMaskSize; nMaskX++)
				{
					int nX_ = nX + nMaskX_ST;
					if ((int)ucSrcdata[nX_] < nMin)
						nMin = (int)ucSrcdata[nX_];
					nMaskX_ST++;
				}
				nMaskY_ST++;
			}
			ucDstdata[nX] = (uchar)nMin;
		}
	}

	prc.copyTo(dst);
}

EXTERNC AFX_API_EXPORT void MaximumFilter(cv::Mat src, cv::Mat& dst, int nMaskSize)
{
	Mat prc;
	src.copyTo(prc);

	if (nMaskSize % 2 == 0)
		nMaskSize++;

	int nCols = src.cols;
	int nRows = src.rows;
	int nStep = src.step;

	int nMaskX_ST, nMaskY_ST, nMask_ST;

	nMask_ST = nMaskSize / 2 * -1;
	int nMax;

	uchar* ucSrcdata;
	uchar* ucDstdata;

	for (int nY = 0; nY < nRows; nY++)
	{
		if (nY + nMask_ST < 0 || nY + abs(nMask_ST) > nRows - 1)
			continue;
		ucDstdata = prc.ptr(nY);
		for (int nX = 0; nX < nCols; nX++)
		{
			if (nX + nMask_ST < 0 || nX + nMask_ST > nCols - 1)
				continue;

			nMax = 0;

			nMaskY_ST = nMask_ST;
			for (int nMaskY = 0; nMaskY < nMaskSize; nMaskY++)
			{
				int nY_ = nY + nMaskY_ST;
				ucSrcdata = src.ptr(nY_);

				nMaskX_ST = nMask_ST;
				for (int nMaskX = 0; nMaskX < nMaskSize; nMaskX++)
				{
					int nX_ = nX + nMaskX_ST;
					if ((int)ucSrcdata[nX_] > nMax)
						nMax = (int)ucSrcdata[nX_];
					nMaskX_ST++;
				}
				nMaskY_ST++;
			}
			ucDstdata[nX] = (uchar)nMax;
		}
	}

	prc.copyTo(dst);
}
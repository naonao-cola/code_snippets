#include "StdAfx.h"
#include "Blob_Sub.h"


CBlob_Sub::CBlob_Sub(void)
{
}


CBlob_Sub::~CBlob_Sub(void)
{
}


BOOL CBlob_Sub::FindBiggestBlob(cv::Mat& src, cv::Mat& dst)
{
	if( src.empty() )		return FALSE;
	int largest_area=0;
	int largest_contour_index=0;

	vector< vector<Point> > contours; // Vector for storing contour
	vector<Vec4i> hierarchy;

	findContours( src, contours, hierarchy,CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE ); // Find the contours in the image
	for( int i = 0; i< contours.size(); i++ ) {// iterate through each contour. 
		double a=contourArea( contours[i],false);  //  Find the area of contour
		if(a>largest_area){
			largest_area=a;
			largest_contour_index=i;              //Store the index of largest contour
		}
	}

	drawContours( dst, contours, largest_contour_index, Scalar(255), CV_FILLED, 8, hierarchy ); // Draw the largest contour using previously stored index.

	return TRUE;
}

BOOL CBlob_Sub::GetBiggestBlobMinAreaRect(cv::Mat& src, cv::RotatedRect& rtRect)
{
	if( src.empty())		return FALSE;

	int largest_area=0;
	int largest_contour_index=0;

	vector< vector<Point> > contours; // Vector for storing contour
	vector<Vec4i> hierarchy;

	findContours( src, contours, hierarchy,CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE ); // Find the contours in the image
	cv::Rect bounding_rect;
	for( int i = 0; i< contours.size(); i++ ) {// iterate through each contour. 
		double a=contourArea( contours[i],false);  //  Find the area of contour
		if(a>largest_area){
			largest_area=a;
			largest_contour_index=i;              //Store the index of largest contour
		}
	}

	rtRect = minAreaRect(contours[largest_contour_index]);

	return TRUE;
}

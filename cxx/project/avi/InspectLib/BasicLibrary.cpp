#include "StdAfx.h"
#include "BasicLibrary.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

static cv::Mat boxfilter(const cv::Mat &I, int r)
{
	cv::Mat result;
	cv::blur(I, result, cv::Size(r, r));
	return result;
}

static cv::Mat convertTo(const cv::Mat &mat, int depth)
{
	if (mat.depth() == depth)
		return mat;

	cv::Mat result;
	mat.convertTo(result, depth);
	return result;
}

cv::Mat GuidedFilterImpl::filter(const cv::Mat &p, int depth)
{
	cv::Mat p2 = convertTo(p, Idepth);

	cv::Mat result;
	if (p.channels() == 1)
	{
		result = filterSingleChannel(p2);
	}
	else
	{

	}

	return convertTo(result, depth == -1 ? p.depth() : depth);
}

GuidedFilterMono::GuidedFilterMono(const cv::Mat &origI, int r, double eps) : r(r), eps(eps)
{
	if (origI.depth() == CV_32F || origI.depth() == CV_64F)
		I = origI.clone();
	else
		I = convertTo(origI, CV_32F);

	Idepth = I.depth();

	mean_I = boxfilter(I, r);
	cv::Mat mean_II = boxfilter(I.mul(I), r);
	var_I = mean_II - mean_I.mul(mean_I);
}

cv::Mat GuidedFilterMono::filterSingleChannel(const cv::Mat &p) const
{
	cv::Mat mean_p = boxfilter(p, r);
	cv::Mat mean_Ip = boxfilter(I.mul(p), r);
	cv::Mat cov_Ip = mean_Ip - mean_I.mul(mean_p); // this is the covariance of (I, p) in each local patch.

	cv::Mat a = cov_Ip / (var_I + eps); // (5) in the paper, Kaiming He
	cv::Mat b = mean_p - a.mul(mean_I); // (6) in the paper,  Kaiming He	 

	cv::Mat mean_a = boxfilter(a, r);
	cv::Mat mean_b = boxfilter(b, r);

	return mean_a.mul(I) + mean_b;		// (8) in the paper,  Kaiming He	
}

GuidedFilter::GuidedFilter(const cv::Mat &I, int r, double eps)
{
	CV_Assert(I.channels() == 1 || I.channels() == 3);

	if (I.channels() == 1)
		impl_ = new GuidedFilterMono(I, 2 * r + 1, eps);   
}

GuidedFilter::~GuidedFilter()
{
	delete impl_;
}

cv::Mat GuidedFilter::filter(const cv::Mat &p, int depth) const
{
	return impl_->filter(p, depth);
}

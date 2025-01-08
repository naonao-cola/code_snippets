#pragma once

#include <iostream> 
#include <stdlib.h> 
#include <limits> 
#include "math.h" 
#include "Define.h"

using namespace std;
using namespace cv;

class DomainFilter
{
public:
	DomainFilter(void);
	~DomainFilter(void);
	double myinf;
	Mat ct_H, ct_V, horiz, vert, O, O_t, lower_idx, upper_idx;
	void init(const Mat& img, int flags, float sigma_s, float sigma_r);
	void getGradientx(const Mat& img, Mat& gx);
	void getGradienty(const Mat& img, Mat& gy);
	void diffx(const Mat& img, Mat& temp);
	void diffy(const Mat& img, Mat& temp);
	void find_magnitude(Mat& img, Mat& mag);
	void compute_boxfilter(Mat& output, Mat& hz, Mat& psketch, float radius);
	void compute_Rfilter(Mat& O, Mat& horiz, float sigma_h);
	void compute_NCfilter(Mat& O, Mat& horiz, Mat& psketch, float radius);
	void filter(const Mat& img, Mat& res, float sigma_s, float sigma_r, int flags);
	void pencil_sketch(const Mat& img, Mat& sketch, Mat& color_res, float sigma_s, float sigma_r, float shade_factor);
	void Depth_of_field(const Mat& img, Mat& img1, float sigma_s, float sigma_r);
};


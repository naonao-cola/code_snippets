#include "StdAfx.h"
#include "DomainFilter.h"

DomainFilter::DomainFilter(void)
{
	myinf = std::numeric_limits<double>::infinity();
}

DomainFilter::~DomainFilter(void)
{

}

void DomainFilter::diffx(const Mat& img, Mat& temp)
{
	int channel = img.channels();
	for (int i = 0; i < img.size().height; i++)
		for (int j = 0; j < img.size().width - 1; j++)
		{
			for (int c = 0; c < channel; c++)
			{
				temp.at<float>(i, j * channel + c) =
					img.at<float>(i, (j + 1) * channel + c) - img.at<float>(i, j * channel + c);
			}
		}
}

void DomainFilter::diffy(const Mat& img, Mat& temp)
{
	int channel = img.channels();
	for (int i = 0; i < img.size().height - 1; i++)
		for (int j = 0; j < img.size().width; j++)
		{
			for (int c = 0; c < channel; c++)
			{
				temp.at<float>(i, j * channel + c) =
					img.at<float>((i + 1), j * channel + c) - img.at<float>(i, j * channel + c);
			}
		}
}

void DomainFilter::getGradientx(const Mat& img, Mat& gx)
{
	int w = img.cols;
	int h = img.rows;
	int channel = img.channels();

	for (int i = 0; i < h; i++)
		for (int j = 0; j < w; j++)
			for (int c = 0; c < channel; ++c)
			{
				gx.at<float>(i, j * channel + c) =
					img.at<float>(i, (j + 1) * channel + c) - img.at<float>(i, j * channel + c);
			}
}

void DomainFilter::getGradienty(const Mat& img, Mat& gy)
{
	int w = img.cols;
	int h = img.rows;
	int channel = img.channels();

	for (int i = 0; i < h; i++)
		for (int j = 0; j < w; j++)
			for (int c = 0; c < channel; ++c)
			{
				gy.at<float>(i, j * channel + c) =
					img.at<float>(i + 1, j * channel + c) - img.at<float>(i, j * channel + c);
			}
}

void DomainFilter::find_magnitude(Mat& img, Mat& mag)
{
	int h = img.rows;
	int w = img.cols;

	vector <Mat> planes;
	split(img, planes);

	Mat magXR = Mat(h, w, CV_32FC1);
	Mat magYR = Mat(h, w, CV_32FC1);

	Mat magXG = Mat(h, w, CV_32FC1);
	Mat magYG = Mat(h, w, CV_32FC1);

	Mat magXB = Mat(h, w, CV_32FC1);
	Mat magYB = Mat(h, w, CV_32FC1);

	Sobel(planes[0], magXR, CV_32FC1, 1, 0, 3);
	Sobel(planes[0], magYR, CV_32FC1, 0, 1, 3);

	Sobel(planes[1], magXG, CV_32FC1, 1, 0, 3);
	Sobel(planes[1], magYG, CV_32FC1, 0, 1, 3);

	Sobel(planes[2], magXB, CV_32FC1, 1, 0, 3);
	Sobel(planes[2], magYB, CV_32FC1, 0, 1, 3);

	Mat mag1 = Mat(h, w, CV_32FC1);
	Mat mag2 = Mat(h, w, CV_32FC1);
	Mat mag3 = Mat(h, w, CV_32FC1);

	cv::magnitude(magXR, magYR, mag1);
	cv::magnitude(magXG, magYG, mag2);
	cv::magnitude(magXB, magYB, mag3);

	mag = mag1 + mag2 + mag3;
	mag = 1.0f - mag;
}

void DomainFilter::compute_Rfilter(Mat& output, Mat& hz, float sigma_h)
{
	int h = output.rows;
	int w = output.cols;
	int channel = output.channels();

	float a = (float)exp((-1.0 * sqrt(2.0)) / sigma_h);

	Mat temp = Mat(h, w, CV_32FC3);
	output.copyTo(temp);
	Mat V = Mat(h, w, CV_32FC1);

	for (int i = 0; i < h; i++)
		for (int j = 0; j < w; j++)
			V.at<float>(i, j) = pow(a, hz.at<float>(i, j));

	for (int i = 0; i < h; i++)
	{
		for (int j = 1; j < w; j++)
		{
			for (int c = 0; c < channel; c++)
			{
				temp.at<float>(i, j * channel + c) = temp.at<float>(i, j * channel + c) +
					(temp.at<float>(i, (j - 1) * channel + c) - temp.at<float>(i, j * channel + c)) * V.at<float>(i, j);
			}
		}
	}

	for (int i = 0; i < h; i++)
	{
		for (int j = w - 2; j >= 0; j--)
		{
			for (int c = 0; c < channel; c++)
			{
				temp.at<float>(i, j * channel + c) = temp.at<float>(i, j * channel + c) +
					(temp.at<float>(i, (j + 1) * channel + c) - temp.at<float>(i, j * channel + c)) * V.at<float>(i, j + 1);
			}
		}
	}

	temp.copyTo(output);
}

void DomainFilter::compute_boxfilter(Mat& output, Mat& hz, Mat& psketch, float radius)
{
	int h = output.rows;
	int w = output.cols;
	Mat lower_pos = Mat(h, w, CV_32FC1);
	Mat upper_pos = Mat(h, w, CV_32FC1);

	lower_pos = hz - radius;
	upper_pos = hz + radius;

	lower_idx = Mat::zeros(h, w, CV_32FC1);
	upper_idx = Mat::zeros(h, w, CV_32FC1);

	Mat domain_row = Mat::zeros(1, w + 1, CV_32FC1);

	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
			domain_row.at<float>(0, j) = hz.at<float>(i, j);
		domain_row.at<float>(0, w) = (float)myinf;

		Mat lower_pos_row = Mat::zeros(1, w, CV_32FC1);
		Mat upper_pos_row = Mat::zeros(1, w, CV_32FC1);

		for (int j = 0; j < w; j++)
		{
			lower_pos_row.at<float>(0, j) = lower_pos.at<float>(i, j);
			upper_pos_row.at<float>(0, j) = upper_pos.at<float>(i, j);
		}

		Mat temp_lower_idx = Mat::zeros(1, w, CV_32FC1);
		Mat temp_upper_idx = Mat::zeros(1, w, CV_32FC1);

		for (int j = 0; j < w; j++)
		{
			if (domain_row.at<float>(0, j) > lower_pos_row.at<float>(0, 0))
			{
				temp_lower_idx.at<float>(0, 0) = (float)j;
				break;
			}
		}
		for (int j = 0; j < w; j++)
		{
			if (domain_row.at<float>(0, j) > upper_pos_row.at<float>(0, 0))
			{
				temp_upper_idx.at<float>(0, 0) = (float)j;
				break;
			}
		}

		int temp = 0;
		for (int j = 1; j < w; j++)
		{
			int count = 0;
			for (int k = (int)temp_lower_idx.at<float>(0, j - 1); k < w + 1; k++)
			{
				if (domain_row.at<float>(0, k) > lower_pos_row.at<float>(0, j))
				{
					temp = count;
					break;
				}
				count++;
			}

			temp_lower_idx.at<float>(0, j) = temp_lower_idx.at<float>(0, j - 1) + temp;

			count = 0;
			for (int k = (int)temp_upper_idx.at<float>(0, j - 1); k < w + 1; k++)
			{
				if (domain_row.at<float>(0, k) > upper_pos_row.at<float>(0, j))
				{
					temp = count;
					break;
				}
				count++;
			}
			temp_upper_idx.at<float>(0, j) = temp_upper_idx.at<float>(0, j - 1) + temp;
		}

		for (int j = 0; j < w; j++)
		{
			lower_idx.at<float>(i, j) = temp_lower_idx.at<float>(0, j) + 1;
			upper_idx.at<float>(i, j) = temp_upper_idx.at<float>(0, j) + 1;
		}
	}
	psketch = upper_idx - lower_idx;
}

void DomainFilter::compute_NCfilter(Mat& output, Mat& hz, Mat& psketch, float radius)
{
	int h = output.rows;
	int w = output.cols;
	int channel = output.channels();

	compute_boxfilter(output, hz, psketch, radius);
	Mat box_filter = Mat::zeros(h, w + 1, CV_32FC3);

	for (int i = 0; i < h; i++)
	{
		box_filter.at<float>(i, 1 * channel + 0) = output.at<float>(i, 0 * channel + 0);
		box_filter.at<float>(i, 1 * channel + 1) = output.at<float>(i, 0 * channel + 1);
		box_filter.at<float>(i, 1 * channel + 2) = output.at<float>(i, 0 * channel + 2);
		for (int j = 2; j < w + 1; j++)
		{
			for (int c = 0; c < channel; c++)
				box_filter.at<float>(i, j * channel + c) = output.at<float>(i, (j - 1) * channel + c) + box_filter.at<float>(i, (j - 1) * channel + c);
		}
	}

	Mat indices = Mat::zeros(h, w, CV_32FC1);
	Mat final = Mat::zeros(h, w, CV_32FC3);

	for (int i = 0; i < h; i++)
		for (int j = 0; j < w; j++)
			indices.at<float>(i, j) = (float)i + 1;

	Mat a = Mat::zeros(h, w, CV_32FC1);
	Mat b = Mat::zeros(h, w, CV_32FC1);

	// Compute the box filter using a summed area table. 
	for (int c = 0; c < channel; c++)
	{
		Mat flag = Mat::ones(h, w, CV_32FC1);
		cv::multiply(flag, c + 1, flag);

		Mat temp1, temp2;
		cv::multiply(flag - 1, h * (w + 1), temp1);
		cv::multiply(lower_idx - 1, h, temp2);
		a = temp1 + temp2 + indices;

		cv::multiply(flag - 1, h * (w + 1), temp1);
		cv::multiply(upper_idx - 1, h, temp2);
		b = temp1 + temp2 + indices;

		int p, q, r, rem;
		int p1, q1, r1, rem1;

		// Calculating indices 
		for (int i = 0; i < h; i++)
		{
			for (int j = 0; j < w; j++)
			{
				r = (int)b.at<float>(i, j) / (h * (w + 1));
				rem = (int)b.at<float>(i, j) - r * h * (w + 1);
				q = rem / h;
				p = rem - q * h;
				if (q == 0)
				{
					p = h;
					q = w;
					r = r - 1;
				}
				if (p == 0)
				{
					p = h;
					q = q - 1;
				}

				r1 = (int)a.at<float>(i, j) / (h * (w + 1));
				rem1 = (int)a.at<float>(i, j) - r1 * h * (w + 1);
				q1 = rem1 / h;
				p1 = rem1 - q1 * h;
				if (p1 == 0)
				{
					p1 = h;
					q1 = q1 - 1;
				}
				final.at<float>(i, j * channel + 2 - c) = (box_filter.at<float>(p - 1, q * channel + (2 - r)) - box_filter.at<float>(p1 - 1, q1 * channel + (2 - r1)))
					/ (upper_idx.at<float>(i, j) - lower_idx.at<float>(i, j));
			}
		}
	}

	final.copyTo(output);
}

void DomainFilter::init(const Mat& img, int flags, float sigma_s, float sigma_r)
{
	int h = img.size().height;
	int w = img.size().width;
	int channel = img.channels();

	////////////////////////////////////     horizontal and vertical partial derivatives ///////////////////////////////// 
	Mat derivx = Mat::zeros(h, w - 1, CV_32FC3);
	Mat derivy = Mat::zeros(h - 1, w, CV_32FC3);

	diffx(img, derivx);
	diffy(img, derivy);

	Mat distx = Mat::zeros(h, w, CV_32FC1);
	Mat disty = Mat::zeros(h, w, CV_32FC1);
	//////////////////////// Compute the l1-norm distance of neighbor pixels //////////////////////////////////////////////// 

	for (int i = 0; i < h; i++)
		for (int j = 0, k = 1; j < w - 1; j++, k++)
			for (int c = 0; c < channel; c++)
			{
				distx.at<float>(i, k) =
					distx.at<float>(i, k) + abs(derivx.at<float>(i, j * channel + c));
			}

	for (int i = 0, k = 1; i < h - 1; i++, k++)
		for (int j = 0; j < w; j++)
			for (int c = 0; c < channel; c++)
			{
				disty.at<float>(k, j) =
					disty.at<float>(k, j) + abs(derivy.at<float>(i, j * channel + c));
			}
	////////////////////// Compute the derivatives of the horizontal and vertical domain transforms. ///////////////////////////// 
	horiz = Mat(h, w, CV_32FC1);
	vert = Mat(h, w, CV_32FC1);

	Mat final = Mat(h, w, CV_32FC3);
	Mat tempx, tempy;
	cv::multiply(distx, sigma_s / sigma_r, tempx);
	cv::multiply(disty, sigma_s / sigma_r, tempy);

	horiz = 1.0f + tempx;
	vert = 1.0f + tempy;

	O = Mat(h, w, CV_32FC3);
	img.copyTo(O);

	O_t = Mat(w, h, CV_32FC3);

	if (flags == 2)
	{
		ct_H = Mat(h, w, CV_32FC1);
		ct_V = Mat(h, w, CV_32FC1);

		for (int i = 0; i < h; i++)
		{
			ct_H.at<float>(i, 0) = horiz.at<float>(i, 0);
			for (int j = 1; j < w; j++)
			{
				ct_H.at<float>(i, j) = horiz.at<float>(i, j) + ct_H.at<float>(i, j - 1);
			}
		}

		for (int j = 0; j < w; j++)
		{
			ct_V.at<float>(0, j) = vert.at<float>(0, j);
			for (int i = 1; i < h; i++)
			{
				ct_V.at<float>(i, j) = vert.at<float>(i, j) + ct_V.at<float>(i - 1, j);
			}
		}
	}
}

void DomainFilter::filter(const Mat& img, Mat& res, float sigma_s = 60, float sigma_r = 0.4, int flags = 1)
{
	int no_of_iter = 3;
	int h = img.size().height;
	int w = img.size().width;
	float sigma_h = sigma_s;

	init(img, flags, sigma_s, sigma_r);

	if (flags == 1)
	{
		Mat vert_t = vert.t();
		for (int i = 0; i < no_of_iter; i++)
		{
			sigma_h = (float)(sigma_s * sqrt(3.0) * pow(2.0, (no_of_iter - (i + 1))) / sqrt(pow(4.0, no_of_iter) - 1));

			compute_Rfilter(O, horiz, sigma_h);
			O_t = O.t();

			compute_Rfilter(O_t, vert_t, sigma_h);
			O = O_t.t();
		}
	}
	else if (flags == 2)
	{
		Mat vert_t = ct_V.t();
		Mat temp = Mat(h, w, CV_32FC1);
		Mat temp1 = Mat(w, h, CV_32FC1);

		float radius;
		for (int i = 0; i < no_of_iter; i++)
		{
			sigma_h = (float)(sigma_s * sqrt(3.0) * pow(2.0, (no_of_iter - (i + 1))) / sqrt(pow(4.0, no_of_iter) - 1));
			radius = (float)sqrt(3.0) * sigma_h;
			compute_NCfilter(O, ct_H, temp, radius);
			O_t = O.t();
			compute_NCfilter(O_t, vert_t, temp1, radius);
			O = O_t.t();
		}
	}
	res = O.clone();
}

void DomainFilter::pencil_sketch(const Mat& img, Mat& sketch, Mat& color_res, float sigma_s, float sigma_r, float shade_factor)
{
	int no_of_iter = 3;
	init(img, 2, sigma_s, sigma_r);
	int h = img.size().height;
	int w = img.size().width;

	/////////////////////// convert to YCBCR model for color pencil drawing ////////////////////////////////////////////////////// 
	Mat color_sketch = Mat(h, w, CV_32FC3);
	cv::cvtColor(img, color_sketch, COLOR_BGR2YCrCb);

	vector <Mat> YUV_channel;
	Mat vert_t = ct_V.t();

	float sigma_h = sigma_s;
	Mat penx = Mat(h, w, CV_32FC1);
	Mat pen_res = Mat::zeros(h, w, CV_32FC1);
	Mat peny = Mat(w, h, CV_32FC1);
	Mat peny_t;

	float radius;
	for (int i = 0; i < no_of_iter; i++)
	{
		sigma_h = (float)(sigma_s * sqrt(3.0) * pow(2.0, (no_of_iter - (i + 1))) / sqrt(pow(4.0, no_of_iter) - 1));
		radius = (float)sqrt(3.0) * sigma_h;

		compute_boxfilter(O, ct_H, penx, radius);
		O_t = O.t();

		compute_boxfilter(O_t, vert_t, peny, radius);
		O = O_t.t();

		peny_t = peny.t();

		for (int k = 0; k < h; k++)
			for (int j = 0; j < w; j++)
				pen_res.at<float>(k, j) = (shade_factor * (penx.at<float>(k, j) + peny_t.at<float>(k, j)));

		if (i == 0)
		{
			sketch = pen_res.clone();
			split(color_sketch, YUV_channel);
			pen_res.copyTo(YUV_channel[0]);
			merge(YUV_channel, color_sketch);
			cv::cvtColor(color_sketch, color_res, COLOR_YCrCb2BGR);
		}
	}
}

#pragma once
/*--------------------------------------------------------------------------------------------------
* modify_author:
* modify_time:      2021/10/27 13:38:26
* modify_content:
* modify_reference:
* modify_other:
* modify_version:
--------------------------------------------------------------------------------------------------*/
#ifndef __IMG_FEATURE_H__
#define __IMG_FEATURE_H__
#pragma warning(disable:4244)
#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#else
#include <sys/time.h>
#endif
#include <vector>
#include <iostream>
#include <algorithm>

#include "opencv2/opencv.hpp"


namespace nao {
	namespace img {
		namespace feature {
			class HogTransform {
			public:

				HogTransform(std::vector<cv::Mat> images, int num_cells, int cell_size, int num_bins, cv::Size train_size, int vlhog_variant = 1) :
					_images(images), _num_cells(num_cells), _cell_size(cell_size), _num_bins(num_bins), _train_size(train_size), _vlhog_variant(vlhog_variant) {};


				cv::Mat operator()();
			private:
				std::vector<cv::Mat> _images;
				int _vlhog_variant;
				cv::Size _train_size;
				cv::Mat _hog_descriptors;
				int _num_cells;
				int _cell_size;
				int _num_bins;
			};//class HogTransform

			class LBPTransform {
			public:

				LBPTransform(std::vector<cv::Mat> images, cv::Size train_size) :_images(images), _train_size(train_size) {};
				~LBPTransform() = default;
				cv::Mat operator()();
			private:
				std::vector<cv::Mat> _images;
				cv::Size _train_size;
				cv::Mat _lbp_descriptors;
			};// class LBPTransform
		}//namespace feature
	}//namespace img
}//namespace nao
#endif  //__IMG_FEATURE_H__
/*----------------------------------------------------------------------------- (C) COPYRIGHT LEI *****END OF FILE------------------------------------------------------------------------------*/
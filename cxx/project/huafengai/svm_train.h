#pragma once
/*--------------------------------------------------------------------------------------------------
	*  @Copyright (c) , All rights reserved.
	*  @file:       svm_train.h
	*  @version:    ver 1.0
	*  @author:
	*  @brief:
	*  @change:
	*  @email: 	1319144981@qq.com
	*  Date             Version    Changed By      Changes
	*  2021/10/27 9:21:33    1.0                   create
--------------------------------------------------------------------------------------------------*/
#ifndef __SVM_TRAIN_H__
#define __SVM_TRAIN_H__
#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#pragma warning(disable:4244)

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#include <direct.h>
#else
#include <sys/time.h>
#endif

#include <utility>
#include <iostream>
#include <algorithm>
#include <vector>
#include <list>
#include <string>
#include <fstream>
#include <io.h>

#include "svm.h"
#include "opencv2/opencv.hpp"

#include "../test/fs.h"

namespace nao {
	namespace svm {
		static svm_parameter _default_param;
		class SvmTrain {
		public:
			explicit SvmTrain(cv::Size train_size);
			SvmTrain();
			~SvmTrain(void);
			bool init(std::string svm_model_path, std::string model_name);
			void getFileList(const std::string& filePath, const int& lable, const std::string& format = "(.*)(.png|jpg|bmp)");
			void readTestFileList(const std::string& filePath, const std::string& format = "(.*)(.png|jpg|bmp)");
			void processHogFeature(const cv::HOGDescriptor& hog);
			void trainLibSVM(svm_parameter& param = _default_param, const int& used = 0);
			double testLibSVM(const cv::Mat& src, const cv::HOGDescriptor& hog, double prob_estimates[]);
			void test(const cv::HOGDescriptor& hog);
			void addFeatureLabel(const cv::Mat& feature, const std::vector<int>& label);
			void copyFeatureLabel();
			double testFeatureLibSVM(const cv::Mat& feature, double prob_estimates[]);
			void   test();
		public:
			//the list of train images  //the lable of train image
			std::vector<std::pair<int, std::string>> _train_data;
			std::vector<std::pair<int, cv::Mat>> _train_feature_data;
			//the list of test images
			std::vector<std::string> _test_img_list;
			// the path  and name of model
			std::string _base_path;
			std::string _model_name;
			//feature data and lable
			cv::Mat _data_mat;
			cv::Mat _label_mat;
			svm_model* _svm;
			cv::Size _train_size;
		};//class SvmTrain
	}//namespace svm
}//namespace nao
#endif  //__SVM_TRAIN_H__
/*----------------------------------------------------------------------------- (C) COPYRIGHT LEI *****END OF FILE------------------------------------------------------------------------------*/
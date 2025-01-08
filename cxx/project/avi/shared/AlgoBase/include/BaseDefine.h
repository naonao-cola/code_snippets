#pragma once


//////////////////////////////////////////////////////////////////////////
// OpenCV 3.1
//////////////////////////////////////////////////////////////////////////
#include <opencv\cv.h>
#include <opencv\highgui.h>
#include <opencv2\opencv.hpp>
#include <opencv2\core\cuda.hpp>
#include <opencv2\highgui\highgui.hpp>
//#include <opencv2\cudafilters.hpp>
//#include <opencv2\cudaimgproc.hpp>
//#include <opencv2\cudaarithm.hpp>
//#include <opencv2\cudabgsegm.hpp>
//#include <opencv2\cudacodec.hpp>
//#include <opencv2\cudafeatures2d.hpp>
//#include <opencv2\cudaobjdetect.hpp>
//#include <opencv2\cudawarping.hpp>

using namespace cv;
using namespace cv::ml;
using namespace cv::cuda;

#  if defined(_EXPORT_BASE_DLL)
#    define ExportAPI __declspec(dllexport)
#  elif defined(_IMPORT_BASE_DLL)
#    define ExportAPI __declspec(dllimport)
#  else
#    define ExportAPI
#  endif

#ifndef PI
	#define PI acos(-1.)
#endif // !PI

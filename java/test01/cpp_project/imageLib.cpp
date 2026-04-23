

#include "imageLib.h"
#include <opencv2/opencv.hpp>

// 缩放实现
bool resizeImage(const char *srcPath, const char *destPath, int width,
                 int height) {
  try {
    cv::Mat img = cv::imread(srcPath);
    if (img.empty())
      return false;

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(width, height));
    return cv::imwrite(destPath, resized);
  } catch (...) {
    return false;
  }
}

// 灰度实现
bool toGrayImage(const char *srcPath, const char *destPath) {
  try {
    cv::Mat img = cv::imread(srcPath);
    if (img.empty())
      return false;

    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    return cv::imwrite(destPath, gray);
  } catch (...) {
    return false;
  }
}

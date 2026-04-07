/**
 * @FilePath     : /pcl_test/src/win_dll_demo.cpp
 * @Description  :
 * @Author       : weiwei.wang
 * @Date         : 2026-01-13 08:43:51
 * @Version      : 0.0.1
 * @LastEditors  : weiwei.wang
 * @LastEditTime : 2026-01-13 08:44:05
 * @Copyright (c) 2026 by G, All Rights Reserved.
 **/
#include "rsldSDK.h"
// #include <winsock.h>

#include <thread>

#include <pcl/console/parse.h>
#include <pcl/console/print.h>
#include <pcl/console/time.h>
#include <pcl/filters/median_filter.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>

//
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <opencv2/opencv.hpp>

cv::Mat imageDataToMat(const ImageData &img) {
  int type = 0;

  switch (img.format) {
  case PixelFormat::GRAY8:
    type = CV_8UC1;
    break;
  case PixelFormat::RGB8:
    type = CV_8UC3;
    break;
  case PixelFormat::BGR8:
    type = CV_8UC3;
    break;
  case PixelFormat::RGBA8:
    type = CV_8UC4;
    break;
  case PixelFormat::DEPTH16:
    type = CV_16UC1;
    break;
  case PixelFormat::FLOAT32:
    type = CV_32FC1;
    break;
  default:
    throw std::runtime_error("Unsupported PixelFormat");
  }

  // 注意：构造 Mat 时要传入 stride 作为 step 参数
  return cv::Mat(img.height, img.width, type, img.data, img.stride);
}

int main(int argc, char **argv) {
  void *receiver1 = CreateInterface(600, 832);
  void *receiver2 = CreateInterface(600, 832);

  uint8_t *colordata = new uint8_t[2448 * 1736 * 3];
  uint8_t *depthdata = new uint8_t[832 * 600 * 2];
  memset(colordata, 0, 2448 * 1736 * 3);
  memset(depthdata, 0, 832 * 600 * 2);

  // Configure devices IP
  setTargetIp(receiver1, "192.168.1.64", ""); // lidar 1
  setTargetIp(receiver2, "192.168.1.65", ""); // lidar 2

  if (StartRecv(receiver1))
    printf("Open devices1  successfully\n");
  else
    printf("Open devices1 failed\n");

  if (StartRecv(receiver2))
    printf("Open devices2  successfully\n");
  else
    printf("Open devices2 failed\n");

  // laserEnable(receiver1, true);
  // laserEnable(receiver2, true);
  // Sleep(10000);

  // get lidar 1 images
  ImageData imgColor, imgDepth;
  imgColor.width = 0;
  imgColor.height = 0;
  imgColor.data = colordata;
  imgDepth.width = 0;
  imgDepth.height = 0;
  imgDepth.data = depthdata;

  getColorVsDepth(receiver1, &imgColor, &imgDepth);
  if (imgDepth.width != 0) {
    cv::Mat depth = imageDataToMat(imgDepth);
    cv::imwrite("depth1.png", depth);
  }
  if (imgColor.width != 0) {
    cv::Mat color = imageDataToMat(imgColor);
    cv::imwrite("color1.png", color);
  }
  memset(colordata, 0, 2448 * 1736 * 3);
  memset(depthdata, 0, 800 * 600 * 2);

  ImageData imgColorReg, imgDepthReg;
  imgColorReg.data = colordata;
  imgColorReg.width = 0;
  imgColorReg.height = 0;
  imgDepthReg.data = depthdata;
  imgDepthReg.width = 0;
  imgDepthReg.height = 0;
  getColorVsDepthRegister(receiver1, &imgColorReg, &imgDepthReg);
  if (imgDepthReg.width != 0) {
    cv::Mat depth = imageDataToMat(imgDepthReg);
    cv::imwrite("depthreg1.png", depth);
  }
  if (imgColorReg.width != 0) {
    cv::Mat color = imageDataToMat(imgColorReg);
    cv::imwrite("colorreg1.png", color);
  }
  // get lidar 1 pointcloud
  std::vector<pointxyzrgb> points1;
  getPointsData(receiver1, points1);

  if (points1.size() > 0) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZRGB>);
    cloud->points.resize(points1.size());
    cloud->width = static_cast<uint32_t>(cloud->points.size());
    cloud->height = 1;
    cloud->is_dense = false;
    for (int i = 0; i < points1.size(); ++i) {
      cloud->points[i].x = points1[i].x;
      cloud->points[i].y = points1[i].y;
      cloud->points[i].z = points1[i].z;
      cloud->points[i].r = points1[i].r;
      cloud->points[i].g = points1[i].g;
      cloud->points[i].b = points1[i].b;
    }

    pcl::io::savePLYFile("cloud1.ply", *cloud);
  }

  // get lidar 2 images
  ImageData imgColor2, imgDepth2;
  imgColor2.data = colordata;
  imgColor2.width = 0;
  imgColor2.height = 0;
  imgDepth2.data = depthdata;
  imgDepth2.width = 0;
  imgDepth2.height = 0;
  getColorVsDepth(receiver2, &imgColor2, &imgDepth2);
  if (imgDepth2.width != 0) {
    cv::Mat depth = imageDataToMat(imgDepth2);
    cv::imwrite("depth2.png", depth);
  }
  if (imgColor2.width != 0) {
    cv::Mat color = imageDataToMat(imgColor2);
    cv::imwrite("color2.png", color);
  }

  ImageData imgColor2Reg, imgDepth2Reg;
  imgColor2Reg.data = colordata;
  imgColor2Reg.width = 0;
  imgColor2Reg.height = 0;
  imgDepth2Reg.data = depthdata;
  imgDepth2Reg.width = 0;
  imgDepth2Reg.height = 0;
  getColorVsDepthRegister(receiver2, &imgColor2Reg, &imgDepth2Reg);
  if (imgDepth2Reg.width != 0) {
    cv::Mat depth = imageDataToMat(imgDepth2Reg);
    cv::imwrite("depthreg2.png", depth);
  }
  if (imgColor2Reg.width != 0) {
    cv::Mat color = imageDataToMat(imgColor2Reg);
    cv::imwrite("colorreg2.png", color);
  }

  // get lidar 2 pointcloud
  std::vector<pointxyzrgb> points2;
  getPointsData(receiver2, points2);
  if (points2.size() > 0) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud2(
        new pcl::PointCloud<pcl::PointXYZRGB>);
    cloud2->points.resize(points2.size());
    cloud2->width = static_cast<uint32_t>(cloud2->points.size());
    cloud2->height = 1;
    cloud2->is_dense = false;
    for (int i = 0; i < points2.size(); ++i) {
      cloud2->points[i].x = points2[i].x;
      cloud2->points[i].y = points2[i].y;
      cloud2->points[i].z = points2[i].z;
      cloud2->points[i].r = points2[i].r;
      cloud2->points[i].g = points2[i].g;
      cloud2->points[i].b = points2[i].b;
    }
    pcl::io::savePLYFile("cloud2.ply", *cloud2);
  }

  // laserEnable(receiver1, false);
  // laserEnable(receiver2, false);

  if (StopRecv(receiver1))
    printf("Close devices1 successfully\n");
  else
    printf("Close devices1 failed\n");

  if (StopRecv(receiver2))
    printf("Close devices2 successfully\n");
  else
    printf("Close devices2 failed\n");
  DestroyInterface(receiver1);
  DestroyInterface(receiver2);

  delete[] colordata;
  delete[] depthdata;
  return 0;
}

/**
 * @FilePath     : /pcl_test/src/test_8.cpp
 * @Description  : Lidar data collection and saving to PCD files without processing
 * @Author       : weiwei.wang
 * @Date         : 2026-04-09
 * @Version      : 0.0.1
 * @Copyright (c) 2026 by G, All Rights Reserved.
 **/

#include "rsldSDK.h"
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <string>
#include <thread>

namespace fs = std::filesystem;

// 辅助函数：将 SDK 的 ImageData 转换为 OpenCV 的 cv::Mat
cv::Mat imageDataToMat(const ImageData& img)
{
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
    return cv::Mat(img.height, img.width, type, img.data, img.stride);
}

int main(int argc, char** argv)
{
    // 创建保存目录
    std::string save_dir = "captured_pcd";
    if (!fs::exists(save_dir)) {
        fs::create_directory(save_dir);
        printf("Created directory: %s\n", save_dir.c_str());
    }

    // 初始化 SDK
    void*    receiver  = CreateInterface(600, 800);
    uint8_t* colordata = new uint8_t[2448 * 1736 * 3];
    uint8_t* depthdata = new uint8_t[800 * 600 * 2];
    setTargetIp(receiver, "192.168.1.64", "");

    if (!StartRecv(receiver)) {
        printf("Open device failed\n");
        delete[] colordata;
        delete[] depthdata;
        DestroyInterface(receiver);
        return -1;
    }

    printf("Open device successfully.\n");
    printf("Controls:\n");


    ImageData imgColorReg, imgDepthReg;
    imgColorReg.data = colordata;
    imgDepthReg.data = depthdata;
    std::string              xmlpath = "E:/test/pcl_test/config/calib_color_readFromCamera 1.yaml";
    std::vector<pointxyzrgb> sdk_points;
    int                      frame_count = 0;
    bool                     auto_save   = true;



    while (true) {
        memset(colordata, 0, 2448 * 1736 * 3);
        memset(depthdata, 0, 800 * 600 * 2);
        imgColorReg.width = 0;
        imgDepthReg.width = 0;

        // 获取图像用于实时预览
        getColorVsDepth(receiver, &imgColorReg, &imgDepthReg);
        cv::Mat color_img;
        if (imgColorReg.width > 0 && imgDepthReg.width > 0) {
            color_img = imageDataToMat(imgColorReg);

            // 在图像上显示状态信息

            if (auto_save) {
               

                std::vector<Point2i> all_objPixels;   // 获取基础全部数据
                for (int r = 0; r < color_img.rows; ++r) {
                    for (int c = 0; c < color_img.cols; ++c) {
                        all_objPixels.push_back({c, r});
                    }
                }
                std::vector<pointxyzrgb> all_points;
                ImageData*               all_rgbd_placeholder = nullptr;   // 不需要RGBD输出
                std::vector<Point2i>     all_rgbindexes_placeholder;
                getObjectPoints(receiver, all_objPixels, imgDepthReg, xmlpath, all_points, all_rgbd_placeholder, all_rgbindexes_placeholder);   // 6. 调用SDK接口获取点云

                if (!all_points.empty()) {
                    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
                    cloud->points.resize(all_points.size());
                    cloud->width    = static_cast<uint32_t>(all_points.size());
                    cloud->height   = 1;
                    cloud->is_dense = false;
                    for (size_t k = 0; k < all_points.size(); ++k) {
                        cloud->points[k].x = all_points[k].x;
                        cloud->points[k].y = all_points[k].y;
                        cloud->points[k].z = all_points[k].z;
                        cloud->points[k].r = all_points[k].r;
                        cloud->points[k].g = all_points[k].g;
                        cloud->points[k].b = all_points[k].b;
                    }

                    // 生成带时间戳的文件名
                    auto              now       = std::chrono::system_clock::now();
                    auto              in_time_t = std::chrono::system_clock::to_time_t(now);
                    std::stringstream ss, sp;
                    ss << save_dir << "/frame_" << std::put_time(std::localtime(&in_time_t), "%Y%m%d_%H%M%S") << "_" << std::setw(4) << std::setfill('0') << frame_count << ".pcd";
                    sp << save_dir << "/frame_" << std::put_time(std::localtime(&in_time_t), "%Y%m%d_%H%M%S") << "_" << std::setw(4) << std::setfill('0') << frame_count << ".jpg";

                    std::string filename     = ss.str();
                    std::string jpg_filename = sp.str();
                    cv::imwrite(jpg_filename, color_img);
                    pcl::io::savePCDFileBinary(filename, *cloud);
                    printf("[%d] Saved: %s (points: %zu)\n", frame_count, filename.c_str(), cloud->points.size());
                    printf("Saved jpg: %s \n", jpg_filename.c_str());
                    frame_count++;

                    // 自动保存模式下限制频率，避免磁盘写入太快
                    if (auto_save) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                    }

                    
                }



            }
        }
    }

    StopRecv(receiver);
    DestroyInterface(receiver);
    delete[] colordata;
    delete[] depthdata;
    printf("Capture stopped. Total frames saved: %d\n", frame_count);
    return 0;
}

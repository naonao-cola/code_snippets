/**
 * @FilePath     : /pcl_test/src/test_5_trt.cpp
 * @Description  :
 * @Author       : weiwei.wang
 * @Date         : 2026-04-21 11:11:51
 * @Version      : 0.0.1
 * @LastEditors  : weiwei.wang
 * @LastEditTime : 2026-04-21 16:48:51
 * @Copyright (c) 2026 by G, All Rights Reserved.
 **/

#include "rsldSDK.h"
#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

#include <pcl/console/parse.h>
#include <pcl/console/print.h>
#include <pcl/console/time.h>
#include <pcl/filters/median_filter.h>
#include <pcl/filters/random_sample.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree_flann.h>

#include "distance_calc.h"
#include "ox_seg.h"
#include "yolov8_seg.hpp"
#include <cfloat>
#include <chrono>
#include <mutex>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <thread>

#define TICK(x) auto bench_##x = std::chrono::high_resolution_clock::now();
#define TOCK(x) std::cout << #x ": " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - bench_##x).count() << "us" << std::endl;

#define ENABLE_COLLISION_WARNING 1        // 设置为 1 开启预警
#define COLLISION_WARNING_DISTANCE 0.5f   // 预警距离（米）
#define RANDOM_SAMPLE_POINTS 2000         // 随机采样保留的点数

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

struct ObjectCloud
{
    int                                    object_id;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
};

// 新增：处理单个目标的函数
void processSingleObject(int i, const Object& obj, const cv::Mat& color_img, const ImageData& imgDepthReg, void* receiver1, const std::string& xmlpath, std::vector<ObjectCloud>& object_list, std::mutex& list_mutex, std::mutex& sdk_mutex)
{
    if (obj.prob < 0.45)
        return;

    cv::Mat        mask         = obj.boxMask;   // 这是 rect 大小的 mask
    cv::Mat        mask_cropped = cv::Mat::zeros(color_img.size(), CV_8UC1);
    cv::Rect_<int> roi_int      = obj.rect;
    cv::Rect       roi          = roi_int & cv::Rect(0, 0, color_img.cols, color_img.rows);

    if (roi.width > 0 && roi.height > 0) {
        cv::Mat mask_to_copy;
        if (mask.size() != roi.size()) {
            cv::resize(mask, mask_to_copy, roi.size(), 0, 0, cv::INTER_NEAREST);
        }
        else {
            mask_to_copy = mask;
        }
        mask_to_copy.copyTo(mask_cropped(roi));
        mask_cropped   = mask_cropped * 255;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
        cv::erode(mask_cropped, mask_cropped, kernel);
    }

    std::vector<Point2i> objPixels;
    cv::Mat              locations;
    cv::findNonZero(mask_cropped, locations);
    if (!locations.empty()) {
        objPixels.resize(locations.total());
        std::memcpy(objPixels.data(), locations.data, locations.total() * sizeof(Point2i));
    }

    std::vector<pointxyzrgb> person_points;
    ImageData*               rgbd_placeholder = nullptr;
    std::vector<Point2i>     rgbindexes_placeholder;
    if (!objPixels.empty()) {
        {
            TICK(getObjectPoints)
            std::lock_guard<std::mutex> lock(sdk_mutex);
            getObjectPoints(receiver1, objPixels, imgDepthReg, xmlpath, person_points, rgbd_placeholder, rgbindexes_placeholder);
            TOCK(getObjectPoints)
        }

        if (!person_points.empty()) {
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
            cloud->width    = person_points.size();
            cloud->height   = 1;
            cloud->is_dense = false;
            cloud->points.resize(cloud->width * cloud->height);
            for (size_t k = 0; k < person_points.size(); ++k) {
                cloud->points[k].x = person_points[k].x;
                cloud->points[k].y = person_points[k].y;
                cloud->points[k].z = person_points[k].z;
                cloud->points[k].r = person_points[k].r;
                cloud->points[k].g = person_points[k].g;
                cloud->points[k].b = person_points[k].b;
            }
            if (!cloud->empty()) {


                // 2. 随机下采样
                TICK(RANDOM_SAMPLE_POINTS)
                pcl::PointCloud<pcl::PointXYZRGB>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
                if (cloud->size() > RANDOM_SAMPLE_POINTS) {
                    pcl::RandomSample<pcl::PointXYZRGB> ran;
                    ran.setInputCloud(cloud);
                    ran.setSample(RANDOM_SAMPLE_POINTS);
                    ran.filter(*downsampled_cloud);
                }
                else {
                    downsampled_cloud = cloud;
                }
                TOCK(RANDOM_SAMPLE_POINTS)


                TICK(setStddevMulThresh)
                pcl::PointCloud<pcl::PointXYZRGB>::Ptr           sor_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
                pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
                sor.setInputCloud(downsampled_cloud);
                sor.setMeanK(50);
                sor.setStddevMulThresh(1.0);
                sor.filter(*sor_cloud);
                TOCK(setStddevMulThresh)
                if (sor_cloud->size() >= 10) {
                    ObjectCloud obj_cloud;
                    obj_cloud.object_id = i;
                    obj_cloud.cloud     = sor_cloud;

                    std::lock_guard<std::mutex> lock(list_mutex);
                    object_list.push_back(obj_cloud);
                }
            }
        }
    }
}

ImageData matToImageData(const cv::Mat& mat, PixelFormat format)
{
    ImageData img;
    img.width    = mat.cols;
    img.height   = mat.rows;
    img.format   = format;
    img.channels = mat.channels();

    // 计算每行字节数 (stride)
    size_t elementSize = (mat.depth() == CV_16U || mat.depth() == CV_16S) ? 2 : (mat.depth() == CV_32F ? 4 : 1);
    img.stride         = img.width * img.channels * elementSize;

    size_t dataSize = img.height * img.stride;
    img.data        = new uint8_t[dataSize];
    memcpy(img.data, mat.data, dataSize);
    return img;
}

/**
 * @brief 点云(相机坐标系)转深度图
 * @param cloud 输入点云 pcl::PointXYZ 相机坐标系
 * @param K 相机内参矩阵 3x3
 * @param img_w 输出深度图宽度
 * @param img_h 输出深度图高度
 * @param max_depth 最大有效深度(米)，超出视为无效
 * @param min_depth 最小有效深度(米)
 * @param depth_16bit 输出16位深度图(单位:毫米)
 * @param depth_8bit 输出8位归一化深度图(0~255)
 */
void pointCloud2DepthMap(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud, const cv::Mat& K, int img_w, int img_h, float min_depth, float max_depth, cv::Mat& depth_16bit, cv::Mat& depth_8bit)
{
    // 初始化深度图：0 代表无效值
    depth_16bit = cv::Mat::zeros(img_h, img_w, CV_16UC1);
    depth_8bit  = cv::Mat::zeros(img_h, img_w, CV_8UC1);

    // 相机内参解析
    float fx = K.at<double>(0, 0);
    float fy = K.at<double>(1, 1);
    float cx = K.at<double>(0, 2);
    float cy = K.at<double>(1, 2);

    // 遍历点云投影
    for (const auto& pt : cloud->points) {
        float x = pt.x;
        float y = pt.y;
        float z = pt.z;

        // 过滤无效深度
        if (z < min_depth || z > max_depth)
            continue;

        // 相机投影公式：u = fx * X/Z + cx, v = fy * Y/Z + cy
        int u = static_cast<int>(fx * x / z + cx);
        int v = static_cast<int>(fy * y / z + cy);

        // 过滤超出图像范围的点
        if (u < 0 || u >= img_w || v < 0 || v >= img_h)
            continue;

        // 转换为毫米存入16位深度图
        uint16_t depth_mm = static_cast<uint16_t>(z * 1000.0f);
        // 近点覆盖远点，如需远点覆盖可加判断
        depth_16bit.at<uint16_t>(v, u) = depth_mm;
    }

    // 生成8位归一化深度图
    if (max_depth > min_depth) {
        float scale = 255.0f / ((max_depth - min_depth) * 1000.0f);
        for (int y = 0; y < img_h; ++y) {
            for (int x = 0; x < img_w; ++x) {
                uint16_t d = depth_16bit.at<uint16_t>(y, x);
                if (d == 0)
                    depth_8bit.at<uchar>(y, x) = 0;
                else
                    depth_8bit.at<uchar>(y, x) = static_cast<uchar>((d - min_depth * 1000.0f) * scale);
            }
        }
    }
}


// 检测所有点云之间的碰撞
std::vector<std::pair<int, int>> detectAllCollisions(const std::vector<ObjectCloud>& objects, float warning_distance)
{
    std::vector<std::pair<int, int>> collisions;
    TICK(CalculateDistance_CUDA);
    for (size_t i = 0; i < objects.size(); ++i) {
        for (size_t j = i + 1; j < objects.size(); ++j) {

            // 对比测试：CUDA加速法
            // 首先将点云转换为连续的 float 数组

            int                numA = objects[i].cloud->size();
            int                numB = objects[j].cloud->size();
            std::vector<float> cloudA_data(numA * 3);
            std::vector<float> cloudB_data(numB * 3);
            for (int k = 0; k < numA; ++k) {
                cloudA_data[k * 3 + 0] = objects[i].cloud->points[k].x;
                cloudA_data[k * 3 + 1] = objects[i].cloud->points[k].y;
                cloudA_data[k * 3 + 2] = objects[i].cloud->points[k].z;
            }
            for (int k = 0; k < numB; ++k) {
                cloudB_data[k * 3 + 0] = objects[j].cloud->points[k].x;
                cloudB_data[k * 3 + 1] = objects[j].cloud->points[k].y;
                cloudB_data[k * 3 + 2] = objects[j].cloud->points[k].z;
            }


            float dist_cuda = calculateMinDistanceCUDA(cloudA_data.data(), numA, cloudB_data.data(), numB);


            // 实际使用CUDA计算的结果
            float dist = dist_cuda;

            std::cout << "Distance between object " << objects[i].object_id << " and " << objects[j].object_id << " -> CUDA: m " << dist_cuda << std::endl;

            if (dist < warning_distance) {
                collisions.push_back({objects[i].object_id, objects[j].object_id});
                std::cout << ">>> COLLISION WARNING: Object " << objects[i].object_id << " and Object " << objects[j].object_id << " are too close! (" << dist << "m) <<<" << std::endl;
            }
        }
    }
    TOCK(CalculateDistance_CUDA);
    return collisions;
}

// 可视化信息（在图像上绘制）
void visualizeCollisions(cv::Mat& image, const std::vector<ObjectCloud>& objects, const std::vector<std::pair<int, int>>& collisions)
{
    for (size_t i = 0; i < objects.size(); ++i) {
        const auto& obj = objects[i];

        bool has_collision = false;
        for (auto& col : collisions) {
            if (col.first == obj.object_id || col.second == obj.object_id) {
                has_collision = true;
                break;
            }
        }

        std::string info = "ID:" + std::to_string(obj.object_id) + " Pts:" + std::to_string(obj.cloud->size());

        if (has_collision) {
            info += " [COLLISION!]";
        }

        cv::Scalar color = has_collision ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
        int        y_pos = 30 + i * 30;
        cv::putText(image, info, cv::Point(10, y_pos), cv::FONT_HERSHEY_SIMPLEX, 0.7, color, 2);
    }
}

int test_realtime_point2point_collision()
{
    void*       receiver1   = CreateInterface(600, 800);
    std::string xmlpath     = R"(E:\test\pcl_test\config\calib_color_readFromCamera 1.yaml)";
    std::string folder_path = R"(E:\test\pcl_test\build\windows\x64\releasedbg\captured_pcd)";   // 数据文件夹路径

    if (!std::filesystem::exists(folder_path)) {
        printf("Folder %s does not exist\n", folder_path.c_str());
        DestroyInterface(receiver1);
        return -1;
    }

    std::vector<std::string> image_files;
    for (const auto& entry : std::filesystem::directory_iterator(folder_path)) {
        if (entry.path().extension() == ".jpg") {
            image_files.push_back(entry.path().string());
        }
    }
    std::sort(image_files.begin(), image_files.end());

    if (image_files.empty()) {
        printf("No image files found in %s\n", folder_path.c_str());
        DestroyInterface(receiver1);
        return -1;
    }

    auto yolov8 = new YOLOv8_seg(R"(E:\test\pcl_test\model\mechanical.engine)");
    yolov8->make_pipe(true);
    std::vector<Object> objs;

    for (const auto& img_path : image_files) {
        std::string pcd_path = img_path.substr(0, img_path.find_last_of('.')) + ".pcd";
        if (!std::filesystem::exists(pcd_path)) {
            continue;
        }

        cv::Mat color_img = cv::imread(img_path);
        if (color_img.empty())
            continue;

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(pcd_path, *scene_cloud) == -1) {
            continue;
        }
        // 点云转深度图
        cv::Mat depth_16bit, depth_8bit;
        cv::Mat K = (cv::Mat_<double>(3, 3) << 8.4224589700000001e+02, 0., 3.9477131200000002e+02, 0., 8.4307144700000003e+02, 3.0220172100000002e+02, 0., 0., 1.);
        pointCloud2DepthMap(scene_cloud, K, 800, 600, 0.1f, 10.0f, depth_16bit, depth_8bit);
        ImageData imgDepthReg;
        imgDepthReg = matToImageData(depth_16bit, PixelFormat::DEPTH16);

        printf("Processing: %s\n", img_path.c_str());

        TICK(TOTAL_TIME)
        TICK(INFER)
        cv::Size size = cv::Size{1280, 1280};
        yolov8->copy_from_Mat(color_img, size);
        yolov8->infer();
        yolov8->postprocess(objs, 0.25f, 0.65f, 100, 32, 320, 320);
        TOCK(INFER)

        std::vector<ObjectCloud> object_list;
        std::mutex               list_mutex;
        std::mutex               sdk_mutex;
        std::vector<std::thread> threads;

        for (int i = 0; i < (int)objs.size(); ++i) {
            threads.emplace_back(processSingleObject, i, std::ref(objs[i]), std::ref(color_img), std::ref(imgDepthReg), receiver1, std::ref(xmlpath), std::ref(object_list), std::ref(list_mutex), std::ref(sdk_mutex));
        }

        for (auto& t : threads) {
            if (t.joinable())
                t.join();
        }

        delete[] imgDepthReg.data;
        if (object_list.size() >= 2) {
            auto collisions = detectAllCollisions(object_list, COLLISION_WARNING_DISTANCE);
            // visualizeCollisions(color_img, object_list, collisions);
        }
        TOCK(TOTAL_TIME)
        // cv::namedWindow("Offline Point-to-Point Collision", cv::WINDOW_NORMAL);
        // cv::imshow("Offline Point-to-Point Collision", color_img);
        // int key = cv::waitKey(500);   // 延时 500ms 以便观察
        // if (key == 'q' || key == 27)
        //     break;
    }

    delete yolov8;
    DestroyInterface(receiver1);
    return 0;
}

int main(int argc, char** argv)
{
    test_realtime_point2point_collision();
    return 0;
}

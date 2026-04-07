/**
 * @FilePath     : /pcl_test/src/test_1.cpp
 * @Description  :
 * @Author       : weiwei.wang
 * @Date         : 2026-01-20 13:49:49
 * @Version      : 0.0.1
 * @LastEditors  : weiwei.wang
 * @LastEditTime : 2026-01-21 13:11:31
 * @Copyright (c) 2026 by G, All Rights Reserved.
 **/

#include "rsldSDK.h"

#include <string>
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

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "ox_d.h"
#include "ox_seg.h"


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

    return cv::Mat(img.height, img.width, type, img.data, img.stride);   // 注意：构造 Mat 时要传入 stride 作为 step 参数
}


int test_laser()
{

    void* receiver1 = CreateInterface(600, 800);

    uint8_t* colordata = new uint8_t[2448 * 1736 * 3];
    uint8_t* depthdata = new uint8_t[800 * 600 * 2];
    memset(colordata, 0, 2448 * 1736 * 3);
    memset(depthdata, 0, 800 * 600 * 2);

    setTargetIp(receiver1, "192.168.1.64", "");   // Configure devices IP, lidar 1

    if (StartRecv(receiver1))
        printf("Open devices1  successfully\n");
    else
        printf("Open devices1 failed\n");

    // laserEnable(receiver1, true);
    // laserEnable(receiver2, true);
    // Sleep(10000);

    ImageData imgColor, imgDepth;   // get lidar 1 images
    imgColor.width  = 0;
    imgColor.height = 0;
    imgColor.data   = colordata;
    imgDepth.width  = 0;
    imgDepth.height = 0;
    imgDepth.data   = depthdata;

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
    imgColorReg.data   = colordata;
    imgColorReg.width  = 0;
    imgColorReg.height = 0;
    imgDepthReg.data   = depthdata;
    imgDepthReg.width  = 0;
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
    std::vector<pointxyzrgb> points1;   // get lidar 1 pointcloud
    getPointsData(receiver1, points1);

    if (points1.size() > 0) {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        cloud->points.resize(points1.size());
        cloud->width    = static_cast<uint32_t>(cloud->points.size());
        cloud->height   = 1;
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

    if (StopRecv(receiver1))
        printf("Close devices1 successfully\n");
    else
        printf("Close devices1 failed\n");

    DestroyInterface(receiver1);

    delete[] colordata;
    delete[] depthdata;
    return 0;
    return 0;
}

int test_yolo()
{

    auto detector = std::make_unique<YOLOv8Detector>("E:/test/pcl_test/model/best.onnx", false, 0.25f, 0.4f);   // 创建检测器

    cv::Mat         image  = cv::imread("E:/test/pcl_test/data/images/000000305103.jpg");   // 执行检测
    DetectionResult result = detector->Detect(image);

    for (size_t i = 0; i < result.boxes.size(); ++i) {   // 可视化结果
        cv::rectangle(image, result.boxes[i], cv::Scalar(0, 255, 0), 2);
    }
    std::cout << "Detected " << result.boxes.size() << " objects." << std::endl;
    cv::imwrite("E:/test/pcl_test/result/result_image.jpg", image);
    return 0;
}

int test_yolo_seg()
{
    auto            detector = std::make_unique<YOLOv8SegDetector>("E:/test/pcl_test/model/yolov8m-seg.onnx", false, 0.25f, 0.6f);   // 创建检测器
    cv::Mat         image    = cv::imread("F:/wangguanglei/savedata_2/savedata/3/color/color_00002.png");                            // 执行检测
    DetectionResult result   = detector->Detect(image);
    std::cout << "Detected " << result.boxes.size() << " objects." << std::endl;
    cv::Mat image_with_boxes = image.clone();
    cv::Mat visual           = image.clone();
    for (size_t i = 0; i < result.boxes.size(); ++i) {
        const cv::Rect& box = result.boxes[i];
        // 绘制边界框
        // cv::rectangle(image_with_boxes, box, cv::Scalar(0, 255, 0), 2);
        if (i < result.masks.size() && i < result.class_ids.size() && (result.class_ids[i] == 0 || result.class_ids[i] == 39)) {   // 如果有对应的mask, 得到人的id为0的mask
            cv::Mat mask = result.masks[i];
            if (mask.size() != image.size()) {   // 确保mask和图像尺寸一致
                cv::resize(mask, mask, image.size(), 0, 0, cv::INTER_NEAREST);
            }

            cv::Mat  mask_cropped = cv::Mat::zeros(image.size(), CV_8UC1);   // 创建一个只包含框内mask的图像
            cv::Rect roi          = box & cv::Rect(0, 0, image.cols, image.rows);
            if (roi.width > 0 && roi.height > 0) {
                mask(roi).copyTo(mask_cropped(roi));
            }

            // 保存裁剪后的mask
            // cv::imwrite("E:/test/pcl_test/result/mask_cropped_" + std::to_string(i) + ".jpg", mask_cropped * 255);
            // // 创建可视化：原图+mask叠加

            // // 创建红色mask层
            // cv::Mat red_mask(image.size(), CV_8UC3, cv::Scalar(0, 0, 255));
            // cv::Mat mask_3channel;
            // cv::cvtColor(mask_cropped * 255, mask_3channel, cv::COLOR_GRAY2BGR);
            // // 应用mask到红色层
            // red_mask = red_mask & mask_3channel;
            // // 叠加到原图
            // cv::addWeighted(visual, 0.7, red_mask, 0.3, 0, visual);
            cv::rectangle(visual, box, cv::Scalar(0, 255, 0), 2);
            // cv::imwrite("E:/test/pcl_test/result/visual_cropped_" + std::to_string(i) + ".jpg", visual);
        }
    }

    // cv::imwrite("E:/test/pcl_test/result/result_image_with_boxes.jpg", image_with_boxes);
    return 0;
}


float calculate_min_distance(const std::vector<pointxyzrgb>& cloud1, const std::vector<pointxyzrgb>& cloud2)   // 计算两个点云之间的最小距离
{
    float min_dist_sq = std::numeric_limits<float>::max();

    // 注意：如果点云很大，这会非常慢，建议使用KD-Tree加速（如pcl::KdTreeFLANN）
    for (const auto& p1 : cloud1) {   // 简单暴力法：遍历所有点对
        for (const auto& p2 : cloud2) {
            float dx      = p1.x - p2.x;
            float dy      = p1.y - p2.y;
            float dz      = p1.z - p2.z;
            float dist_sq = dx * dx + dy * dy + dz * dz;
            if (dist_sq < min_dist_sq) {
                min_dist_sq = dist_sq;
            }
        }
    }
    return std::sqrt(min_dist_sq);
}

void check_collision(const DetectionResult& result, const std::vector<std::vector<pointxyzrgb>>& all_detect_person_points, float distance_threshold = 0.5f)   // 碰撞检测函数
{
    if (result.masks.empty() || all_detect_person_points.empty())
        return;

    std::vector<int> person_indices;   // 筛选出人类别的索引
    for (size_t i = 0; i < result.class_ids.size(); ++i) {
        if (result.class_ids[i] == 0) {   // 假设0是人
            person_indices.push_back(i);
        }
    }

    if (person_indices.size() < 2)
        return;   // 只有一个人，不可能发生碰撞

    for (size_t i = 0; i < person_indices.size(); ++i) {   // 遍历所有可能的两两组合
        for (size_t j = i + 1; j < person_indices.size(); ++j) {
            int idx1 = person_indices[i];
            int idx2 = person_indices[j];

            // 1. 检查 Mask 轮廓是否有交集
            // 注意：YOLO实例分割的Mask通常是二值图像，且大小与原图一致（经过resize后）
            // 如果Mask是二值的且互斥（通常实例分割尽量互斥，但也可能重叠），
            // 直接做按位与运算可以快速判断像素级重叠。
            // 但如果Mask没有重叠，我们需要判断"轮廓"是否接近或相交。
            // 简单的做法是：计算Mask的交集，如果交集非空，则认为2D层面有接触。

            cv::Mat intersection;
            cv::bitwise_and(result.masks[idx1], result.masks[idx2], intersection);

            int overlap_pixels = cv::countNonZero(intersection);   // 计算非零像素数

            bool collision_2d = overlap_pixels > 0;   // 如果有像素重叠，直接认为碰撞

            // 如果2D mask没有直接重叠，也可以通过膨胀后再检测交集，或者直接跳到3D距离检测。
            // 按照需求：如果 Mask 轮廓有交集 -> 认为有潜在碰撞 -> 计算点云距离
            // 这里我们放宽条件：只要是检测到的两个人，我们都去算一下距离，
            // 或者你可以仅在 bounding box 有交集时才算。
            // 为了更严格符合"mask轮廓交集"的描述，可以先膨胀一下mask再求交，
            // 或者直接假设像素重叠即为交集。

            // 如果 bounding box 都不相交，那 mask 肯定不相交，也不太可能碰撞（除非深度方向）
            cv::Rect intersect_rect = result.boxes[idx1] & result.boxes[idx2];
            if (intersect_rect.area() > 0 || collision_2d) {
                // 2. 计算点云最小距离
                // 注意：all_detect_person_points 存储的是每一帧里检测到的所有人的点云
                // 我们需要确保索引对应关系正确。
                // 在 test_realtime_seg_pointcloud 中，all_detect_person_points 是按检测顺序 push_back 的
                // 而这里 person_indices 也是按检测顺序提取的，所以 idx1/idx2 可能需要映射回 all_detect_person_points 的索引
                // 但 all_detect_person_points 里只存了"检测到的人"的点云，
                // 而 result 包含了所有检测结果（可能包含非人）。
                // 因此我们需要一个映射。
                // 修正逻辑：在主函数里，all_detect_person_points 是只存了人的。
                // 所以这里的 idx1 和 idx2 不能直接用。
                // 我们需要在主函数里把这个映射关系处理好，或者传递对齐的数据。

                // 为了简化，假设 check_collision 接收的 indices 是已经在 all_detect_person_points 里的下标。
                // 这一步比较麻烦，建议在主循环里，把 mask 和 pointcloud 整理成一一对应的结构体再传进来。
                // 但如果不改结构，我们需要在主函数里计数。

                // 让我们重新审视入参：
                // result: 包含所有检测结果（可能混杂其他类）
                // all_detect_person_points: 仅包含"人"的点云

                // 这是一个潜在的索引不对齐问题。
                // 建议：在主函数调用 check_collision 前，构造两个对齐的 vector，
                // 一个是 masks，一个是 pointclouds。
            }
        }
    }
}

void check_collision_aligned(const std::vector<cv::Mat>& person_masks, const std::vector<std::vector<pointxyzrgb>>& person_clouds, float distance_threshold = 0.5f)   // 修正后的 check_collision，入参直接是对齐的 mask 和 pointcloud
{
    if (person_masks.size() < 2 || person_clouds.size() < 2)
        return;
    if (person_masks.size() != person_clouds.size())
        return;

    for (size_t i = 0; i < person_masks.size(); ++i) {
        for (size_t j = i + 1; j < person_masks.size(); ++j) {
            cv::Mat intersection;   // 1. Mask 交集检测
            cv::bitwise_and(person_masks[i], person_masks[j], intersection);
            int overlap = cv::countNonZero(intersection);

            // 如果 Mask 有重叠，或者我们需要更宽松的条件（比如膨胀后重叠）
            // 这里按用户要求：如果 Mask 轮廓有交集（理解为有重叠）
            if (overlap > 0) {
                std::cout << "[Warning] Person " << i << " and Person " << j << " masks overlap!" << std::endl;

                float dist = calculate_min_distance(person_clouds[i], person_clouds[j]);   // 2. 计算点云距离
                std::cout << "  -> Min distance: " << dist << " meters" << std::endl;

                if (dist < distance_threshold) {
                    std::cout << "  -> COLLISION DETECTED! (Distance < " << distance_threshold << ")" << std::endl;
                }
            }
            else {
                // 如果mask不重叠，也可以检查是否非常接近（比如膨胀）
                // 暂时按严格重叠处理，或者你可以把 mask 膨胀一下再算
                // cv::Mat dilated_i, dilated_j;
                // cv::dilate(person_masks[i], dilated_i, cv::Mat(), cv::Point(-1,-1), 5); // 膨胀5像素
                // cv::dilate(person_masks[j], dilated_j, cv::Mat(), cv::Point(-1,-1), 5);
                // cv::bitwise_and(dilated_i, dilated_j, intersection);
                // if (cv::countNonZero(intersection) > 0) { ... }
            }
        }
    }
}

int test_realtime_seg_pointcloud()
{
    void*    receiver1 = CreateInterface(600, 800);      // 1. 初始化激光雷达
    uint8_t* colordata = new uint8_t[2448 * 1736 * 3];   // 分配图像缓冲
    uint8_t* depthdata = new uint8_t[800 * 600 * 2];
    setTargetIp(receiver1, "192.168.1.64", "");   // 设置设备IP
    if (!StartRecv(receiver1)) {
        printf("Open devices1 failed\n");
        delete[] colordata;
        delete[] depthdata;
        DestroyInterface(receiver1);
        return -1;
    }
    printf("Open devices1 successfully\n");
    auto        detector = std::make_unique<YOLOv8SegDetector>("D:/workspace/weiwei/pcl_test/model/yolov8m-seg.onnx", false, 0.6f, 0.6f);   // 2. 初始化YOLO分割模型
    std::string xmlpath  = "D:/workspace/weiwei/pcl_test/config/calib_color_readFromCamera 1.yml";
    ImageData   imgColorReg, imgDepthReg;
    imgColorReg.data = colordata;
    imgDepthReg.data = depthdata;
    int frame_id     = 0;

    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));   // 初始化 PCL 可视化器
    viewer->setBackgroundColor(0, 0, 0);
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();

    bool capture_next = true;   // 控制采集标志，默认为true，采集第一帧
    while (true) {
        if (capture_next) {
            ++frame_id;
            memset(colordata, 0, 2448 * 1736 * 3);
            memset(depthdata, 0, 800 * 600 * 2);
            imgColorReg.width = 0;
            imgDepthReg.width = 0;
            getColorVsDepthRegister(receiver1, &imgColorReg, &imgDepthReg);   // 3. 获取配准后的RGB和深度图

            if (imgColorReg.width > 0 && imgDepthReg.width > 0) {
                cv::Mat                               color_img = imageDataToMat(imgColorReg);   // 转为OpenCV格式
                cv::Mat                               depth_img = imageDataToMat(imgDepthReg);   // CV_16UC1
                DetectionResult                       result    = detector->Detect(color_img);   // 4. 执行YOLO分割检测
                std::vector<pointxyzrgb>              all_person_points;                         // 5. 遍历检测结果，寻找“人”（class_id == 0）
                size_t                                person_index = 0;
                std::vector<std::vector<pointxyzrgb>> all_detect_person_points;
                std::string                           origin_img_path = "D:/aaa/color_" + std::to_string(frame_id) + ".png";
                cv::imwrite(origin_img_path, color_img);   // 保存图像
                for (size_t i = 0; i < result.boxes.size(); ++i) {
                    if (i < result.class_ids.size() && (result.class_ids[i] == 0 || result.class_ids[i] == 39) && i < result.masks.size()) {   // 假设 0 是 person 类 或 39 是 杯子类
                        cv::Mat mask = result.masks[i];
                        if (mask.size() != color_img.size()) {   // 确保mask尺寸与图像一致
                            cv::resize(mask, mask, color_img.size(), 0, 0, cv::INTER_NEAREST);
                        }
                        cv::Mat  mask_cropped = cv::Mat::zeros(color_img.size(), CV_8UC1);   // 只保留box区域的mask
                        cv::Rect roi          = result.boxes[i] & cv::Rect(0, 0, color_img.cols, color_img.rows);
                        if (roi.width > 0 && roi.height > 0) {
                            mask(roi).copyTo(mask_cropped(roi));
                        }
                        mask_cropped = mask_cropped * 255;
                        std::vector<Point2i> objPixels;                 // 收集mask区域内的像素点坐标
                        for (int r = 0; r < mask_cropped.rows; ++r) {   // mask是CV_8UC1，非0即为目标区域
                            const uint8_t* ptr = mask_cropped.ptr<uint8_t>(r);
                            for (int c = 0; c < mask_cropped.cols; ++c) {
                                if (ptr[c] > 0) {
                                    objPixels.push_back({c, r});
                                }
                            }
                        }
                        if (!objPixels.empty()) {
                            std::vector<pointxyzrgb> person_points;
                            ImageData*               rgbd_placeholder = nullptr;   // 不需要RGBD输出
                            std::vector<Point2i>     rgbindexes_placeholder;
                            getObjectPoints(receiver1, objPixels, imgDepthReg, xmlpath, person_points, rgbd_placeholder, rgbindexes_placeholder);   // 6. 调用SDK接口获取点云
                            all_detect_person_points.push_back(person_points);
                            all_person_points.insert(all_person_points.end(), person_points.begin(), person_points.end());   // 将获取到的点加入总集合
                        }
                        cv::rectangle(color_img, result.boxes[i], cv::Scalar(0, 255, 0), 2);   // 可视化：在图上画框和mask轮廓
                        // if (result.boxes[i].area() > 0) {
                        //     std::string win_name = "Person " + std::to_string(i);
                        //     cv::imshow(win_name, color_img);
                        // }
                    }
                }
                // 准备对齐的Mask数据，用于碰撞检测
                // 注意：all_detect_person_points 仅包含检测到的"人"的点云
                // 我们需要从 result.masks 中筛选出对应的 mask
                // std::vector<cv::Mat> person_masks;
                // for (size_t i = 0; i < result.class_ids.size(); ++i) {
                //     if (result.class_ids[i] == 0 && i < result.masks.size()) {
                //         person_masks.push_back(result.masks[i]);
                //     }
                // }
                // check_collision_aligned(person_masks, all_detect_person_points, 0.5f); // 执行碰撞检测


                std::vector<Point2i> objPixels;   // 获取基础全部数据
                for (int r = 0; r < color_img.rows; ++r) {
                    for (int c = 0; c < color_img.cols; ++c) {
                        objPixels.push_back({c, r});
                    }
                }
                std::vector<pointxyzrgb> all_points;
                ImageData*               rgbd_placeholder = nullptr;   // 不需要RGBD输出
                std::vector<Point2i>     rgbindexes_placeholder;
                getObjectPoints(receiver1, objPixels, imgDepthReg, xmlpath, all_points, rgbd_placeholder, rgbindexes_placeholder);   // 6. 调用SDK接口获取点云
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
                    std::string all_filename = "D:/aaa/all_person_cloud_frame_origin_" + std::to_string(frame_id) + ".ply";
                    pcl::io::savePLYFile(all_filename, *cloud);
                }


                if (!all_person_points.empty()) {   // 7. 实时显示点云
                    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
                    cloud->points.resize(all_person_points.size());
                    cloud->width    = static_cast<uint32_t>(all_person_points.size());
                    cloud->height   = 1;
                    cloud->is_dense = false;
                    for (size_t k = 0; k < all_person_points.size(); ++k) {
                        cloud->points[k].x = all_person_points[k].x;
                        cloud->points[k].y = all_person_points[k].y;
                        cloud->points[k].z = all_person_points[k].z;
                        cloud->points[k].r = all_person_points[k].r;
                        cloud->points[k].g = all_person_points[k].g;
                        cloud->points[k].b = all_person_points[k].b;
                    }
                    std::string all_filename = "D:/aaa/all_person_cloud_frame_sub_" + std::to_string(frame_id) + ".ply";
                    pcl::io::savePLYFile(all_filename, *cloud);
                    viewer->removeAllPointClouds();   // 更新点云显示
                    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
                    viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "sample cloud");
                    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
                    viewer->spinOnce(10);

                    cv::imshow("Realtime Detection", color_img);   // 显示检测结果图像
                    capture_next = false;                          // 采集完成，等待下一次触发
                    std::cout << "Detected person! Capture done. Press 'N' to capture next frame, 'ESC' to exit." << std::endl;
                }
                else {
                    cv::imshow("Realtime Detection", color_img);   // 如果没有检测到人，不暂停，继续采集下一帧
                    cv::waitKey(1);                                // 刷新图像窗口
                }
            }
        }
        else {
            viewer->spinOnce(10);   // 仅刷新显示窗口，不采集
            if (cv::waitKey(10) == 'n' || cv::waitKey(10) == 'N') {
                capture_next = true;
                std::cout << "Capturing next frame..." << std::endl;
            }
            if (cv::waitKey(1) == 27) {   // ESC 退出
                break;
            }
        }
    }
    StopRecv(receiver1);   // 清理资源
    DestroyInterface(receiver1);
    delete[] colordata;
    delete[] depthdata;
    return 0;
}

// int test_yaml_read()
// {
//     std::string     filename = "E:/test/pcl_test/config/camera.yaml";
//     cv::FileStorage fs(filename, cv::FileStorage::READ);

//     if (!fs.isOpened()) {
//         std::cerr << "Failed to open " << filename << std::endl;
//         return -1;
//     }

//     cv::Mat cameraMatrixColor, distortionColor;   // 读取相机内参和畸变系数
//     cv::Mat cameraMatrixDepth, distortionDepth;
//     cv::Mat rotation, translation;

//     fs["cameraMatrixColor"] >> cameraMatrixColor;
//     fs["distortionColor"] >> distortionColor;
//     fs["cameraMatrixDepth"] >> cameraMatrixDepth;
//     fs["distortionDepth"] >> distortionDepth;
//     fs["rotation"] >> rotation;
//     fs["translation"] >> translation;

//     std::cout << "Color Camera Matrix:\n" << cameraMatrixColor << std::endl;
//     std::cout << "Color Distortion:\n" << distortionColor << std::endl;
//     std::cout << "Depth Camera Matrix:\n" << cameraMatrixDepth << std::endl;
//     std::cout << "Depth Distortion:\n" << distortionDepth << std::endl;
//     std::cout << "Rotation:\n" << rotation << std::endl;
//     std::cout << "Translation:\n" << translation << std::endl;

//     fs.release();
//     return 0;
// }

int main(int argc, char** argv)
{
    // test_yolo();
    test_yolo_seg();
    // test_realtime_seg_pointcloud();
    //  test_yaml_read();
}

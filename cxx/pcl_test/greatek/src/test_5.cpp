/**
 * @FilePath     : /pcl_test/src/test_5.cpp
 * @Description  : Collision detection using brute-force point-to-point distance calculation
 * @Author       : weiwei.wang
 * @Date         : 2026-03-18
 * @Version      : 0.0.1
 * @Copyright (c) 2026 by G, All Rights Reserved.
 **/

#include "rsldSDK.h"
#include <string>

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
#include "ox_d.h"
#include "ox_seg.h"
#include <cfloat>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

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

// 暴力计算两个点云之间的最小距离 (CPU双重循环版)
float calculateMinDistanceBruteForce(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloudA, const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloudB)
{
    float min_dist_sq = FLT_MAX;
    // 暴力双重循环计算最小距离
    for (const auto& ptA : cloudA->points) {
        for (const auto& ptB : cloudB->points) {
            float dx      = ptA.x - ptB.x;
            float dy      = ptA.y - ptB.y;
            float dz      = ptA.z - ptB.z;
            float dist_sq = dx * dx + dy * dy + dz * dz;
            if (dist_sq < min_dist_sq) {
                min_dist_sq = dist_sq;
            }
        }
    }

    return std::sqrt(min_dist_sq);
}

// 基于 Eigen 矩阵运算计算两个点云之间的最小距离
float calculateMinDistanceMatrix(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloudA, const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloudB)
{
    int numA = cloudA->size();
    int numB = cloudB->size();

    if (numA == 0 || numB == 0)
        return FLT_MAX;

    // 1. 将点云数据映射为 Eigen 矩阵 (N x 3)
    // 注意：PCL的 PointXYZRGB 内存结构并不完全是连续的 float，所以我们需要手动拷贝到 Eigen 矩阵中
    Eigen::MatrixXf matA(numA, 3);
    for (int i = 0; i < numA; ++i) {
        matA(i, 0) = cloudA->points[i].x;
        matA(i, 1) = cloudA->points[i].y;
        matA(i, 2) = cloudA->points[i].z;
    }

    Eigen::MatrixXf matB(numB, 3);
    for (int i = 0; i < numB; ++i) {
        matB(i, 0) = cloudB->points[i].x;
        matB(i, 1) = cloudB->points[i].y;
        matB(i, 2) = cloudB->points[i].z;
    }

    // 2. 利用公式展开计算距离矩阵 D: D_ij = ||A_i - B_j||^2 = ||A_i||^2 + ||B_j||^2 - 2 * <A_i, B_j>
    // 计算 ||A_i||^2 (N x 1)
    Eigen::VectorXf sqA = matA.rowwise().squaredNorm();
    // 计算 ||B_j||^2 (M x 1) -> 转置为 (1 x M)
    Eigen::RowVectorXf sqB = matB.rowwise().squaredNorm().transpose();

    // 计算 - 2 * A * B^T (N x M)
    Eigen::MatrixXf cross_term = -2.0f * (matA * matB.transpose());

    // 组合得到距离平方矩阵 (N x M)
    // 通过 broadcast 机制将 sqA 加到每一列，将 sqB 加到每一行
    Eigen::MatrixXf dist_sq_matrix = (cross_term.colwise() + sqA).rowwise() + sqB;

    // 3. 找出矩阵中的最小值
    float min_dist_sq = dist_sq_matrix.minCoeff();

    // 防止浮点数精度问题导致出现微小的负数
    return std::sqrt(std::max(0.0f, min_dist_sq));
}

// 基于 KD-Tree 的最小距离计算
float calculateMinDistanceKDTree(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloudA, const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloudB)
{
    if (!cloudA || !cloudB || cloudA->empty() || cloudB->empty())
        return FLT_MAX;

    const auto& queryCloud = cloudA;
    const auto& refCloud   = cloudB;

    pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree;
    kdtree.setInputCloud(refCloud);

    std::vector<int>   indices(1);
    std::vector<float> sqrDistances(1);
    float              min_sq = FLT_MAX;

    for (const auto& p : queryCloud->points) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
            continue;
        if (kdtree.nearestKSearch(p, 1, indices, sqrDistances) > 0) {
            if (sqrDistances[0] < min_sq)
                min_sq = sqrDistances[0];
        }
    }
    return std::sqrt(std::max(0.0f, min_sq));
}

// 检测所有点云之间的碰撞
std::vector<std::pair<int, int>> detectAllCollisions(const std::vector<ObjectCloud>& objects, float warning_distance)
{
    std::vector<std::pair<int, int>> collisions;
    for (size_t i = 0; i < objects.size(); ++i) {
        for (size_t j = i + 1; j < objects.size(); ++j) {

            // 对比测试：暴力法
            TICK(CalculateDistance_BruteForce);
            float dist_bf = calculateMinDistanceBruteForce(objects[i].cloud, objects[j].cloud);
            TOCK(CalculateDistance_BruteForce);

            // 对比测试：Eigen矩阵法
            TICK(CalculateDistance_Matrix);
            float dist_mat = calculateMinDistanceMatrix(objects[i].cloud, objects[j].cloud);
            TOCK(CalculateDistance_Matrix);

            // 对比测试：KD-Tree
            TICK(CalculateDistance_KDTree);
            float dist_kd = calculateMinDistanceKDTree(objects[i].cloud, objects[j].cloud);
            TOCK(CalculateDistance_KDTree);

            // 对比测试：CUDA加速法
            // 首先将点云转换为连续的 float 数组
            TICK(CalculateDistance_CUDA);
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
            TOCK(CalculateDistance_CUDA);

            // 实际使用CUDA计算的结果
            float dist = dist_cuda;

            std::cout << "Distance between object " << objects[i].object_id << " and " << objects[j].object_id << " -> BF/MAT/KD/CUDA: " << dist_bf << " / " << dist_mat << " / " << dist_kd << " / " << dist << " m" << std::endl;

            if (dist < warning_distance) {
                collisions.push_back({objects[i].object_id, objects[j].object_id});
                std::cout << ">>> COLLISION WARNING: Object " << objects[i].object_id << " and Object " << objects[j].object_id << " are too close! (" << dist << "m) <<<" << std::endl;
            }
        }
    }
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
    void*    receiver1 = CreateInterface(600, 800);
    uint8_t* colordata = new uint8_t[2448 * 1736 * 3];
    uint8_t* depthdata = new uint8_t[800 * 600 * 2];
    setTargetIp(receiver1, "192.168.1.64", "");
    if (!StartRecv(receiver1)) {
        printf("Open devices1 failed\n");
        delete[] colordata;
        delete[] depthdata;
        DestroyInterface(receiver1);
        return -1;
    }

    printf("Open devices1 successfully\n");
    auto        detector = std::make_unique<YOLOv8SegDetector>("E:/test/pcl_test/model/yolov8m-seg.onnx", false, 0.3f, 0.6f);
    std::string xmlpath  = "E:/test/pcl_test/config/calib_color_readFromCamera 1.yaml";
    ImageData   imgColorReg, imgDepthReg;
    imgColorReg.data = colordata;
    imgDepthReg.data = depthdata;

    while (true) {
        memset(colordata, 0, 2448 * 1736 * 3);
        memset(depthdata, 0, 800 * 600 * 2);
        imgColorReg.width = 0;
        imgDepthReg.width = 0;
        getColorVsDepth(receiver1, &imgColorReg, &imgDepthReg);

        if (imgColorReg.width > 0 && imgDepthReg.width > 0) {
            cv::Mat         color_img = imageDataToMat(imgColorReg);
            cv::Mat         depth_img = imageDataToMat(imgDepthReg);
            DetectionResult result    = detector->Detect(color_img);

            std::vector<ObjectCloud> object_list;

            for (size_t i = 0; i < result.boxes.size(); ++i) {
                if (i < result.class_ids.size() && (result.class_ids[i] == 0) && i < result.masks.size()) {
                    cv::Mat mask = result.masks[i];
                    if (mask.size() != color_img.size()) {
                        cv::resize(mask, mask, color_img.size(), 0, 0, cv::INTER_NEAREST);
                    }

                    cv::Mat  mask_cropped = cv::Mat::zeros(color_img.size(), CV_8UC1);
                    cv::Rect roi          = result.boxes[i] & cv::Rect(0, 0, color_img.cols, color_img.rows);
                    if (roi.width > 0 && roi.height > 0) {
                        mask(roi).copyTo(mask_cropped(roi));
                        mask_cropped = mask_cropped * 255;
                        // 形态学腐蚀去噪
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
                        getObjectPoints(receiver1, objPixels, imgDepthReg, xmlpath, person_points, rgbd_placeholder, rgbindexes_placeholder);

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

                            // 1. SOR 去噪  去除离散点
                            pcl::PointCloud<pcl::PointXYZRGB>::Ptr           sor_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
                            pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
                            sor.setInputCloud(cloud);
                            sor.setMeanK(50);
                            sor.setStddevMulThresh(1.0);
                            sor.filter(*sor_cloud);

                            // 2. 随机下采样
                            // pcl::PointCloud<pcl::PointXYZRGB>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
                            // if (sor_cloud->size() > RANDOM_SAMPLE_POINTS) {
                            //     pcl::RandomSample<pcl::PointXYZRGB> ran;
                            //     ran.setInputCloud(sor_cloud);
                            //     ran.setSample(RANDOM_SAMPLE_POINTS);
                            //     ran.filter(*downsampled_cloud);
                            // }
                            // else {
                            //     downsampled_cloud = sor_cloud;
                            // }

                            if (sor_cloud->size() >= 10) {
                                ObjectCloud obj;
                                obj.object_id = i;
                                obj.cloud     = sor_cloud;
                                object_list.push_back(obj);
                            }
                        }
                    }
                    cv::rectangle(color_img, result.boxes[i], cv::Scalar(0, 255, 0), 2);
                }
            }

            if (object_list.size() >= 2) {
                std::cout << "\n=== detected " << object_list.size() << " targets, calculating distance ===" << std::endl;

                // 执行点到点碰撞检测
                auto collisions = detectAllCollisions(object_list, COLLISION_WARNING_DISTANCE);

                // 可视化结果
                visualizeCollisions(color_img, object_list, collisions);
            }

            cv::namedWindow("Realtime Point-to-Point Collision", cv::WINDOW_NORMAL);
            cv::imshow("Realtime Point-to-Point Collision", color_img);
            cv::waitKey(150);
        }
    }
    StopRecv(receiver1);
    DestroyInterface(receiver1);
    delete[] colordata;
    delete[] depthdata;
    return 0;
}

int main(int argc, char** argv)
{
    test_realtime_point2point_collision();
    return 0;
}

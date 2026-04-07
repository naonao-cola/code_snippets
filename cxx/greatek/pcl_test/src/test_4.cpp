/**
 * @FilePath     : /pcl_test/src/test_4.cpp
 * @Description  : Test OBB computation with different downsampling parameters
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
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "ox_d.h"
#include "ox_seg.h"
#include <cfloat>
#include <chrono>
#include <iomanip>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <pcl/common/common.h>
#include <pcl/common/pca.h>
#include <pcl/common/transforms.h>
#include <pcl/features/impl/moment_of_inertia_estimation.hpp>




#define TICK(x) auto bench_##x = std::chrono::high_resolution_clock::now();
#define TOCK(x) std::cout << #x ": " << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - bench_##x).count() << "us" << std::endl;

struct ObjectOBB
{
    int                                    object_id;
    pcl::PointXYZRGB                       position;     // 物体中心位置
    Eigen::Matrix3f                        rotation;     // 旋转矩阵
    Eigen::Vector3f                        dimensions;   // 长宽高
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
};

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

// 提取的测试函数：针对单个点云，测试不同的下采样参数和不同OBB算法
void testOBBComputationWithParams(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& original_cloud, int object_id)
{
    std::cout << "\n========================================================" << std::endl;
    std::cout << "Testing Object ID: " << object_id << " | Original Points: " << original_cloud->size() << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << std::left << std::setw(15) << "Sample Count" << std::setw(15) << "Method" << std::setw(15) << "Points Left" << std::setw(15) << "Time (us)" << std::setw(30) << "Dimensions (x, y, z)" << std::setw(30) << "Position (x, y, z)" << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;

    // 测试的采样点数 (0 表示不进行下采样)
    std::vector<unsigned int> sample_counts = {0, 500, 1000, 2000, 5000};

    for (unsigned int sample_count : sample_counts) {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_to_use(new pcl::PointCloud<pcl::PointXYZRGB>);

        if (sample_count > 0 && sample_count < original_cloud->size()) {
            pcl::RandomSample<pcl::PointXYZRGB> ran;
            ran.setInputCloud(original_cloud);
            ran.setSample(sample_count);
            ran.filter(*cloud_to_use);

            if (cloud_to_use->size() < 10) {
                continue;
            }
        }
        else {
            cloud_to_use = original_cloud;
        }

        std::string sample_str = (sample_count == 0) ? "Original" : std::to_string(sample_count);


        // ==================== Method 1: PCL MomentOfInertiaEstimation ====================
        auto                                             start_moie = std::chrono::high_resolution_clock::now();
        pcl::MomentOfInertiaEstimation<pcl::PointXYZRGB> feature_extractor;
        feature_extractor.setInputCloud(cloud_to_use);
        feature_extractor.compute();
        pcl::PointXYZRGB min_pt_moie, max_pt_moie, position_moie;
        Eigen::Matrix3f  rotation_moie;
        feature_extractor.getOBB(min_pt_moie, max_pt_moie, position_moie, rotation_moie);
        auto end_moie      = std::chrono::high_resolution_clock::now();
        auto duration_moie = std::chrono::duration_cast<std::chrono::microseconds>(end_moie - start_moie).count();

        // 结果
        Eigen::Vector3f   dim_moie(max_pt_moie.x - min_pt_moie.x, max_pt_moie.y - min_pt_moie.y, max_pt_moie.z - min_pt_moie.z);
        std::stringstream d_moie, p_moie;
        d_moie << std::fixed << std::setprecision(3) << dim_moie.x() << ", " << dim_moie.y() << ", " << dim_moie.z();
        p_moie << std::fixed << std::setprecision(3) << position_moie.x << ", " << position_moie.y << ", " << position_moie.z;
        std::cout << std::left << std::setw(15) << sample_str << std::setw(15) << "MoIE (PCL)" << std::setw(15) << cloud_to_use->size() << std::setw(15) << duration_moie << std::setw(30) << d_moie.str() << std::setw(30) << p_moie.str() << std::endl;


        // ==================== Method 2: Fast PCA-based OBB ====================
        auto start_pca = std::chrono::high_resolution_clock::now();
        // 1. Compute Centroid
        Eigen::Vector4f pcaCentroid;
        pcl::compute3DCentroid(*cloud_to_use, pcaCentroid);
        // 2. Compute Covariance Matrix
        Eigen::Matrix3f covariance;
        pcl::computeCovarianceMatrixNormalized(*cloud_to_use, pcaCentroid, covariance);

        // 3. (PCA),协方差矩阵分解求特征值和特征向量
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
        Eigen::Matrix3f                                eigenVectorsPCA = eigen_solver.eigenvectors();
        Eigen::Vector3f                                eigenValuesPCA  = eigen_solver.eigenvalues();
        // 校正主方向
        eigenVectorsPCA.col(2) = eigenVectorsPCA.col(0).cross(eigenVectorsPCA.col(1));

        // 4. 将输入点云转换至原点
        Eigen::Matrix4f projectionTransform(Eigen::Matrix4f::Identity());
        projectionTransform.block<3, 3>(0, 0) = eigenVectorsPCA.transpose();
        projectionTransform.block<3, 1>(0, 3) = -1.f * (projectionTransform.block<3, 3>(0, 0) * pcaCentroid.head<3>());
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudPointsProjected(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::transformPointCloud(*cloud_to_use, *cloudPointsProjected, projectionTransform);

        // 5. 计算包围盒
        pcl::PointXYZRGB minPoint, maxPoint;
        pcl::getMinMax3D(*cloudPointsProjected, minPoint, maxPoint);
        Eigen::Vector3f meanDiagonal = 0.5f * (maxPoint.getVector3fMap() + minPoint.getVector3fMap());

        // 6. Calculate actual center
        Eigen::Vector3f position_pca = eigenVectorsPCA * meanDiagonal + pcaCentroid.head<3>();
        Eigen::Vector3f dim_pca      = maxPoint.getVector3fMap() - minPoint.getVector3fMap();

        auto end_pca      = std::chrono::high_resolution_clock::now();
        auto duration_pca = std::chrono::duration_cast<std::chrono::microseconds>(end_pca - start_pca).count();

        std::stringstream d_pca, p_pca;
        d_pca << std::fixed << std::setprecision(3) << dim_pca.x() << ", " << dim_pca.y() << ", " << dim_pca.z();
        p_pca << std::fixed << std::setprecision(3) << position_pca.x() << ", " << position_pca.y() << ", " << position_pca.z();

        std::cout << std::left << std::setw(15) << sample_count << std::setw(15) << "PCA (Fast)" << std::setw(15) << cloud_to_use->size() << std::setw(15) << duration_pca << std::setw(30) << d_pca.str() << std::setw(30) << p_pca.str() << std::endl;
        std::cout << "  " << std::endl;
    }
}




int test_realtime_seg_pointcloud_param_test()
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
    auto        detector = std::make_unique<YOLOv8SegDetector>("D:/workspace/weiwei/pcl_test/model/yolov8m-seg.onnx", false, 0.3f, 0.6f);
    std::string xmlpath  = "D:/workspace/weiwei/pcl_test/config/calib_color_readFromCamera 1.yaml";
    ImageData   imgColorReg, imgDepthReg;
    imgColorReg.data = colordata;
    imgDepthReg.data = depthdata;
    int frame_id     = 0;

    // 只测试几帧，不需要一直循环
    int max_test_frames = 5;
    int current_tested  = 0;

    while (current_tested < max_test_frames) {
        memset(colordata, 0, 2448 * 1736 * 3);
        memset(depthdata, 0, 800 * 600 * 2);
        imgColorReg.width = 0;
        imgDepthReg.width = 0;
        getColorVsDepth(receiver1, &imgColorReg, &imgDepthReg);

        ++frame_id;
        if (imgColorReg.width > 0 && imgDepthReg.width > 0) {
            cv::Mat         color_img = imageDataToMat(imgColorReg);
            cv::Mat         depth_img = imageDataToMat(imgDepthReg);
            DetectionResult result    = detector->Detect(color_img);

            std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> object_clouds;
            std::vector<int>                                    object_ids;

            for (size_t i = 0; i < result.boxes.size(); ++i) {
                if (i < result.class_ids.size() && (result.class_ids[i] == 0 || result.class_ids[i] == 39) && i < result.masks.size()) {
                    cv::Mat mask = result.masks[i];
                    if (mask.size() != color_img.size()) {
                        cv::resize(mask, mask, color_img.size(), 0, 0, cv::INTER_NEAREST);
                    }
                    cv::Mat  mask_cropped = cv::Mat::zeros(color_img.size(), CV_8UC1);
                    cv::Rect roi          = result.boxes[i] & cv::Rect(0, 0, color_img.cols, color_img.rows);
                    if (roi.width > 0 && roi.height > 0) {
                        mask(roi).copyTo(mask_cropped(roi));
                        mask_cropped = mask_cropped * 255;

                        // 形态学腐蚀：向内腐蚀以去除边缘干扰噪声
                        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));   // 7x7核约等于向内腐蚀3个像素
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

                            pcl::PointCloud<pcl::PointXYZRGB>::Ptr           sor_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
                            pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
                            sor.setInputCloud(cloud);
                            sor.setMeanK(50);
                            sor.setStddevMulThresh(1.0);
                            sor.filter(*sor_cloud);

                            // 直接保存SOR滤波后的云，不进行固定下采样，以便后续测试
                            if (sor_cloud->size() >= 10) {
                                object_clouds.push_back(sor_cloud);
                                object_ids.push_back(i);
                            }
                        }
                    }
                }
            }

            if (!object_clouds.empty()) {
                std::cout << "\n\n>>> Frame " << frame_id << " processing " << object_clouds.size() << " objects <<<" << std::endl;
                for (size_t i = 0; i < object_clouds.size(); ++i) {
                    testOBBComputationWithParams(object_clouds[i], object_ids[i]);
                }
                current_tested++;
            }
        }
    }

    std::cout << "\nTest finished for " << max_test_frames << " frames with objects." << std::endl;

    StopRecv(receiver1);
    DestroyInterface(receiver1);
    delete[] colordata;
    delete[] depthdata;
    return 0;
}

int main(int argc, char** argv)
{
    test_realtime_seg_pointcloud_param_test();
    return 0;
}

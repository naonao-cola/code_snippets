
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

    // 注意：构造 Mat 时要传入 stride 作为 step 参数
    return cv::Mat(img.height, img.width, type, img.data, img.stride);
}


int test_laser()
{

    void* receiver1 = CreateInterface(600, 800);

    uint8_t* colordata = new uint8_t[2448 * 1736 * 3];
    uint8_t* depthdata = new uint8_t[800 * 600 * 2];
    memset(colordata, 0, 2448 * 1736 * 3);
    memset(depthdata, 0, 800 * 600 * 2);

    // Configure devices IP
    setTargetIp(receiver1, "192.168.1.64", "");   // lidar 1

    if (StartRecv(receiver1))
        printf("Open devices1  successfully\n");
    else
        printf("Open devices1 failed\n");

    // laserEnable(receiver1, true);
    // laserEnable(receiver2, true);
    // Sleep(10000);

    // get lidar 1 images
    ImageData imgColor, imgDepth;
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
    // get lidar 1 pointcloud
    std::vector<pointxyzrgb> points1;
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

    // 创建检测器
    auto detector = std::make_unique<YOLOv8Detector>("E:/test/pcl_test/model/best.onnx", false, 0.25f, 0.4f);

    // 执行检测
    cv::Mat         image  = cv::imread("E:/test/pcl_test/data/images/000000305103.jpg");
    DetectionResult result = detector->Detect(image);

    // 可视化结果
    for (size_t i = 0; i < result.boxes.size(); ++i) {
        cv::rectangle(image, result.boxes[i], cv::Scalar(0, 255, 0), 2);
    }
    std::cout << "Detected " << result.boxes.size() << " objects." << std::endl;
    cv::imwrite("E:/test/pcl_test/result/result_image.jpg", image);
    return 0;
}

int test_yolo_seg()
{
    // 创建检测器
    auto detector = std::make_unique<YOLOv8SegDetector>("E:/test/pcl_test/model/yolov8m-seg.onnx", false, 0.5f, 0.6f);
    // 执行检测
    cv::Mat         image  = cv::imread("F:/wangguanglei/savedata/savedata/5/color/000000305103.jpg");
    DetectionResult result = detector->Detect(image);
    std::cout << "Detected " << result.boxes.size() << " objects." << std::endl;
    cv::Mat image_with_boxes = image.clone();

    for (size_t i = 0; i < result.boxes.size(); ++i) {
        const cv::Rect& box = result.boxes[i];
        // 绘制边界框
        // cv::rectangle(image_with_boxes, box, cv::Scalar(0, 255, 0), 2);
        // 如果有对应的mask
        // 得到人的id为0的mask
        if (i < result.masks.size() && i < result.class_ids.size() /*&& result.class_ids[i] == 0*/) {
            cv::Mat mask = result.masks[i];
            // 确保mask和图像尺寸一致
            if (mask.size() != image.size()) {
                cv::resize(mask, mask, image.size(), 0, 0, cv::INTER_NEAREST);
            }

            // 创建一个只包含框内mask的图像
            cv::Mat  mask_cropped = cv::Mat::zeros(image.size(), CV_8UC1);
            cv::Rect roi          = box & cv::Rect(0, 0, image.cols, image.rows);
            if (roi.width > 0 && roi.height > 0) {
                mask(roi).copyTo(mask_cropped(roi));
            }

            // 保存裁剪后的mask
            // cv::imwrite("E:/test/pcl_test/result/mask_cropped_" + std::to_string(i) + ".jpg", mask_cropped * 255);
            // // 创建可视化：原图+mask叠加
            // cv::Mat visual = image.clone();
            // // 创建红色mask层
            // cv::Mat red_mask(image.size(), CV_8UC3, cv::Scalar(0, 0, 255));
            // cv::Mat mask_3channel;
            // cv::cvtColor(mask_cropped * 255, mask_3channel, cv::COLOR_GRAY2BGR);
            // // 应用mask到红色层
            // red_mask = red_mask & mask_3channel;
            // // 叠加到原图
            // cv::addWeighted(visual, 0.7, red_mask, 0.3, 0, visual);
            // cv::rectangle(visual, box, cv::Scalar(0, 255, 0), 2);
            // cv::imwrite("E:/test/pcl_test/result/visual_cropped_" + std::to_string(i) + ".jpg", visual);
        }
    }

    // cv::imwrite("E:/test/pcl_test/result/result_image_with_boxes.jpg", image_with_boxes);
    return 0;
}


// 计算两个点云之间的最小距离
float calculate_min_distance(const std::vector<pointxyzrgb>& cloud1, const std::vector<pointxyzrgb>& cloud2)
{
    float min_dist_sq = std::numeric_limits<float>::max();

    // 简单暴力法：遍历所有点对
    // 注意：如果点云很大，这会非常慢，建议使用KD-Tree加速（如pcl::KdTreeFLANN）
    for (const auto& p1 : cloud1) {
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

// 碰撞检测函数
void check_collision(const DetectionResult& result, const std::vector<std::vector<pointxyzrgb>>& all_detect_person_points, float distance_threshold = 0.5f)
{
    if (result.masks.empty() || all_detect_person_points.empty())
        return;

    // 筛选出人类别的索引
    std::vector<int> person_indices;
    for (size_t i = 0; i < result.class_ids.size(); ++i) {
        if (result.class_ids[i] == 0) {   // 假设0是人
            person_indices.push_back(i);
        }
    }

    if (person_indices.size() < 2)
        return;   // 只有一个人，不可能发生碰撞

    // 遍历所有可能的两两组合
    for (size_t i = 0; i < person_indices.size(); ++i) {
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

            // 计算非零像素数
            int overlap_pixels = cv::countNonZero(intersection);

            // 如果有像素重叠，直接认为碰撞
            bool collision_2d = overlap_pixels > 0;

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

// 修正后的 check_collision，入参直接是对齐的 mask 和 pointcloud
void check_collision_aligned(const std::vector<cv::Mat>& person_masks, const std::vector<std::vector<pointxyzrgb>>& person_clouds, float distance_threshold = 0.5f)
{
    if (person_masks.size() < 2 || person_clouds.size() < 2)
        return;
    if (person_masks.size() != person_clouds.size())
        return;

    for (size_t i = 0; i < person_masks.size(); ++i) {
        for (size_t j = i + 1; j < person_masks.size(); ++j) {
            // 1. Mask 交集检测
            cv::Mat intersection;
            cv::bitwise_and(person_masks[i], person_masks[j], intersection);
            int overlap = cv::countNonZero(intersection);

            // 如果 Mask 有重叠，或者我们需要更宽松的条件（比如膨胀后重叠）
            // 这里按用户要求：如果 Mask 轮廓有交集（理解为有重叠）
            if (overlap > 0) {
                std::cout << "[Warning] Person " << i << " and Person " << j << " masks overlap!" << std::endl;

                // 2. 计算点云距离
                float dist = calculate_min_distance(person_clouds[i], person_clouds[j]);
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
    // 1. 初始化激光雷达
    void* receiver1 = CreateInterface(600, 800);
    // 分配图像缓冲
    uint8_t* colordata = new uint8_t[2448 * 1736 * 3];
    uint8_t* depthdata = new uint8_t[800 * 600 * 2];
    // 设置设备IP
    setTargetIp(receiver1, "192.168.1.64", "");
    if (!StartRecv(receiver1)) {
        printf("Open devices1 failed\n");
        delete[] colordata;
        delete[] depthdata;
        DestroyInterface(receiver1);
        return -1;
    }
    printf("Open devices1 successfully\n");

    // 2. 初始化YOLO分割模型
    // 注意：请确保模型路径正确，如果使用GPU请将第二个参数设为true
    auto        detector = std::make_unique<YOLOv8SegDetector>("E:/test/pcl_test/model/yolov8m-seg.onnx", false, 0.6f, 0.6f);
    std::string xmlpath  = "E:/test/pcl_test/config/camera.yml";
    ImageData   imgColorReg, imgDepthReg;
    imgColorReg.data = colordata;
    imgDepthReg.data = depthdata;
    int frame_id     = 0;

    // 初始化 PCL 可视化器
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
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
            // 3. 获取配准后的RGB和深度图
            getColorVsDepthRegister(receiver1, &imgColorReg, &imgDepthReg);

            if (imgColorReg.width > 0 && imgDepthReg.width > 0) {
                // 转为OpenCV格式
                cv::Mat color_img = imageDataToMat(imgColorReg);
                cv::Mat depth_img = imageDataToMat(imgDepthReg);   // CV_16UC1
                // 4. 执行YOLO分割检测
                DetectionResult result = detector->Detect(color_img);
                // 5. 遍历检测结果，寻找“人”（class_id == 0）
                std::vector<pointxyzrgb>              all_person_points;
                size_t                                person_index = 0;
                std::vector<std::vector<pointxyzrgb>> all_detect_person_points;
                for (size_t i = 0; i < result.boxes.size(); ++i) {
                    // 假设 0 是 person 类
                    if (i < result.class_ids.size() && result.class_ids[i] == 0 && i < result.masks.size()) {
                        cv::Mat mask = result.masks[i];
                        // 确保mask尺寸与图像一致
                        if (mask.size() != color_img.size()) {
                            cv::resize(mask, mask, color_img.size(), 0, 0, cv::INTER_NEAREST);
                        }
                        //只保留box区域的mask
                        cv::Mat  mask_cropped = cv::Mat::zeros(color_img.size(), CV_8UC1);
                        cv::Rect roi          = result.boxes[i] & cv::Rect(0, 0, color_img.cols, color_img.rows);
                        if (roi.width > 0 && roi.height > 0) {
                            mask(roi).copyTo(mask_cropped(roi));
                        }


                        // 收集mask区域内的像素点坐标
                        std::vector<Point2i> objPixels;
                        // mask是CV_8UC1，非0即为目标区域
                        for (int r = 0; r < mask_cropped.rows; ++r) {
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
                            // 6. 调用SDK接口获取点云
                            // 注意：xmlpath 需要根据实际情况提供，这里暂时留空或填默认
                            getObjectPoints(receiver1, objPixels, imgDepthReg, xmlpath, person_points, rgbd_placeholder, rgbindexes_placeholder);
                            all_detect_person_points.push_back(person_points);
                            // 将获取到的点加入总集合
                            all_person_points.insert(all_person_points.end(), person_points.begin(), person_points.end());
                        }

                        // 可视化：在图上画框和mask轮廓
                        cv::rectangle(color_img, result.boxes[i], cv::Scalar(0, 255, 0), 2);

                        // 创建一个只包含当前人的图像并显示
                        cv::Mat person_img = cv::Mat::zeros(color_img.size(), color_img.type());
                        color_img.copyTo(person_img, mask_cropped);
                        // 裁剪到bbox
                        cv::Rect box = result.boxes[i] & cv::Rect(0, 0, color_img.cols, color_img.rows);
                        if (box.area() > 0) {
                            cv::Mat     person_roi = person_img(box);
                            std::string win_name   = "Person " + std::to_string(i);
                            cv::imshow(win_name, person_roi);
                        }
                    }
                }

                // 准备对齐的Mask数据，用于碰撞检测
                // 注意：all_detect_person_points 仅包含检测到的"人"的点云
                // 我们需要从 result.masks 中筛选出对应的 mask
                std::vector<cv::Mat> person_masks;
                for (size_t i = 0; i < result.class_ids.size(); ++i) {
                    if (result.class_ids[i] == 0 && i < result.masks.size()) {
                        person_masks.push_back(result.masks[i]);
                    }
                }

                // 执行碰撞检测
                check_collision_aligned(person_masks, all_detect_person_points, 0.5f);

                // 7. 实时显示点云
                if (!all_person_points.empty()) {
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

                    // 更新点云显示
                    viewer->removeAllPointClouds();
                    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
                    viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "sample cloud");
                    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
                    viewer->spinOnce(10);

                    // 显示检测结果图像
                    cv::imshow("Realtime Detection", color_img);
                    capture_next = false;   // 采集完成，等待下一次触发
                    std::cout << "Detected person! Capture done. Press 'N' to capture next frame, 'ESC' to exit." << std::endl;
                }
                else {
                    // 如果没有检测到人，不暂停，继续采集下一帧
                    cv::imshow("Realtime Detection", color_img);
                    cv::waitKey(1);   // 刷新图像窗口
                }
            }
        }
        else {
            // 仅刷新显示窗口，不采集
            viewer->spinOnce(10);
            if (cv::waitKey(10) == 'n' || cv::waitKey(10) == 'N') {
                capture_next = true;
                std::cout << "Capturing next frame..." << std::endl;
            }
            if (cv::waitKey(1) == 27) {   // ESC 退出
                break;
            }
        }
    }
    // 清理资源
    StopRecv(receiver1);
    DestroyInterface(receiver1);
    delete[] colordata;
    delete[] depthdata;
    return 0;
}

int test_yaml_read()
{
    std::string     filename = "E:/test/pcl_test/config/camera.yaml";
    cv::FileStorage fs(filename, cv::FileStorage::READ);

    if (!fs.isOpened()) {
        std::cerr << "Failed to open " << filename << std::endl;
        return -1;
    }

    // 读取相机内参和畸变系数
    cv::Mat cameraMatrixColor, distortionColor;
    cv::Mat cameraMatrixDepth, distortionDepth;
    cv::Mat rotation, translation;

    fs["cameraMatrixColor"] >> cameraMatrixColor;
    fs["distortionColor"] >> distortionColor;
    fs["cameraMatrixDepth"] >> cameraMatrixDepth;
    fs["distortionDepth"] >> distortionDepth;
    fs["rotation"] >> rotation;
    fs["translation"] >> translation;

    std::cout << "Color Camera Matrix:\n" << cameraMatrixColor << std::endl;
    std::cout << "Color Distortion:\n" << distortionColor << std::endl;
    std::cout << "Depth Camera Matrix:\n" << cameraMatrixDepth << std::endl;
    std::cout << "Depth Distortion:\n" << distortionDepth << std::endl;
    std::cout << "Rotation:\n" << rotation << std::endl;
    std::cout << "Translation:\n" << translation << std::endl;

    fs.release();
    return 0;
}

int main(int argc, char** argv)
{
    // test_yolo();
    test_yolo_seg();
    //test_realtime_seg_pointcloud();
    // test_yaml_read();
}

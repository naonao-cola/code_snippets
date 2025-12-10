//
// Created by ubuntu on 4/7/23.
//
#include <filesystem>
#include "opencv2/opencv.hpp"
#include "yolov8_pose.hpp"
#include <chrono>

namespace fs = std::filesystem;

const std::vector<std::vector<unsigned int>> KPS_COLORS = {{0, 255, 0}, {0, 255, 0}, {0, 255, 0}, {0, 255, 0}, {0, 255, 0}, {255, 128, 0}, {255, 128, 0}, {255, 128, 0}, {255, 128, 0}, {255, 128, 0}, {255, 128, 0}, {51, 153, 255}, {51, 153, 255}, {51, 153, 255}, {51, 153, 255}, {51, 153, 255}, {51, 153, 255}};

const std::vector<std::vector<unsigned int>> SKELETON = {{16, 14}, {14, 12}, {17, 15}, {15, 13}, {12, 13}, {6, 12}, {7, 13}, {6, 7}, {6, 8}, {7, 9}, {8, 10}, {9, 11}, {2, 3}, {1, 2}, {1, 3}, {2, 4}, {3, 5}, {4, 6}, {5, 7}};

const std::vector<std::vector<unsigned int>> LIMB_COLORS = {{51, 153, 255}, {51, 153, 255}, {51, 153, 255}, {51, 153, 255}, {255, 51, 255}, {255, 51, 255}, {255, 51, 255}, {255, 128, 0}, {255, 128, 0}, {255, 128, 0}, {255, 128, 0}, {255, 128, 0}, {0, 255, 0}, {0, 255, 0}, {0, 255, 0}, {0, 255, 0}, {0, 255, 0}, {0, 255, 0}, {0, 255, 0}};

int main(int argc, char** argv)
{

    // cuda:0
    cudaSetDevice(0);
    const std::string        engine_file_path = R"(./yolov8s-pose.engine)";
    std::vector<std::string> imagePathList;
    imagePathList.push_back(R"(E:\test\pose_test\data\2175e6c724e565d291f9d1b385e60e36.jpg)");

    auto yolov8_pose = new YOLOv8_pose(engine_file_path);
    yolov8_pose->make_pipe(true);

    bool isVideo{true};


    cv::Mat  res, image;
    cv::Size size        = cv::Size{640, 640};
    int      topk        = 100;
    float    score_thres = 0.65f;
    float    iou_thres   = 0.65f;

    std::vector<Object> objs;
    int count = 0;

    if (isVideo) {
        std::string      path = R"(./1.mp4)";
        cv::VideoCapture cap(path);
        if (!cap.isOpened()) {
            printf("can not open %s\n", path.c_str());
            return -1;
        }
        int img_count = 0;
        while (cap.read(image)) {
            if(image.empty()){
                printf("image empty \n");
                break;
            }
            objs.clear();
            yolov8_pose->copy_from_Mat(image, size);
            yolov8_pose->infer();
            yolov8_pose->postprocess(objs, score_thres, iou_thres, topk);
            bool fall_flag = false;
            for(auto obj:objs){
                fall_flag = yolov8_pose->judge(obj.kps);
                if (fall_flag) {
                    std::cout << "index : "<< img_count << "obj socre: " << obj.prob << std::endl;
                    break;
                }
            }
            if (fall_flag) {
                yolov8_pose->draw_objects(image, res, objs, SKELETON, KPS_COLORS, LIMB_COLORS);
                std::string res_path = std::string("./res/").append(std::to_string(img_count)).append(".jpg"); ;
                cv::imwrite(res_path, res);
            }
            img_count++;
        }
    }
    else {
        for (auto& p : imagePathList) {
            objs.clear();
            image = cv::imread(p);
            yolov8_pose->copy_from_Mat(image, size);
            auto start = std::chrono::system_clock::now();
            yolov8_pose->infer();
            auto end = std::chrono::system_clock::now();
            yolov8_pose->postprocess(objs, score_thres, iou_thres, topk);
            yolov8_pose->draw_objects(image, res, objs, SKELETON, KPS_COLORS, LIMB_COLORS);
            auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
            printf("cost %2.4lf ms\n", tc);
            std::string res_path = std::string("E:/test/pose_test/res/").append(std::to_string(count++)).append(".jpg"); ;
            cv::imwrite(res_path, res);
        }
    }
    delete yolov8_pose;
    return 0;
}
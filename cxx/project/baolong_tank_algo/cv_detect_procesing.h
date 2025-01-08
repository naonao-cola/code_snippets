#include <iostream>
using namespace std;
#include <nlohmann/json.hpp>
#include "logger.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include "define_algorithm_pram.h"
#include "bl_config.h"
using json = nlohmann::json;
// using namespace BL_CONFIG;

struct ContourInfo {
    std::vector<double> areas;
    int numContours;
    int area_sum;
    std::vector<cv::Rect> boundingBoxes;
    std::vector<cv::Point2f> centers;

    ContourInfo() : numContours(0), area_sum(0) { }
};
class cv_detect : public Cv_Pram {
    public:
        cv_detect();
        ~cv_detect(){};
        // cv::Mat &gray_img, vector<int>& center_pos, vector<int>& radius_pos, int gray_threshold
        virtual std::string check_c_circle(cv::Mat src_img);
        virtual std::string check_o_circle(cv::Mat src_img);
        virtual bl_config get_bl_config();
        virtual std::string check_plug(cv::Mat img_src);
        virtual bool find_circle_center_easy(cv::Mat circle_img);
        virtual bool find_circle_center_hard(cv::Mat circle_img);
        bool gv_abnormal(cv::Mat img, int threshold_low_value, int threshold_high_value);
        void updateImgCheck(img_check& img);
    private:
        virtual void get_petal_num(cv::Mat src);
        virtual float stat_gray_ratio(cv::Point radius, cv::Point center_pos, cv::Mat src_img);
        virtual void get_petal_ratio();
        virtual cv::Mat preproces_petal(cv::Mat gray, int default);
        virtual int cv_detect::thres_YEN(cv::Mat gary);
        virtual int plug_gv_abnormal(vector<vector<cv::Point>> contours, cv::Mat src);
        virtual float cont_mean_gv(vector<vector<cv::Point>> contours, cv::Mat src);
        virtual cv::Mat blConnectedComponentsWithStats(const cv::Mat &inputImage);
        virtual cv::Mat plug_num(cv::Mat src);
        virtual cv::Mat blplugConnectedComponentsWithStats(const cv::Mat &inputImage);
        // virtual cv::Mat blplug(const cv::Mat &inputImage);
        virtual cv::Mat preproces_petal2(cv::Mat gray);
        virtual void get_petal_ratio2();
        
  
    public:
        int width;
        int height;
        vector<int>* rect_c_pos;
        vector<int>* rect_o_pos;
        // params_O PO;
        // params_C PC;

        ContourInfo contourInfo;
    public:
        bl_config bl_json;
        Param bl_cv_pram;
        img_check cv_check;
        float score = 0;
    private:
        cv::Point center_pos;
        cv::Mat src_img;
        cv::Point C_radius;
        cv::Point O_radius;
        int petal_num;
        vector<vector<cv::Point>> contours;
        float cross_ratio;
        float area_plug_ratio = 0;//202304 hjf add plug judge bug
        vector<double> max_area;
        int over_max = 0;

        // int offset_x = 32;//圆心偏移
        // int offset_y = 27;
        int offset_x = 15;//圆心偏移
        int offset_y = 35;
    public:
        in_param run_config;
        // new_bl_config bl_json;
        

        
};
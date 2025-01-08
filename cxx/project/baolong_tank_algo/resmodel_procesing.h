#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/dnn/dnn.hpp>
#include<string>
#include<iostream>
#include<fstream>
#include<vector>
#include "logger.h"
#include "bl_config.h"
#include "defines.h"
#include "tapp.h"
// using namespace BL_CONFIG;

class ONNXClassifier {
    public:
        ONNXClassifier();
        ~ONNXClassifier();
        float Classify(const cv::Mat& input_image, std::string& out_name, DebugType cls);
        bool ONNXClassifier::init_model( const char* buffer, size_t sizeBuffer, cv::Size _input_size);
        cv::Mat BL_IMG_PROCESS(cv::Mat src, std::string save_path);
        bool ONNXClassifier::gv_abnormal(cv::Mat img, int threshold_low_value, int threshold_high_value);
        void updateImgCheck(img_check& img);
        void ai_normalize(cv::Mat *im, const std::vector<float> &mean, const std::vector<float> &scale, const bool is_scale);
    private:
        void preprocess_input(cv::Mat& image, DebugType cls);
        bool read_labels(const std::string& label_paht);
        
    private:
        cv::Size input_size;
        cv::dnn::Net resnet18;
        cv::Scalar default_mean;
        cv::Scalar default_std;
        std::vector<std::string> labels;
    public:
        cv::Mat detect_img;
        std::string model_path;
        std::string label_path;
        cv::Size _input_size;
        Param bl_dl_pram;
        img_check dl_check;
        std::vector<float> probs;

};


// int main(int argc, char* argv[])
// {
// 	if (argc != 2)
// 	{
// 		std::cout << "input a image file path" << std::endl;
// 		return -1;
// 	}
// 	std::string model_path("../model/classifier.onnx");
// 	std::string label_path("../model/labels.txt");
// 	cv::Size input_size(224, 224);
// 	cv::Mat test_image = cv::imread(argv[1]);
// 	ONNXClassifier classifier(model_path, label_path, input_size);
// 	std::string result;
// 	classifier.Classify(test_image, result);
//         std::cout<<"result: "<<result<<std::endl;
// 	return 0;
// }

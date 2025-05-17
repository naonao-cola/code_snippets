//
// Created by y on 24-4-3.
//
#include "TinnyNetwork.h"
//#include "DihLog.h"
#include "algLog.h"
namespace ALG_DEPLOY{
namespace PSEUDO_NETWORK{

void sigmoid(cv::Mat src, cv::Mat& dst) {
  src = -1 * src;
  exp(src, src);
  src = 1 + src;
  dst = 1 / src;
}

int ReshapeNormalNetworkParams(cv::Mat& conv1_weight, cv::Mat& conv1_bias,
                               cv::Mat& conv2_weight, cv::Mat& conv2_bias,
                               cv::Mat& conv3_weight) {
  try {
    conv1_weight = conv1_weight.reshape(0, 1);  // 1*4
    conv1_bias = conv1_bias.reshape(0, 1);      // 1*4
    conv1_weight.convertTo(conv1_weight, CV_32F);
    conv1_bias.convertTo(conv1_bias, CV_32F);

    conv2_weight = conv2_weight.reshape(0, 4);  // 4*4
    conv2_bias = conv2_bias.reshape(0, 1);      // 1*4
    conv2_weight.convertTo(conv2_weight, CV_32F);
    conv2_bias.convertTo(conv2_bias, CV_32F);

    conv3_weight = conv3_weight.reshape(0, 4);  // 4*1
    conv3_weight.convertTo(conv3_weight, CV_32F);

  } catch (std::exception& e) {
    return -1;
  }
  return 0;
}

void ForwardNormalNetwork(const cv::Mat& conv1_weight,
                          const cv::Mat& conv1_bias,
                          const cv::Mat& conv2_weight,
                          const cv::Mat& conv2_bias,
                          const cv::Mat& conv3_weight, const cv::Mat& input,
                          cv::Mat& output, int batch_size) {
  cv::Mat input_ = input.reshape(0, batch_size);  // b*3
  cv::Mat x = input_ * conv1_weight;     // b*3 3*4 -> b*4
  cv::Mat conv1_bias_dynamic_shape;
  cv::resize(conv1_bias, conv1_bias_dynamic_shape, cv::Size(x.cols,x.rows));// 复制使能够做加法
  x = x + conv1_bias_dynamic_shape;
  sigmoid(x, x);

  x = x * conv2_weight;  // b*4 4*4 -> b*4
  cv::Mat conv2_bias_dynamic_shape;
  cv::resize(conv2_bias, conv2_bias_dynamic_shape, cv::Size(x.cols,x.rows));
  x = x + conv2_bias_dynamic_shape;
  sigmoid(x, x);

  output = x * conv3_weight;  // b*4 4*1 -> b*1
}


int ReshapeLinearNetworkParams(cv::Mat& conv1_weight, cv::Mat& conv1_bias){
  try {
    conv1_weight = conv1_weight.reshape(0, 1);  // 1*4
    conv1_bias = conv1_bias.reshape(0, 1);      // 1*4
    conv1_weight.convertTo(conv1_weight, CV_32F);
    conv1_bias.convertTo(conv1_bias, CV_32F);

  } catch (std::exception& e) {
    return -1;
  }
  return 0;
}

void ForwardLinearNetwork(const cv::Mat& conv1_weight, const cv::Mat& conv1_bias,
                          const cv::Mat& input, cv::Mat& output,
                          int batch_size){
  cv::Mat input_ = input.reshape(0, batch_size);  // 1*3
  cv::Mat x = input_ * conv1_weight;     // 1*3 3*4 -> 1*4
  cv::Mat conv1_bias_dynamic_shape;
  cv::resize(conv1_bias, conv1_bias_dynamic_shape, cv::Size(x.cols,x.rows));
  x = x + conv1_bias_dynamic_shape;
  output = x;
}

/////////////////////////
////hgb
/////////////////////////
int HgbNetwork::Init() {
  try {
    conv1_weight = conv1_weight.reshape(0, required_param_nums);      //2*1
    conv1_bias = conv1_bias.reshape(0, 1);          //1*1
    conv1_weight.convertTo(conv1_weight, CV_32F);
    conv1_bias.convertTo(conv1_bias, CV_32F);

  } catch (std::exception& e) {
    ALGLogError << "Failed to construct HgbNetwork network params.";
    return -1;
  }
  return 0;
}

int HgbNetwork::Forward(const std::vector<float>& input_param_v,
                        std::vector<float>& output_v) {
  if (input_param_v.size() != required_param_nums) {
    ALGLogError << "For normal hgb Network, param size must be "
                << required_param_nums << "  but " << input_param_v.size()
                << " was given.";
    return -1;
  }
  float input1 = input_param_v[0] / norm_input1;
  float input2 = input_param_v[1] / norm_input1;
  float input3 = input_param_v[2] / norm_input1;
  cv::Mat input{input1, input2, input3};
  cv::Mat x;

  cv::Mat input_ = input.reshape(0, 1);  // 1*3
  x = input_ * conv1_weight;             // 1*3 3*4 -> 1*4
  x = x + conv1_bias;
//  sigmoid(x, x);
//  x = x * conv2_weight;  // 1*4 4*4 -> 1*4
//  x = x + conv2_bias;

  x = x * norm_output;
  output_v = x.reshape(0, 1);
  return 0;
}

/////////
//count
////////
int CellCountNetwork::Init() {
  if (ReshapeLinearNetworkParams(conv1_weight, conv1_bias)) {
    ALGLogError << "Failed to construct SphericalMcvNetwork network params.";
    return -1;
  }
}
int CellCountNetwork::Forward(const std::vector<float>& input_param_v,
                        std::vector<float>& output_v) {

  if (input_param_v.empty()) {
    ALGLogError << "For cell count, param size must be > 0, but empty was given";

    return -1;
  }

  cv::Mat input{input_param_v};
  input = input / norm_input1;

  cv::Mat x;
  ForwardLinearNetwork(conv1_weight, conv1_bias, input, x, (int)input_param_v.size());

  x = x * norm_output;
  output_v = x.reshape(0, 1);
  return 0;

}



}

}
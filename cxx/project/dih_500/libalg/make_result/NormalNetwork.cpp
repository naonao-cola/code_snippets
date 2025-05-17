//
// Created by y on 24-3-28.
//
#include <opencv2/core.hpp>
#include <numeric>

#include "TinnyNetwork.h"
//#include "DihLog.h"
#include "algLog.h"
namespace ALG_DEPLOY {
namespace PSEUDO_NETWORK {

/////////////////////////
////mcv
/////////////////////////
int NormalMcvNetwork::Init() {
  try {
    conv1_weight = conv1_weight.reshape(0, 3);  // 3*4
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
    ALGLogError << "Failed to construct NormalMcvNetwork network params.";
    return -1;
  }
  return 0;
}

int NormalMcvNetwork::Forward(const std::vector<float>& input_param_v,
                              std::vector<float>& output_v) {
  if (input_param_v.size() != required_param_nums) {
    ALGLogError << "For normal mcv Network, param size must be "
                << required_param_nums << "  but " << input_param_v.size()
                << " was given.";
    return -1;
  }
  float input1 = input_param_v[0] / norm_input1;
  float input2 = input_param_v[1] / norm_input2;
  float input3 = input1 * input2;
  cv::Mat input{input1, input2, input3};
  cv::Mat x;
  ForwardNormalNetwork(conv1_weight, conv1_bias, conv2_weight, conv2_bias,
                       conv3_weight, input, x);

  x = x * norm_output;
  output_v = x.reshape(0, 1);
  return 0;
}

/////////////////////////
////mpv
/////////////////////////
int NormalMpvNetwork::Init() {
  if (ReshapeNormalNetworkParams(conv1_weight, conv1_bias, conv2_weight,
                                 conv2_bias, conv3_weight)) {
    ALGLogError << "Failed to construct NormalMpvNetwork network params.";
    return -1;
  }
  return 0;
}

int NormalMpvNetwork::Forward(const std::vector<float>& input_param_v,
                              std::vector<float>& output_v) {
  if (input_param_v.empty()) {
    ALGLogError << "For normal mpv Network, param size must be > 0, but empty was given";

    return -1;
  }

  cv::Mat input{input_param_v};
  input = input / norm_input1;

  cv::Mat x;
  ForwardNormalNetwork(conv1_weight, conv1_bias, conv2_weight, conv2_bias,
                       conv3_weight, input, x, (int)input_param_v.size());

  x = x * norm_output;
  output_v = x.reshape(0, 1);
  return 0;
}

/////////////////////////
////rdw cv
/////////////////////////
int NormalRdwCvNetwork::Init() {
  if (ReshapeNormalNetworkParams(conv1_weight, conv1_bias, conv2_weight,
                                 conv2_bias, conv3_weight)) {
    ALGLogError << "Failed to construct NormalRdwCvNetwork network params.";
    return -1;
  }
  return 0;
}

int NormalRdwCvNetwork::Forward(const std::vector<float>& input_param_v,
                                std::vector<float>& output_v) {
  if (input_param_v.size() != required_param_nums) {
    ALGLogError << "For normal rdw_cv Network, param size must be "
                << required_param_nums << "  but " << input_param_v.size()
                << " was given.";
    return -1;
  }

  cv::Mat input{input_param_v};
  input = input / norm_input1;

  cv::Mat x;
  ForwardNormalNetwork(conv1_weight, conv1_bias, conv2_weight, conv2_bias,
                       conv3_weight, input, x);

  x = x * norm_output;
  output_v = x.reshape(0, 1);
  return 0;
}


/////////////////////////
////rdw sd
/////////////////////////
int NormalRdwSdNetwork::Init() {
  if (ReshapeNormalNetworkParams(conv1_weight, conv1_bias, conv2_weight,
                                 conv2_bias, conv3_weight)) {
    ALGLogError << "Failed to construct NormalRdwSdNetwork network params.";
    return -1;
  }
  return 0;
}

int NormalRdwSdNetwork::Forward(const std::vector<float>& input_param_v,
                                std::vector<float>& output_v) {
  if (input_param_v.size() != required_param_nums) {
    ALGLogError << "For normal rdw_sd Network, param size must be "
                << required_param_nums << "  but " << input_param_v.size()
                << " was given.";
    return -1;
  }

  cv::Mat input{input_param_v};
  input = input / norm_input1;

  cv::Mat x;
  ForwardNormalNetwork(conv1_weight, conv1_bias, conv2_weight, conv2_bias,
                       conv3_weight, input, x);

  x = x * norm_output;
  output_v = x.reshape(0, 1);
  return 0;
}



}
}
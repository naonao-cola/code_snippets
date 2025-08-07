//
// Created by y on 24-4-3.
//
#include "TinnyNetwork.h"
//#include "DihLog.h"
#include "algLog.h"
namespace ALG_DEPLOY{
namespace PSEUDO_NETWORK{

int SphericalMcvNetwork::Init() {
  if (ReshapeLinearNetworkParams(conv1_weight, conv1_bias)) {
    ALGLogError << "Failed to construct SphericalMcvNetwork network params.";
    return -1;
  }
  return 0;
}

int SphericalMcvNetwork::Forward(const std::vector<float>& input_param_v,
                              std::vector<float>& output_v) {
  if (input_param_v.empty()) {
    ALGLogError << "For normal  SphericalMcvNetwork, param size must be > 0, but empty was given";

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

/////////////////////////
////rdw cv
/////////////////////////
int SphericalRdwCvNetwork::Init() {
  if (ReshapeNormalNetworkParams(conv1_weight, conv1_bias, conv2_weight,
                                 conv2_bias, conv3_weight)) {
    ALGLogError << "Failed to construct NormalRdwCvNetwork network params.";
    return -1;
  }
  return 0;
}

int SphericalRdwCvNetwork::Forward(const std::vector<float>& input_param_v,
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

////pdw
int SphericalPdwCvNetwork::Init()
{
    if (ReshapeNormalNetworkParams(conv1_weight, conv1_bias, conv2_weight, conv2_bias, conv3_weight)) {
        ALGLogError << "Failed to construct NormalPdwCvNetwork network params.";
        return -1;
    }
    return 0;
}

int SphericalPdwCvNetwork::Forward(const std::vector<float>& input_param_v, std::vector<float>& output_v)
{
    if (input_param_v.size() != required_param_nums) {
        ALGLogError << "For normal PdwCv Network, param size must be " << required_param_nums << "  but " << input_param_v.size() << " was given.";
        return -1;
    }

    cv::Mat input{input_param_v};
    input = input / norm_input1;

    cv::Mat x;
    ForwardNormalNetwork(conv1_weight, conv1_bias, conv2_weight, conv2_bias, conv3_weight, input, x);
    x        = x * norm_output;
    output_v = x.reshape(0, 1);
    return 0;
}

/////////////////////////
////rdw sd
/////////////////////////
int SphericalRdwSdNetwork::Init() {
  if (ReshapeNormalNetworkParams(conv1_weight, conv1_bias, conv2_weight,
                                 conv2_bias, conv3_weight)) {
    ALGLogError << "Failed to construct NormalRdwSdNetwork network params.";
    return -1;
  }
  return 0;
}

int SphericalRdwSdNetwork::Forward(const std::vector<float>& input_param_v,
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

/////////////////////////
////mpv
/////////////////////////
/*int SphericalMpvNetwork::Init() {
  if (ReshapeNormalNetworkParams(conv1_weight, conv1_bias, conv2_weight,
                                 conv2_bias, conv3_weight)) {
    ALGLogError << "Failed to construct NormalMpvNetwork network params.";
    return -1;
  }
  return 0;
}*/

int SphericalMpvNetwork::Init() {
  if (ReshapeLinearNetworkParams(conv1_weight, conv1_bias)) {
    ALGLogError << "Failed to construct SphericalMcvNetwork network params.";
    return -1;
  }
  return 0;
}

int SphericalMpvNetwork::Forward(const std::vector<float>& input_param_v,
                              std::vector<float>& output_v) {
  if (input_param_v.empty()) {
    ALGLogError << "For normal mpv Network, param size must be > 0, but empty was given";

    return -1;
  }

  cv::Mat input{input_param_v};
  input = input / norm_input1;

  cv::Mat x;
//  ForwardNormalNetwork(conv1_weight, conv1_bias, conv2_weight, conv2_bias,
//                       conv3_weight, input, x, (int)input_param_v.size());

  ForwardLinearNetwork(conv1_weight, conv1_bias, input, x, (int)input_param_v.size());

  x = x * norm_output;
  output_v = x.reshape(0, 1);
  return 0;
}


}
}

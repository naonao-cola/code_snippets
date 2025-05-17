//
// Created by y on 24-3-28.
//

#ifndef TEST_LIBALG_TINNYNETWORK_H
#define TEST_LIBALG_TINNYNETWORK_H
#include <vector>
#include <opencv2/opencv.hpp>
namespace ALG_DEPLOY {
namespace PSEUDO_NETWORK {
void sigmoid(cv::Mat src, cv::Mat& dst);
/*!
 * 通用3层网络初始化
 * @param conv1_weight
 * @param conv1_bias
 * @param conv2_weight
 * @param conv2_bias
 * @param conv3_weight
 * @return
 */
int ReshapeNormalNetworkParams(cv::Mat& conv1_weight, cv::Mat& conv1_bias,
                               cv::Mat& conv2_weight, cv::Mat& conv2_bias,
                               cv::Mat& conv3_weight);

/*!
 * 通用3层网络推理
 * @param conv1_weight
 * @param conv1_bias
 * @param conv2_weight
 * @param conv2_bias
 * @param conv3_weight
 * @param input
 * @param output
 * @param batch_size
 */
void ForwardNormalNetwork(const cv::Mat& conv1_weight, const cv::Mat& conv1_bias,
                         const cv::Mat& conv2_weight, const cv::Mat& conv2_bias,
                         const cv::Mat& conv3_weight, const cv::Mat& input,
                         cv::Mat& output, int batch_size = 1);
/*!
 * 线形拟合
 * @param conv1_weight
 * @param conv1_bias
 * @return
 */
int ReshapeLinearNetworkParams(cv::Mat& conv1_weight, cv::Mat& conv1_bias);

void ForwardLinearNetwork(const cv::Mat& conv1_weight, const cv::Mat& conv1_bias,
                          const cv::Mat& input, cv::Mat& output,
                          int batch_size = 1);



class NormalNetwork {
 public:
  NormalNetwork() = default;
  virtual ~NormalNetwork() = default;
  /*!
   * 初始化接口,用于组织参数格式
   * @return
   */
  virtual int Init() = 0;

  /*!
   * 推理接口,用于推理并获取结果
   * @param input_param_v 输入数据
   * @param result_v
   * @return
   */
  virtual int Forward(const std::vector<float>& input_param_v,
                      std::vector<float>& output_v) = 0;
};



///////////////////
////general network
///////////////////
class HgbNetwork : public NormalNetwork {
 public:
  HgbNetwork() = default;
  ~HgbNetwork() override{};

  int Init() override;

  int Forward(const std::vector<float>& input_param_v,
              std::vector<float>& output_v) override;

 private:
  cv::Mat conv1_weight{-0.291, 2.2497, -0.0948};
  cv::Mat conv1_bias{0.0748};

  //  cv::Mat conv2_weight{-2.8881};
  //  cv::Mat conv2_bias{2.8752};

  const int required_param_nums = 3;
  const float norm_input1 = 1;
  const float norm_output = 100;
};



class CellCountNetwork : public NormalNetwork {
 public:
  CellCountNetwork() = default;
  ~CellCountNetwork() override{};

  int Init() override;

  int Forward(const std::vector<float>& input_param_v,
              std::vector<float>& output_v) override;

 private:
  //  cv::Mat conv1_weight{0.879714625*0.8835};
  cv::Mat conv1_weight{0.9074};
  cv::Mat conv1_bias{-0.1477};

  const int required_param_nums = 1;
  const float norm_input1 = 1;
  const float norm_output = 1;
};





///////////////////
////normal
///////////////////
class NormalMcvNetwork : public NormalNetwork {
 public:
  NormalMcvNetwork() = default;
  ~NormalMcvNetwork() override{};

  int Init() override;

  int Forward(const std::vector<float>& input_param_v,
              std::vector<float>& output_v) override;

 private:
  cv::Mat conv1_weight{-0.2080, 1.1787, 0.3284, -0.3995, -0.1844, -2.0602,
                       -0.6971, 1.6079, 0.5850, -0.4011, -0.8080, 0.5265};
  cv::Mat conv1_bias{0.2962, 0.9444, 0.4666, -1.1170};

  cv::Mat conv2_weight{-0.1875, 0.0079,  -0.0247, 0.3554,  -1.3440, -1.3179,
                       1.1578,  -1.6077, -0.5391, -0.4222, 0.1458,  -0.7583,
                       1.2956,  0.8728,  -0.9258, 1.0698};
  cv::Mat conv2_bias{-0.5365, -0.5635, -0.1633, -0.4451};

  cv::Mat conv3_weight{1.4867, 1.1952, -1.3927, 1.5872};  // 拟合方程参数

  const int required_param_nums = 2;  // 输入参数个数
  const float norm_input1 = 10000;    // 对应输入索引的norm值
  const float norm_input2 = 1000;     // 对应输入索引的norm值
  const float norm_output = 100;      // 对应输出索引的norm值
};

class NormalMpvNetwork : public NormalNetwork {
 public:
  NormalMpvNetwork() = default;
  ~NormalMpvNetwork() override{};

  int Init() override;

  int Forward(const std::vector<float>& input_param_v,
              std::vector<float>& output_v) override;

 private:
  cv::Mat conv1_weight{-2.6802, 0.5613, 0.8142, 0.8959};
  cv::Mat conv1_bias{2.6108, -0.4391, -0.8432, -0.5225};

  cv::Mat conv2_weight{
      -2.4015, -1.9613, -1.8608, 0.1832,  0.0777, 0.4762, -0.0383, -0.5789,
      0.4758,  0.2316,  0.0537,  -0.5859, 0.9161, 0.0721, 0.3345,  -0.1040,
  };
  cv::Mat conv2_bias{-0.5795, -0.2853, -0.1254, -0.1728};

  cv::Mat conv3_weight{1.8864, 1.3334, 1.1922, -0.4971};

  const int required_param_nums = 1;
  const float norm_input1 = 1000;

  const float norm_output = 10;
};

class NormalRdwCvNetwork : public NormalNetwork {
 public:
  NormalRdwCvNetwork() = default;
  ~NormalRdwCvNetwork() override{};

  int Init() override;

  int Forward(const std::vector<float>& input_param_v,
              std::vector<float>& output_v) override;

 private:
  cv::Mat conv1_weight{1.2374, -1.4347, -1.1412, 0.9278};
  cv::Mat conv1_bias{-1.0896, 1.2564, 0.1870, -0.7893};

  cv::Mat conv2_weight{-0.8819, -0.2063, 0.5225,  0.8393,
                       0.3081,  -0.1656,-1.2742, -1.0265,
                       -0.4213, -0.2292, -0.8812, -0.2549,
                       -0.2708, -0.0092, 0.3043,  0.3115};
  cv::Mat conv2_bias{-0.6152, -1.0898, -0.9369, -0.4562};

  cv::Mat conv3_weight{0.5814, 0.8361, 1.1918, 1.4228};

  const int required_param_nums = 1;
  const float norm_input1 = 1000;

  const float norm_output = 10;
};

class NormalRdwSdNetwork : public NormalNetwork {
 public:
  NormalRdwSdNetwork() = default;
  ~NormalRdwSdNetwork() override{};

  int Init() override;

  int Forward(const std::vector<float>& input_param_v,
              std::vector<float>& output_v) override;

 private:
  cv::Mat conv1_weight{-0.4477, 0.3463, -0.7095, -0.2908};
  cv::Mat conv1_bias{1.4306, -1.3713, 3.2006, -0.3561};

  cv::Mat conv2_weight{-0.8803, -0.0291, -0.3414, -1.0174,
                       0.2819,  1.4252,-0.6029, 0.3618,
                       -1.2810, 0.5367,  -1.1389, -1.5198,
                       -0.8217, 0.0698,  -0.4909, -0.5018};
  cv::Mat conv2_bias{-0.8932, 1.6338, -0.9049, -0.7290};

  cv::Mat conv3_weight{1.4859, 3.7961, 1.4135, 1.6106};

  const int required_param_nums = 1;
  const float norm_input1 = 10;

  const float norm_output = 10;
};


///////////////////
////Spherical
///////////////////
class SphericalMcvNetwork : public NormalNetwork {
 public:
  SphericalMcvNetwork() = default;
  ~SphericalMcvNetwork() override{};

  int Init() override;

  int Forward(const std::vector<float>& input_param_v,
              std::vector<float>& output_v) override;

 private:
  cv::Mat conv1_weight{0.2431};//290 -- 0.2140, 272-- 0.2431  256 -- 0.2743 240 -- 0.3119
  cv::Mat conv1_bias{-0.1494};


  const int required_param_nums = 1;  // 输入参数个数
  const float norm_input1 = 10;    // 对应输入索引的norm值
  const float norm_output = 100;      // 对应输出索引的norm值
};

class SphericalRdwCvNetwork : public NormalNetwork {
 public:
  SphericalRdwCvNetwork() = default;
  ~SphericalRdwCvNetwork() override{};

  int Init() override;

  int Forward(const std::vector<float>& input_param_v,
              std::vector<float>& output_v) override;

 private:
  cv::Mat conv1_weight{2.2260, -1.6016, -1.1895,  0.8488};
  cv::Mat conv1_bias{-1.9190,  0.9665,  0.9368, -0.3713};

  cv::Mat conv2_weight{1.6900,  1.3286,  1.0906, -0.1717,
                       -1.5055, -1.0963, -0.7696, -0.1374,
                       -0.7892, -0.9399, -0.6264, -0.2982,
                       0.5923,  0.5045, -0.2049,  0.1773};
  cv::Mat conv2_bias{-0.8738, -0.7787, -0.7458, -0.3940};

  cv::Mat conv3_weight{1.7032, 1.4277, 0.8108, -0.1388};

  const int required_param_nums = 1;
  const float norm_input1 = 10;

  const float norm_output = 10;
};

class SphericalRdwSdNetwork : public NormalNetwork {
 public:
  SphericalRdwSdNetwork() = default;
  ~SphericalRdwSdNetwork() override{};

  int Init() override;

  int Forward(const std::vector<float>& input_param_v,
              std::vector<float>& output_v) override;

 private:
  cv::Mat conv1_weight{ -2.0594, -1.5945,  0.4070,  1.7059};
  cv::Mat conv1_bias{0.8064,  0.4100, -0.2212, -0.5196};

  cv::Mat conv2_weight{0.8051, -1.6153, -0.5331, -0.2009,
                       0.4839, -1.2260, -0.6671, -0.4815,
                       -0.5371,  0.1961,  0.1293, -0.2280,
                       -0.8522,  1.3346, -0.0019,  0.1896};
  cv::Mat conv2_bias{0.3575,  0.1393, -0.1179, -0.2977};

  cv::Mat conv3_weight{ -1.4214, 2.3420, 0.6117, 0.2609};

  const int required_param_nums = 1;
  const float norm_input1 = 100;

  const float norm_output = 100;
};


class SphericalMpvNetwork : public NormalNetwork {
 public:
  SphericalMpvNetwork() = default;
  ~SphericalMpvNetwork() override{};

  int Init() override;

  int Forward(const std::vector<float>& input_param_v,
              std::vector<float>& output_v) override;

 private:
  cv::Mat conv1_weight{1.2795};
  cv::Mat conv1_bias{-0.1924};

  cv::Mat conv2_weight{
      -1.0128, -0.8128, -0.4820, -0.9971,
      -3.2353, -2.8607, -0.6382, -3.1462,
      2.2961,  1.8893, -4.4625,  2.2325,
      0.8880,  0.1574,  2.3153,  0.1995,
  };
  cv::Mat conv2_bias{-2.4903, -2.5562, -0.7351, -2.5095};

  cv::Mat conv3_weight{2.4667, 2.1778, 2.4936, 2.3639};

  const int required_param_nums = 1;
  const float norm_input1 = 10;

  const float norm_output = 10;



};


}
}
#endif  // TEST_LIBALG_TINNYNETWORK_H

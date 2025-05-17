//
// Created by y on 23-8-14.
//
//
// Created by y on 23-8-4.
//
#include "UnitTest.h"
#include "ParseXML.h"
#include "utils.h"
#include <dirent.h>
#include <fstream>
#include <sys/stat.h>

namespace ALG_LOCAL {
namespace UNIT {

// 根据传入的数据，初始化不同检测类型
bool UnitTest::InitOneType(
    const XML::UnitDetectTypeInitConfig &detect_type_config) {
  ALG_LOCAL::DetectType *detect_type_object;
  std::vector<std::string> detect_type_alg_name;
  auto detect_type_enum =
      detect_name_to_type_m.at(detect_type_config.detect_type_name);
  std::cout << __FILE__ << __LINE__ << " detect_type_enum: " << detect_type_enum
            << " name " << detect_type_config.detect_type_name << std::endl;

  if (detect_type_enum == DetectTypeName::HUMAN_TYPE &&
      detect_type_config.enable) {
    detect_type_object = new DetectHuman();
    std::cout << "Start init " + detect_type_config.detect_type_name
              << std::endl;
  } else if (detect_type_enum == DetectTypeName::CAT_TYPE &&
             detect_type_config.enable) { // 猫狗暂不支持,不允许测试
    std::cout << "Error, animal algs are not implemented." << std::endl;
    return false;
  } else { // 即使没有算法初始化，也不报错
    std::cout << "Warn, no detect type are initialized." << std::endl;
    return true;
  }
  std::cout << "初始化 InitParam 开始\n";
  InitParam init_param;
  init_param.detect_type = detect_type_enum;
  // 参数传递
  for (const auto &alg_configs : detect_type_config.alg_configs_v) {
    AlgParam alg_param;
    alg_param.enable = alg_configs.enable;
    alg_param.alg_type = alg_name_to_type_m.at(alg_configs.alg_name);
    alg_param.model_paths_v = alg_configs.model_paths_v;
    alg_param.init_param_float_v = alg_configs.float_param_v;
    init_param.alg_param_v.emplace_back(alg_param);

    // 保存已初始化的算法
    if (alg_param.enable) {
      detect_type_alg_name.emplace_back(alg_configs.alg_name);
    }
  }
  std::cout << "NNET Init >>" << std::endl;
  // 算法初始化
  if (!detect_type_object->Init(init_param)) {
    std::cout << "Init " + detect_type_config.detect_type_name + " type wrong"
              << std::endl;
    return false;
  }
  // 一次存储单个检测类型所需的参数，指针-debug-名称-类型下细分算法
  this->detect_type_object_v.push_back(detect_type_object);
  this->detect_type_debug_v.push_back(detect_type_config.debug);
  this->detect_type_names_v.push_back(detect_type_config.detect_type_name);
  this->detect_type_alg_names_v.push_back(detect_type_alg_name);
  return true;
}

bool UnitTest::Init(
    const XML::UnitTestDataDir &img_dir_config,
    const std::vector<XML::UnitDetectTypeInitConfig> &detect_type_config_v) {
  this->MapDetectName();
  this->MapAlgName();
  this->img_dir_config = img_dir_config;
  this->detect_type_config_v = detect_type_config_v;
  try {
    for (const auto &detect_type_config : this->detect_type_config_v) {
      // 初始化一种类型，如人或者动物
      if (detect_type_config.enable) {
        bool ret = InitOneType(detect_type_config);
        if (!ret) {
          return false;
        }
        // 不允许初始化多种检测类型
        break;
      }
    }
  } catch (std::exception &e) {
    std::cout << "Configured Wrong DetectType in XML.\n" << std::endl;
    std::cout << e.what() << std::endl;
    return false;
  }

  // 读取map可能报错
  std::cout << "Init all detect types succeed." << std::endl;
  return true;
}

void PrintResult(const std::vector<NNetResult> &detect_result_v) {
  if (detect_result_v.empty()) {
    std::cout << "Found zero objects." << std::endl;
  } else {
    std::map<std::string, int> objects;
    for (const auto iter : detect_result_v) {
      objects[iter.box.name] = objects[iter.box.name] + 1;
    }
    for (const auto iter : objects) {
      std::cout << iter.first << ":" << iter.second << std::endl;
    }
  }
}

void SaveMidImages(const std::vector<cv::Mat> &mid_result_mat_v,
                   const std::string &img_save_path) {

  std::size_t pos = img_save_path.find_last_of(".");
  std::string image_suffix = img_save_path.substr(pos);
  std::string image_path_prefix = img_save_path.substr(0, pos);
  if (mid_result_mat_v.size() == 1) {
    std::string save_path = image_path_prefix + image_suffix;
    std::cout << "image save path:" << save_path << std::endl;
    //    cv::imwrite(save_path, mid_result_mat_v[0]);
    cv::Mat img;

    cv::cvtColor(mid_result_mat_v[0], img, cv::COLOR_RGB2BGR);
    cv::flip(img, img, 0);
    // 反 正 反 正 反
    SaveImage(save_path, img);
  } else {
    for (int i = 0; i < mid_result_mat_v.size(); ++i) {
      std::string save_path =
          image_path_prefix + std::to_string(i) + image_suffix;
      std::cout << "image save path:" << save_path << std::endl;
      //      cv::imwrite(save_path, mid_result_mat_v[i]);
      // 为与opencv一致,存图函数为bgr格式
      cv::Mat img;

      cv::cvtColor(mid_result_mat_v[i], img, cv::COLOR_RGB2BGR);
      cv::flip(img, img, 0);
      SaveImage(save_path, img);
      //      SaveImage(save_path, mid_result_mat_v[i]);
    }
  }
}

void SaveMidValues(const std::vector<NNetResult> &detect_result_v,
                   const std::string &img_save_path) {
  size_t pos = img_save_path.find_last_of(".");
  std::string value_suffix = ".txt";
  std::string value_path_prefix = img_save_path.substr(0, pos);
  std::ofstream file(value_path_prefix + value_suffix);
  if (file.is_open()) { // 检查文件是否成功打开
    for (const auto &iter : detect_result_v) {
      // box已初始化才写入
      if (iter.write_rect_box) {
        file << iter.box.name + " " + std::to_string(iter.box.prop) + " " +
                    std::to_string(iter.box.left) + " " +
                    std::to_string(iter.box.top) + " " +
                    std::to_string(iter.box.right) + " " +
                    std::to_string(iter.box.bottom)
             << std::endl; // 写入内容
      }
      // 写入类别及概率
      for (int i = 0; i < iter.category_v.size(); ++i) {
        file << std::to_string(i) + " " + std::to_string(iter.category_v[i])
             << std::endl;
      }
    }
    file.close(); // 关闭文件
  } else {
    std::cout << "Fail to open the file: " + value_path_prefix + value_suffix
              << std::endl;
  }
}

void MonitorRestult(const bool &processed, const std::string &one_alg_name,
                    cv::Mat *img_brightness, cv::Mat *img_fluorescence,
                    const std::vector<cv::Mat> &mat_bright_result_v,
                    const std::vector<cv::Mat> &mat_fluo_result_v,
                    const std::string &save_path_bright,
                    const std::string &save_path_fluo,
                    const std::vector<NNetResult> &detect_result_v,
                    const bool &save_result) {
  if (processed) {
    PrintResult(detect_result_v);
    if (save_result) {
      if (img_brightness != nullptr) {
        SaveMidImages(mat_bright_result_v, save_path_bright);
        SaveMidValues(detect_result_v, save_path_bright);
      }
      if (img_fluorescence != nullptr) {
        SaveMidImages(mat_fluo_result_v, save_path_fluo);
        SaveMidValues(detect_result_v, save_path_fluo);
      }
    }
  }
}

// 生存所有类型，所有细分检测文件的目录
bool MakeSaveDirs(const std::string &save_dir,
                  const std::string &detect_type_name,
                  std::string &save_dir_with_time_and_type,
                  const std::vector<std::string> &alg_names_inside_detect) {

  std::string save_dir_bright, save_dir_fluo;
  DIR *mydir = opendir(save_dir.c_str()); // 打开目录
  if (mydir == nullptr) {
    if (mkdir(save_dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) != 0) {
      std::cout << "Failed to save dir " + save_dir << std::endl;
      return false;
    }
  }
  std::string save_dir_with_time =
      save_dir + "/" + std::to_string(time(nullptr)) + "/";
  save_dir_with_time_and_type =
      save_dir_with_time + "/" + detect_type_name + "/";

  if (mkdir(save_dir_with_time.c_str(),
            S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) != 0 ||
      mkdir(save_dir_with_time_and_type.c_str(),
            S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) != 0) {
    std::cout << "Failed to create detect type folder..." << std::endl;
    return false;
  }

  for (const auto &alg_name : alg_names_inside_detect) {
    std::string save_dir_alg = save_dir_with_time_and_type + alg_name + "/";
    save_dir_bright = save_dir_alg + "bright/";
    save_dir_fluo = save_dir_alg + "fluo/";

    std::cout << "Results will be saved to " + save_dir_with_time << std::endl;
    if (mkdir(save_dir_alg.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) !=
            0 ||
        mkdir(save_dir_bright.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) !=
            0 ||
        mkdir(save_dir_fluo.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) !=
            0) {
      std::cout << "Failed to create detect type folder..." << std::endl;
      return false;
    }
  }

  return true;
}

bool ReadMatAndName(const std::string &image_path, cv::Mat &img,
                    std::string &image_name, int &image_height,
                    int &image_width) {
  //  img = cv::imread( image_path, cv::IMREAD_UNCHANGED);
  std::cout << "输入图片路径： " << image_path << std::endl;
  img = cv::imread(image_path, 1);
  if (img.empty()) {
      std::cout << "Emtpy Bright image " << std::endl;
      return false;
  }
  cv::flip(img, img, 0);
  cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

  size_t pos = image_path.find_last_of("/");
  image_name = image_path.substr(pos + 1);
  image_height = img.rows;
  image_width = img.cols;
  std::cout << "Read image: " + image_path << " done." << std::endl;
  return true;
}

bool UnitTest::ForwardOneType(
    DetectType *detect_type, const bool &debug,
    const std::string &detect_type_name,
    const std::vector<std::string> &alg_names_inside_detect,
    const std::vector<std::string> &bright_image_path_v,
    const std::vector<std::string> &fluo_image_path_v) {
  std::cout << "正在推理的类型名称 " + detect_type_name << std::endl;
  std::cout << "是否调试模式: " << debug << std::endl;
  if (detect_type != nullptr) {
    // 设置保存路径
    std::string save_dir_with_time_and_type;
    if (debug) {
      if (!MakeSaveDirs(this->img_dir_config.save_dir, detect_type_name,
                        save_dir_with_time_and_type, alg_names_inside_detect)) {
        return false;
      }
    }
    // 读取图片及推理
    int max_image_nums = std::max(int(bright_image_path_v.size()),
                                  int(fluo_image_path_v.size()));
    std::cout << "图片的最大数量 : " << max_image_nums << std::endl;
    for (int i = 0; i < max_image_nums; ++i) {
      std::cout << "开始推理图片 ," << std::endl;
      std::string bright_image_name, fluo_image_name;
      cv::Mat image_bright_mat, image_fluo_mat;
      ForwardParam forward_param;
      // 明场和荧光场图像存在才读取
      if (i < bright_image_path_v.size()) {
        if (!ReadMatAndName(bright_image_path_v[i], image_bright_mat,
                            bright_image_name, forward_param.img_height,
                            forward_param.img_width)) {
          return false;
        }
        forward_param.img_brightness = &image_bright_mat;
      }
      if (i < fluo_image_path_v.size()) {
        if (!ReadMatAndName(fluo_image_path_v[i], image_fluo_mat,
                            fluo_image_name, forward_param.img_height,
                            forward_param.img_width)) {
          return false;
        }
        forward_param.img_fluorescence = &image_fluo_mat;
      }
      std::cout << "当前推理的明场图片路径：" << bright_image_path_v[i] << "\n";
      std::cout << "当前推理的荧光图片路径：" << fluo_image_path_v[i] << "\n";

      // 逐算法推理
      for (const auto &one_alg_name : alg_names_inside_detect) {
        try {
          forward_param.alg_type = this->alg_name_to_type_m.at(one_alg_name);
        } catch (std::exception &e) {
          std::cout << "Fail to Get alg type with regard to the " +
                           one_alg_name + " in " + detect_type_name + " type."
                    << std::endl;
          continue;
        }
        forward_param.detect_result_v.clear();
        forward_param.mat_bright_result_v.clear();
        forward_param.mat_fluo_result_v.clear();
        forward_param.processed = false;
        // 推理
        if (!detect_type->Forward(forward_param)) {
          std::cout << "Forward alg " + one_alg_name + " Failed" << std::endl;
          return false;
        }
        // 监控检测结果
        std::string save_path_bright = save_dir_with_time_and_type + "/" +
                                       one_alg_name + "/bright/" +
                                       bright_image_name;
        std::string save_path_fluo = save_dir_with_time_and_type + "/" +
                                     one_alg_name + "/fluo/" + fluo_image_name;

        // 保存结果
        MonitorRestult(
            forward_param.processed, one_alg_name, forward_param.img_brightness,
            forward_param.img_fluorescence, forward_param.mat_bright_result_v,
            forward_param.mat_fluo_result_v, save_path_bright, save_path_fluo,
            forward_param.detect_result_v, debug);
      }
      std::cout << "Processing image done.\n" << std::endl;
    }
    detect_type->RunAssistFunction();
    std::cout << "Forward " + detect_type_name + " done" << std::endl;

  } else {
    std::cout << "Error, empty object ptr." << std::endl;
  }
  return true;
}

bool UnitTest::ForwardAllDetectTypes() {
  std::cout << "Start Forward Unit test." << std::endl;
  //  std::vector<cv::String> bright_image_path_v, fluo_image_path_v;
  //  cv::glob(this->img_dir_config.input_bright_dir, bright_image_path_v);
  //  cv::glob(this->img_dir_config.input_fluo_dir, fluo_image_path_v);

  std::vector<std::string> bright_image_path_v, fluo_image_path_v;
  LoadImagePath(this->img_dir_config.input_bright_dir, bright_image_path_v);
  LoadImagePath(this->img_dir_config.input_fluo_dir, fluo_image_path_v);

  std::cout << "Use datas at \n" + this->img_dir_config.input_bright_dir + "\n"
            << this->img_dir_config.input_fluo_dir << std::endl;
  // 逐项推理已初始化的检测类型
  for (int i = 0; i < this->detect_type_object_v.size(); ++i) {

    if (!this->ForwardOneType(
            this->detect_type_object_v[i], this->detect_type_debug_v[i],
            this->detect_type_names_v[i], this->detect_type_alg_names_v[i],
            bright_image_path_v, fluo_image_path_v)) {
      return false;
    }
    this->detect_type_object_v[i]->GetStatisticResult();
  }
  // 其他的预留推理模型
  std::cout << "Forward Unit test succeed." << std::endl;
  return true;
}

void UnitTest::MapAlgName() {
  this->alg_name_to_type_m["rbc"] = AlgType::RBC;
  this->alg_name_to_type_m["wbc"] = AlgType::WBC;
  this->alg_name_to_type_m["wbc_single"] = AlgType::WBC_SINGLE;
  this->alg_name_to_type_m["wbc4"] = AlgType::WBC4;
  this->alg_name_to_type_m["wbc4_single"] = AlgType::WBC4_SINGLE;
  this->alg_name_to_type_m["plt"] = AlgType::PLT;
  this->alg_name_to_type_m["baso"] = AlgType::BASO;
  this->alg_name_to_type_m["ret"] = AlgType::RET;
  this->alg_name_to_type_m["somantic"] = AlgType::SOMATIC;
  this->alg_name_to_type_m["bacteria"] = AlgType::BACTERIA;
  this->alg_name_to_type_m["baso_clarity"] = AlgType::BASCLARITY;
  this->alg_name_to_type_m["grad_clarity"] = AlgType::GRADCLARITY;
  this->alg_name_to_type_m["rbc_volume"] = AlgType::RBC_VOLUME;
  this->alg_name_to_type_m["ai_clarity"] = AlgType::AI_CLARITY;
  this->alg_name_to_type_m["plt_volume"] = AlgType::PLT_VOLUME;
  this->alg_name_to_type_m["ai_clarity_far_near"] =
      AlgType::AI_CLARITY_FAR_NEAR;
  this->alg_name_to_type_m["milk_germ"] = AlgType::MILK_GERM;
  this->alg_name_to_type_m["milk_cell"] = AlgType::MILK_CELL;
  this->alg_name_to_type_m["rbc_volume_spherical_box"] =
      AlgType::RBC_VOL_SPH_BOX;
  this->alg_name_to_type_m["rbc_volume_spherical_seg"] =
      AlgType::RBC_VOL_SPH_SEG;
  this->alg_name_to_type_m["spherical_focal"] = AlgType::SPHERICAL_FOCAL;
  this->alg_name_to_type_m["classification_custom"] =
      AlgType::CLASSIFICATION_CUSTOM;
  this->alg_name_to_type_m["clarity_milk_boardline"] =
      AlgType::CLARITY_MLIK_BOARDLINE;
  this->alg_name_to_type_m["pla"] = AlgType::PLA;
}

void UnitTest::MapDetectName() {
  this->detect_name_to_type_m["milk"] = DetectTypeName::MILK_TYPE;
  this->detect_name_to_type_m["human"] = DetectTypeName::HUMAN_TYPE;
  this->detect_name_to_type_m["cat"] = DetectTypeName::CAT_TYPE;
}
UnitTest::~UnitTest() {
  for (auto &detect_type_object : this->detect_type_object_v) {
    delete detect_type_object;
  }
}

} // namespace UNIT
} // namespace ALG_LOCAL

#include "UnitTest.h"

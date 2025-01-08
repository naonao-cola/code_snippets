#pragma once
#include <fstream>
#include <ctime>
#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <nlohmann/json.hpp>
#include "utils.h"

namespace fs = std::filesystem;
using json = nlohmann::json;

class RoiLabelSave
{
public:
    void init_dir(std::string save_dir);
    void save_ocr_data(cv::Mat img,
                       std::string file_name,
                       const std::string &gt_txt);

    void save_ocr_data(cv::Mat img,
                       const json &in_param,
                       const json &task);

    void parse_inparam(const json &in_param,
                       const json &task,
                       std::string &file_name,
                       std::string &gt_label);

    virtual ~RoiLabelSave();

private:
    std::string get_savedir_name();

private:
    fs::path m_save_dir;
    fs::path m_txt_path;
    std::fstream m_label_fs;
    std::string m_cur_dir;
};

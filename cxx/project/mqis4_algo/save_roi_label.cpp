#include "save_roi_label.h"
#include "logger.h"

void RoiLabelSave::init_dir(std::string save_dir)
{
    m_save_dir = fs::path(save_dir);
    m_txt_path = m_save_dir;

    m_cur_dir = get_savedir_name();
    m_save_dir.append(m_cur_dir);

    if (fs::exists(m_save_dir) == false)
    {
        fs::create_directories(m_save_dir);
    }
    m_txt_path.append(m_cur_dir+"Label.txt");
    m_label_fs.open(m_txt_path.string(), std::ios::app);
}

RoiLabelSave::~RoiLabelSave()
{
    m_label_fs.close();
}

std::string RoiLabelSave::get_savedir_name()
{
    time_t now = time(0);
    std::tm *ltm = localtime(&now);
    std::string s = std::to_string(ltm->tm_year+1900) + "-"
                    + std::to_string(ltm->tm_mon+1) + "-"
                    + std::to_string(ltm->tm_mday)+ "-"
                    + std::to_string(ltm->tm_hour);
    return s;
}

void RoiLabelSave::parse_inparam(const json& in_param, const json& task, std::string& file_name, std::string& gt_label)
{
    std::string img_path_str = in_param["img_path"];
    std::string task_name = task["label"];
    std::filesystem::path img_pth(img_path_str.c_str());
    file_name = img_pth.filename().string();
    file_name = file_name.substr(0, file_name.size()-4);
    file_name = file_name + "_" + task_name + ".jpg";
    file_name = Utf8ToAnsi(file_name);
    std::string uper_file_name;
    std::transform(task_name.begin(), task_name.end(), std::back_inserter(uper_file_name), ::toupper);

    std::string lower_file_name;
    std::transform(task_name.begin(), task_name.end(), std::back_inserter(lower_file_name), ::tolower);

    gt_label = "###No label";
    if(in_param.contains(lower_file_name)) {
        if (!in_param[lower_file_name].empty())
        {
            gt_label = in_param[lower_file_name];
        }
    } else if(in_param.contains(uper_file_name)){
        if (!in_param[uper_file_name].empty()) {
            gt_label = in_param[uper_file_name];
        }
    }
}

void RoiLabelSave::save_ocr_data(cv::Mat img, const json& in_param, const json& task)
{
    std::string file_name;
    std::string gt_txt;
    parse_inparam(in_param, task, file_name, gt_txt);
    cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
    save_ocr_data(img, file_name, gt_txt);
}

void RoiLabelSave::save_ocr_data(cv::Mat img,
                                 std::string file_name,
                                 const std::string &gt_txt)
{
    fs::path pic_dir = m_save_dir;
    pic_dir.append(file_name);
    cv::imwrite(pic_dir.string(), img);
    if (m_label_fs.is_open())
    {
        m_label_fs << m_cur_dir+"/"+file_name << "\t" << Utf8ToAnsi(gt_txt) << std::endl;
    }
}
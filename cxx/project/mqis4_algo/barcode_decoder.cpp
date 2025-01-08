// #include <tvcore.h>
#include "barcode_decoder.h"
#include "logger.h"
#include "utils.h"
#include <fstream>
#include "DataCode.h"


BarcodeDecoder::BarcodeDecoder(json info):
    m_info(info)
{
}

void BarcodeDecoder::config(json config, RefImgTool *ref)
{
    m_ref = ref;
    m_config = config;
}

cv::Mat contrastStretch(cv::Mat srcImage, int pixMin, int pixMax)
{
    assert(pixMax > pixMin);
    // // 计算图像的最大最小值
    // double pixMin,pixMax;
    // cv::minMaxLoc(srcImage,&pixMin,&pixMax);
    // std::cout << "min_a=" << pixMin << " max_b=" << pixMax << std::endl;

    //create lut table
    cv::Mat lut( 1, 256, CV_8U);
    for( int i = 0; i < 256; i++ ){
        if (i < pixMin) lut.at<uchar>(i)= 0;
        else if (i > pixMax) lut.at<uchar>(i)= 255;
        else lut.at<uchar>(i)= static_cast<uchar>(255.0*(i-pixMin)/(pixMax-pixMin)+0.5);
    }
    //apply lut
    cv::Mat result_img;
    LUT( srcImage, lut, result_img);
    return result_img;
}

json BarcodeDecoder::decode(cv::Mat img, float scale) {
    cv::Mat resize_img;
    cv::resize(img, resize_img, cv::Size(int(img.cols * scale), int(img.rows*scale)));
    resize_img = contrastStretch(resize_img, 20, 180);

    std::string text = "";

    json params = {
        {"Num", 3},
        {"Timeout", -1}
    };
    Tival::DataCodeResult rst = Tival::DataCode::FindBarCode(img, params);
    if (rst.num_instances > 0) {
        text = rst.data_strings[0];
    }

    json result = {{"text", text}};

    return result;
}

json BarcodeDecoder::forward(cv::Mat gray_img)
{
    json all_out = json::array();
    for (auto task: m_config["task"]) {
        if (task["type"] == "barcode" || task["type"] == "qrcode") {
            LOG_INFO("Handle task: {}", Utf8ToAnsi(task.dump()));
            cv::Mat crop_img;
            json tfm_roi = m_ref->get_roi_img(gray_img, crop_img, task["roi"], 0, 0);
            // LOG_INFO("crop_img: {} x {}", crop_img.rows, crop_img.cols);

            json result = decode(crop_img);
            if (task["type"] == "barcode") {
                LOG_INFO("[Result]: {}", result.dump());
            } else {
                LOG_INFO("[Result]: QR read {}!", result.empty() ? "fail" : "success");
            }

            std::string name = task["name"];
            json out = {
                {"label", name},
                {"shapeType", "polygon"},
                {"points", m_ref->transform_result(tfm_roi)},
                {"result", result}
            };
            all_out.push_back(out);
        }
    }
    return all_out;
}

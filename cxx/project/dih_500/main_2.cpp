/**
 * @FilePath     : /DIH-ALG/main_2.cpp
 * @Description  :
 * @Author       : naonao
 * @Date         : 2025-06-20 10:07:47
 * @Version      : 0.0.1
 * @LastEditors  : naonao
 * @LastEditTime : 2025-08-04 09:07:35
 * @Copyright (c) 2025 by G, All Rights Reserved.
 **/
#include "libalg/libalgcell.h"
#include "libalg/libalgimm.h"

#include <algorithm>
#include <dirent.h>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <string.h>
#include <string>
#include <sys/types.h>
#include <unistd.h>
#include <vector>


void get_img_list(std::string imgDirPath, std::vector<std::string>& vimgPath)
{
    DIR*           pDir;
    struct dirent* ptr;
    if (!(pDir = opendir(imgDirPath.c_str()))) {
        std::cout << "Folder doesn't Exist! " << imgDirPath << std::endl;
        return;
    }
    else {
        std::cout << "Read " << imgDirPath << " succeed." << std::endl;
    }
    while ((ptr = readdir(pDir)) != 0) {
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0) {
            vimgPath.push_back(imgDirPath + "/" + ptr->d_name);
        }
    }
    std::sort(vimgPath.begin(), vimgPath.end());
    closedir(pDir);
}


std::vector<AlgCellImg_t> get_image(std::string path)
{

    std::vector<std::string> vimgPath;
    get_img_list(path, vimgPath);
    std::vector<AlgCellImg_t> image_vec;
    for (int i = 0; i < vimgPath.size(); i++) {
        AlgCellImg cellimage;
        cv::Mat    image = cv::imread(vimgPath[i]);
        cv::flip(image, image, 0);
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
        cellimage.data   = image.data;
        cellimage.height = image.rows;
        cellimage.width  = image.cols;
        cellimage.size   = image.cols * image.rows * image.channels();
        image_vec.push_back(cellimage);
    }
    return image_vec;
}

typedef struct tagBITMAP_FILE_HEADER
{
    unsigned short bfType;
    unsigned int   bfSize;
    unsigned short bfReserved1;
    unsigned short bfReserved2;
    unsigned int   bfOffBits;
} BITMAP_FILE_HEADER;

typedef struct tag_BITMAP_INFO_HEADER
{
    unsigned int   biSize;
    unsigned int   biWidth;
    unsigned int   biHeight;
    unsigned short biPlanes;
    unsigned short biBitCount;
    unsigned int   biCompression;
    unsigned int   biSizeImage;
    unsigned int   biXPelsPerMeter;
    unsigned int   biYPelsPerMeter;
    unsigned int   biClrUsed;
    unsigned int   biClrImportant;
} BITMAP_INFO_HEADER;

struct ImageBuf
{
    unsigned char* rgbBuf;
    int            bufLen;
    int            width;
    int            height;
    int            bitCount;
};

void SaveImage(const std::string& save_path, const cv::Mat& img)
{
    cv::Mat dst;
    cv::flip(img, dst, 0);
    int n_bytes = img.rows * img.cols * img.channels();

    ImageBuf buf;
    buf.rgbBuf = dst.data;
    buf.bufLen = n_bytes;
    buf.height = dst.rows;
    buf.width  = dst.cols;

    BITMAP_FILE_HEADER stBfh       = {0};
    BITMAP_INFO_HEADER stBih       = {0};
    unsigned long      dwBytesRead = 0;
    FILE*              file;

    stBfh.bfType    = (unsigned short)'M' << 8 | 'B';   // 定义文件类型
    stBfh.bfOffBits = sizeof(BITMAP_FILE_HEADER) + sizeof(BITMAP_INFO_HEADER);
    stBfh.bfSize    = stBfh.bfOffBits + buf.bufLen;   // 文件大小

    stBih.biSize          = sizeof(BITMAP_INFO_HEADER);
    stBih.biWidth         = buf.width;
    stBih.biHeight        = buf.height;
    stBih.biPlanes        = 1;
    stBih.biBitCount      = 24;
    stBih.biCompression   = 0L;
    stBih.biSizeImage     = 0;
    stBih.biXPelsPerMeter = 0;
    stBih.biYPelsPerMeter = 0;
    stBih.biClrUsed       = 0;
    stBih.biClrImportant  = 0;

    unsigned long dwBitmapInfoHeader = (unsigned long)40UL;

    file = fopen(save_path.c_str(), "wb");
    if (file) {
        fwrite(&stBfh, sizeof(BITMAP_FILE_HEADER), 1, file);
        fwrite(&stBih, sizeof(BITMAP_INFO_HEADER), 1, file);
        fwrite(buf.rgbBuf, buf.bufLen, 1, file);
        fclose(file);
    }
}

std::string int_specific_save_dir = "/data/user/dyno/img";

void AlgCellImageCallback(AlgCtxID_t                   ctx_id,
                          uint32_t                     group_idx,
                          uint32_t                     chl_idx,
                          uint32_t                     view_order,
                          uint32_t                     view_idx,
                          uint32_t                     processed_idx,
                          AlgCellStage_e               stage,
                          AlgCellImg_t*                img,
                          void*                        userdata,
                          const int&                   view_pair_idx,
                          std::map<std::string, float> call_back_params)
{


    //std::cout << "call back with mission type " << call_back_params[TASK_TYPE] << std::endl;
    std::stringstream ss;
    ss << 'g';
    ss << std::setw(1) << std::setfill('0') << group_idx;
    ss << 'c';
    ss << std::setw(1) << std::setfill('0') << chl_idx;
    ss << 'n';
    ss << std::setw(4) << std::setfill('0') << view_pair_idx;
    ss << 'v';
    ss << std::setw(4) << std::setfill('0') << view_idx;
    ss << 'p';
    ss << std::setw(2) << std::setfill('0') << processed_idx;
    ss << 'o';
    ss << std::setw(4) << std::setfill('0') << view_order;
    std::string img_name{ss.str()};
    cv::Mat     img_mat(int(img->height), int(img->width), CV_8UC3, img->data);
    if (img_mat.empty()){
        std::cout << "图片为空 "  << std::endl;
    }
    SaveImage(int_specific_save_dir + img_name + ".bmp", img_mat);
}

std::vector<std::string>  g_image_m;
std::vector<std::string>  g_image_a;


// 整体测试需要bgr格式的图
bool ReadImgToBuf(struct AlgCellImg* img_buf, const std::string& img_path, const bool& flip_img = true)
{
    cv::Mat img_mat;
    std::cout << "Pushing image " + img_path << std::endl;
    img_mat = cv::imread(img_path);
    if (img_mat.empty()) {
        std::cout << "ReadImgToBuf 194 Error, empty image" << std::endl;
        return false;
    }
    // 保存的图为上下翻转后的图,为模拟真实算法接受到的图,此处增加flip
    if (flip_img) {
        //cv::flip(img_mat, img_mat, 0);
    }

    int n_flag   = img_mat.channels() * 8;   // 一个像素的bits
    int n_height = img_mat.rows;
    int n_width  = img_mat.cols;
    int n_bytes  = n_height * n_width * n_flag / 8;   // 图像总的字节

    img_buf->data = new unsigned char[n_bytes];
    memcpy(img_buf->data, img_mat.data, n_bytes);
    if (img_buf->data == nullptr) {
        std::cout << "Fail to transform mat to unsigned char*" << std::endl;
    }
    img_buf->height = img_mat.rows;
    img_buf->width  = img_mat.cols;
    return true;
}

#include "replace_std_string.h"
// 测试荧光微球
int test_01()
{
    std::string version = AlgCell_Version();
    std::cout << "版本号： " << version << std::endl;
    AlgCtxID_t ctx_ptr = AlgCell_Init(3);

    // 可修改参数
    // app 路径
    std::string       cfg_path = "/data/alg_test/2reconstruct/data";
    std::vector<char> ret_model_info;
    // 可修改参数
    // 人
    AlgCellModeID_e mode_id = ALGCELL_MODE_HUMAN;
    int             ret;


    ret = AlgCell_RunConfigLoad(ctx_ptr, mode_id, cfg_path.c_str(), ret_model_info);
    std::string model_info_res;
    model_info_res.insert(model_info_res.begin(), ret_model_info.begin(), ret_model_info.end());
    std::cout << "模型信息： " << model_info_res << std::endl;

    // 可修改参数
    std::map<std::string, std::vector<float>> open_params = {
        {"open_param_debug", std::vector<float>{1.0}},       // 保存中间过程
        {"open_param_group_idx", std::vector<float>{1.0}},   // AlgCellGroupID 枚举
        {"open_param_qc", std::vector<float>{0.0}},
        {"open_param_img_h", std::vector<float>{3036.0}},
        {"open_param_img_w", std::vector<float>{4024.0}},
        {"open_param_img_h_um", std::vector<float>{252}},   // 像元
        {"open_param_alarm", std::vector<float>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {"open_param_dilution", std::vector<float>{50, 50, 10, 50, 50, 50, 10, 50}},
        {"open_param_task_append_att", std::vector<float>{0.0, 0.0}},
        {"open_param_calib", std::vector<float>{0.0}},
        {"open_param_pla", std::vector<float>{1.0}},
        {"open_param_cbc", std::vector<float>{0.0}}};

    // 可修改参数
    uint32_t func_mask = ALGCELL_FUNC_HEAMO;   // AlgFunc 枚举
    void* userdata = nullptr;
    ret = AlgCell_HeamoOpen(ctx_ptr, func_mask, AlgCellImageCallback, userdata, open_params);
    if (ret) {
        std::cout << "AlgCell_HeamoOpen 结果： " << ret << "打开失败"<<std::endl;
        return -1;
    }

    // 可修改参数
    uint32_t group_idx = ALG_CELL_GROUP_HUMAN;   // AlgCellGroupID
    uint32_t chl_idx   = 0;                      // 人医机型.xml
    get_img_list("/mnt/user/0/3A2A157A2A15347B/test/1", g_image_m);          // 明场
    get_img_list("/mnt/user/0/3A2A157A2A15347B/test/2", g_image_a);          // 暗场
    std::cout << "明场图片个数： " << g_image_m.size() << std::endl;
    std::cout << "暗场图片个数： " << g_image_a.size() << std::endl;

    AlgCell_HeamoPushHgb(ctx_ptr, {15110, 15097, 15193, 15107, 4825, 9692, 10309, 16850}, {1.f, 1.f, 1.f, 1.f});

    ret = AlgCell_HeamoSetPlaThreshold(5.0);
    for (int n = 0; n < g_image_m.size(); n++)
    {
        std::map<std::string, float> complementary_params = {
            {"view_pair_idx", float(n+1)},
            {"CHECK", 0.0},
            {"X", (n + 1) * 1.0},
            {"Y", (n + 1) * 2.0},
            {"Z", (n + 1) * 3.0},
        };
        std::vector<AlgCellImg>   img_v;

        auto img = new AlgCellImg;
        ReadImgToBuf(img, g_image_m[n]);
        img_v.emplace_back(*img);
        delete img;

        auto img2 = new AlgCellImg;
        ReadImgToBuf(img2, g_image_a[n]);
        img_v.emplace_back(*img2);
        delete img2;

        std::cout << "视野数： " << n << std::endl;
        ret = AlgCell_HeamoPushImage(ctx_ptr, img_v, group_idx, chl_idx, complementary_params);
        // 回收图像
        for (auto img : img_v) {
            delete[] img.data;
        }
    }
    std::cout << "获取结果中 "<< std::endl;
    AlgCellRst_t alg_result;
    ret = AlgCell_HeamoGetResult(ctx_ptr, alg_result, 0xFFFFFFFF);
    // 获取图片结果
    std::vector<ImageRet_t> result;
    try {
        ret = AlgCell_HeamoGetImgResult(ctx_ptr,  result, 0xFFFFFFFF);
        ret = AlgCell_HeamoClose(ctx_ptr);
    } catch (std::exception& e) {
        std::cout << "AlgCell_HeamoGetImgResult 结果： " << e.what() << std::endl;

    }

    std::cout << "result 大小： " << result.size() << std::endl;

    for (int m = 0; m < result.size(); m++) {
        std::cout << "item name  结果： " << result[m].image_name.data() << std::endl;
        std::cout << "item 图片  大小： " << result[m].base64_buff.size() << std::endl;

        // std::string   file_name = std::string("/mnt/user/0/3A2A157A2A15347B/test/3/").append(std::to_string(m)).append(".png");
        // std::cout << " 保存路径: "<< file_name << "\n";
        // std::ofstream out(file_name.c_str(),std::ios::binary);
        // if (!out) {
        //     std::perror("log.txt");
        //     return 1;
        // }

        // out.write(reinterpret_cast<const char*>(result[m].base64_buff.data()), result[m].base64_buff.size());
        // out.close();
    }


    std::cout << "alg_result.heamo.size() 结果： " << alg_result.heamo.size() << std::endl;
    int idx =0;
    for (const auto& item : alg_result.heamo) {
        std::cout << std::setw(10) << idx++ << std::setw(10) << item.name.data() << std::setw(10) << item.value << std::setw(10) << item.unit.data()
                  << std::setw(10) << item.open_flag<< std::endl;
    }

    ////荧光微球
    // MicFluInfo_t result;
    // ret = AlgCell_HeamoGetMicFluResult(ctx_ptr, result, 0xFFFFFFFF);
    // ret = AlgCell_HeamoClose(ctx_ptr);
    // std::cout << "result.cell_count.size() 结果： " << result.cell_count.size() << std::endl;
    // for (int m = 0; m < result.cell_count.size(); m++) {
    //     std::cout << "x: " << result.x_vec[m] << " y: " << result.y_vec[m] << " z: " << result.z_vec[m] << " count: " << result.cell_count[m]
    //               << std::endl;
    // }
    return ret;
}


std::vector<float> read_file(std::string data_path, int length = 400)
{
    std::vector<float> data_v;
    int data[length];
    memset(data, 0x00, length * sizeof(int));
    FILE*    file;
    uint32_t v1;
    char     line[256];
    int      i       = 0;
    int      datalen = 0;
    file             = fopen(data_path.c_str(), "r");
    if (file == NULL) {
        std::cout << "can not open data txt" << std::endl;
        return data_v;
    }
    char* end_ptr;
    while (fgets(line, sizeof(line), file)) {
        //		sscanf_s(line, "%d", &v1);//linux 不支持sscanf_s
        v1      = strtol(line, &end_ptr, 10);
        data[i] = v1;
        i++;
    }
    fclose(file);
    datalen = i;
    data_v  = std::vector<float>{data, data+length};
    return data_v;
}

// 读取免疫试剂卡信息
bool ReadImmuneCard(const std::string& card_path, std::string& card)
{
    FILE* file;
    file = fopen(card_path.c_str(), "r");
    if (file == nullptr) {
        return false;
    }

    do {
        card.push_back(fgetc(file));

    } while (!feof(file));

    fclose(file);
    card.at(card.length() - 1) = '\0';
    return true;
}

//免疫测试
int test_02(){

    std::string cardinfo;
    ReadImmuneCard("./1.card", cardinfo);
    std::vector<AlgImmData_t> data = read_file("./imm_data.txt");

    char        imm_alg_version[ALGIMM_LIB_VERSION_LENGTH];
    char        imm_qr_json_version[ALGIMM_LIB_VERSION_LENGTH];
    char        l_version[ALGIMM_LIB_VERSION_LENGTH];
    char        m_version[ALGIMM_LIB_VERSION_LENGTH];

    std::string version = AlgImm_Version(imm_alg_version, imm_qr_json_version, l_version, m_version);
    std::cout << "版本号： " << version << std::endl;

    AlgImmCtxID_t ctx_ptr  = AlgImm_Init();
    int           ret      = 0;
    std::string   cfg_path = "/data/alg_test/2reconstruct/data";
    AlgImm_RunConfigLoad(ctx_ptr,cfg_path.c_str());


    float            calib_coef = 1.0;
    uint32_t         group_idx  = 2;            // 1  di-50   2 DIH-500
    AlgImmFuncMask_t func_mask = 0x00000004;   // 1 AI-50  2 DI-50

    char encodetext[MAX_ENCODE_CARD_INFO] = "";
    memset(encodetext, 0x00, MAX_ENCODE_CARD_INFO);
    strcpy(encodetext, cardinfo.c_str());

    char decoded_card_info_char[MAX_DECODE_CARD_INFO];
    memset(&decoded_card_info_char, 0x00, MAX_DECODE_CARD_INFO);

    std::cout << "encodetext: " << encodetext << std::endl;
    std::cout << "encodetext sizeof: " << sizeof(encodetext) << std::endl;
    ret= AlgImm_GetCardInfo(ctx_ptr, group_idx, func_mask, encodetext, sizeof(encodetext), decoded_card_info_char);
    if(ret){
        std::cout << "Failed to AlgImm_GetCardInfo" << std::endl;
    }


    std::cout << "decoded_card_info_char: " << decoded_card_info_char << std::endl;

    std::string str_temp =
        R"({"DEV":2,"MASK":7,"Bnum":4,"Tnum":20,"BN":"CC25012201","PD":"2025-01-22","ED":"2026-01-22","Gain":2,"WTim":3,"MID":4,"PID":120,"AID":1,"PNam":"test","LCID":1,"LSta":60,"LErr":0,"TWin":14,"Line":[{"LID":1,"LOft":50,"LWid":60},{"LID":2,"LOft":50,"LWid":60}],"Chl":[{"CID":1,"Deci":0,"CNam":"test","Unit":" ","TMode":2,"CMode":1,"HLmt":["","0"],"LLmt":["","0"],"Equa":[],"ECnt":0,"PGat":["000","0.360","000","000"]}]})";

    ret = AlgImm_Open(ctx_ptr, 1, std::string(decoded_card_info_char), calib_coef);
    if (ret) {
        std::cout << "Failed to AlgImm_Open" << std::endl;
        return 0;
    }
    std::cout << "sucessed to AlgImm_Open" << std::endl;
    uint32_t samp_idx  = 0;
    ret    = AlgImm_PushData(ctx_ptr, data, 0, 0);
    if (ret) {
        std::cout << "Failed to AlgImm_PushData" << std::endl;
        return 0;
    }

    AlgImmRst_t result;
    ret = AlgImm_GetResult(ctx_ptr, result);
    if (ret) {
        std::cout << "Failed to AlgImm_GetResult" << std::endl;
        return 0;
    }
    AlgImm_Close(ctx_ptr);
}


// 测试聚焦
int test_03(int channel_input,std::string file_path)
{
    std::string version = AlgCell_Version();
    std::cout << "版本号： " << version << std::endl;
    AlgCtxID_t ctx_ptr = AlgCell_Init(3);
    // 可修改参数
    // app 路径
    std::string       cfg_path = "/data/alg_test/2reconstruct/data";
    std::vector<char> ret_model_info;
    // 可修改参数
    // 人
    AlgCellModeID_e mode_id = ALGCELL_MODE_HUMAN;
    int             ret;

    ret = AlgCell_RunConfigLoad(ctx_ptr, mode_id, cfg_path.c_str(), ret_model_info);
    std::string model_info_res;
    model_info_res.insert(model_info_res.begin(), ret_model_info.begin(), ret_model_info.end());
    std::cout << "模型信息： " << model_info_res << std::endl;

    // 可修改参数
    std::map<std::string, std::vector<float>> open_params = {
        {"open_param_debug", std::vector<float>{1.0}},       // 保存中间过程
        {"open_param_group_idx", std::vector<float>{1.0}},   // AlgCellGroupID 枚举
        {"open_param_qc", std::vector<float>{0.0}},
        {"open_param_img_h", std::vector<float>{3036.0}},
        {"open_param_img_w", std::vector<float>{4024.0}},
        {"open_param_img_h_um", std::vector<float>{252}},   // 像元
        {"open_param_alarm", std::vector<float>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {"open_param_dilution", std::vector<float>{50, 50, 10, 50, 50, 50, 10, 50}},
        {"open_param_task_append_att", std::vector<float>{0.0}},
        {"open_param_calib", std::vector<float>{0.0}},
        {"open_param_pla", std::vector<float>{0.0}},
    };

    // 可修改参数
    uint32_t func_mask = ALGCELL_FUNC_HEAMO;   // AlgFunc 枚举
    void*    userdata  = nullptr;
    // ret                = AlgCell_ClarityOpen(ctx_ptr, func_mask, AlgCellImageCallback, userdata, open_params);
    // if (ret) {
    //     std::cout << "AlgCell_HeamoOpen 结果： " << ret << "打开失败" << std::endl;
    //     return -1;
    // }

    // 可修改参数
    uint32_t group_idx = 0;                                // AlgCellGroupID
    uint32_t chl_idx   = channel_input;                    // 人医机型.xml
    //粗聚焦1
    //细聚焦 2
    get_img_list(file_path, g_image_m);   // 明场
    std::cout << "图片个数： " << g_image_m.size() << std::endl;

        ret = AlgCell_ClarityOpen(ctx_ptr, func_mask, AlgCellImageCallback, userdata, open_params);
        if (ret) {
            std::cout << "AlgCell_HeamoOpen 结果： " << ret << "打开失败" << std::endl;
          return 0;
        }
        for (int n = 0; n < g_image_m.size(); n++) {
            std::map<std::string, float> complementary_params = {
                {"view_pair_idx", float(n)},
            };

            auto img = new AlgCellImg;
            ReadImgToBuf(img, g_image_m[n]);
            std::cout << " 传入的图片路径: " << g_image_m[n] << std::endl;
            std::cout << " 传入的图片大小 宽: " << img->width << " 高 " << img->height << std::endl;

            ret                     = AlgCell_ClarityPushImage(ctx_ptr, group_idx, chl_idx, img, 1, complementary_params);
            AlgClarityValue_t value = 0;
            IMAGE_TYPE        type  = IMAGE_NORMAL;
            uint32_t          index = 0;
            std::cout << "处理图片数： " << n << std::endl;
            //粗聚焦
            ret = AlgCell_ClarityGetResultCoarse(ctx_ptr, &index, &value, type);
            std::cout << "粗聚焦 结果 value: " << value << std::endl;
            // 细聚焦
            ret = AlgCell_ClarityGetResultFarNear(ctx_ptr, &index, &value, type);
            // 回收图像
            delete img;
            std::cout << "细聚焦 结果 value: " << value << std::endl;
            std::cout << "图片类型   type: " << type << std::endl;
        }
        ret = AlgCell_ClarityClose(ctx_ptr);


    //ret = AlgCell_ClarityClose(ctx_ptr);
    std::cout << "AlgCell_ClarityClose" << std::endl;
    return ret;
}


//三个流道整体测试
int test_04(){
    std::string version = AlgCell_Version();
    std::cout << "版本号： " << version << std::endl;
    AlgCtxID_t ctx_ptr = AlgCell_Init(3);

    // app 路径
    std::string       cfg_path = "/data/alg_test/2reconstruct/data";
    std::vector<char> ret_model_info;
    // 可修改参数
    // 人
    AlgCellModeID_e mode_id = ALGCELL_MODE_HUMAN;
    int             ret;

    ret = AlgCell_RunConfigLoad(ctx_ptr, mode_id, cfg_path.c_str(), ret_model_info);
    std::string model_info_res;
    model_info_res.insert(model_info_res.begin(), ret_model_info.begin(), ret_model_info.end());
    std::cout << "模型信息： " << model_info_res << std::endl;

    // 可修改参数
    std::map<std::string, std::vector<float>> open_params = {
        {"open_param_debug", std::vector<float>{0.0}},       // 保存中间过程
        {"open_param_group_idx", std::vector<float>{1.0}},   // AlgCellGroupID 枚举
        {"open_param_qc", std::vector<float>{0.0}},
        {"open_param_img_h", std::vector<float>{3036.0}},
        {"open_param_img_w", std::vector<float>{4024.0}},
        {"open_param_img_h_um", std::vector<float>{252}},   // 像元
        {"open_param_alarm", std::vector<float>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0}},
        {"open_param_dilution", std::vector<float>{50, 50, 10, 50, 50, 50, 10, 50}},
        {"open_param_task_append_att", std::vector<float>{0.0,0.0}},
        {"open_param_calib", std::vector<float>{0.0}},
        {"open_param_pla", std::vector<float>{1.0}},
    };

    // 可修改参数
    uint32_t func_mask = ALGCELL_FUNC_HEAMO;   // AlgFunc 枚举
    void*    userdata  = nullptr;
    ret                = AlgCell_HeamoOpen(ctx_ptr, func_mask, AlgCellImageCallback, userdata, open_params);
    if (ret) {
        std::cout << "AlgCell_HeamoOpen 结果： " << ret << "打开失败" << std::endl;
        return -1;
    }
    //添加数据
    std::vector<std::string> bir_image__path_list;
    bir_image__path_list.push_back("");   //红细胞明场
    bir_image__path_list.push_back("");   // 白细胞明场
    bir_image__path_list.push_back("");   //嗜碱明场
    std::vector<std::string> ful_image__path_list;
    ful_image__path_list.push_back("");   // 红细胞暗场
    ful_image__path_list.push_back("");   // 白细胞暗场
    ful_image__path_list.push_back("");   // 嗜碱暗场




    //三个流道循环
    for (int channel_count = 0; channel_count < 3; channel_count++) {
        // 可修改参数
        uint32_t group_idx = ALG_CELL_GROUP_HUMAN;   // AlgCellGroupID
        uint32_t chl_idx   = channel_count;          // 人医机型.xml

        std::vector<std::string>  bir_path_vec;
        std::vector<std::string>  ful_path_vec;

        get_img_list(bir_image__path_list[channel_count], bir_path_vec);                  // 明场
        get_img_list(ful_image__path_list[channel_count], ful_path_vec);                  // 暗场


        std::cout << "明场图片个数： " << bir_path_vec.size() << std::endl;
        std::cout << "暗场图片个数： " << ful_path_vec.size() << std::endl;
        if (bir_path_vec.size() != ful_path_vec.size()) {
            std::cout << "明场和暗场图片个数不一致，退出测试,此时的流道是: " << channel_count << std::endl;
            AlgCell_HeamoClose(ctx_ptr);
            return -1;
        }
        //循环每个流道的明场和暗场
        for (int n = 0; n < bir_path_vec.size(); n++) {
            std::map<std::string, float> complementary_params = {
                {"view_pair_idx", float(n + 1)},
                {"CHECK", 0.0},
                {"X", (n + 1) * 1.0},
                {"Y", (n + 1) * 2.0},
                {"Z", (n + 1) * 3.0},
            };
            std::vector<AlgCellImg> img_v;

            auto img = new AlgCellImg;
            ReadImgToBuf(img, bir_path_vec[n]);
            img_v.emplace_back(*img);
            delete img;

            auto img2 = new AlgCellImg;
            ReadImgToBuf(img2, ful_path_vec[n]);
            img_v.emplace_back(*img2);
            delete img2;

            std::cout << "视野数： " << n << std::endl;
            ret = AlgCell_HeamoPushImage(ctx_ptr, img_v, group_idx, chl_idx, complementary_params);
            // 回收图像
            for (auto img : img_v) {
                delete[] img.data;
            }
        }
    }

    AlgCellRst_t result;
    ret = AlgCell_HeamoGetResult(ctx_ptr, result, 0xFFFFFFFF);

    std::vector<ImageRet_t> img_result;
    ret = AlgCell_HeamoGetImgResult(ctx_ptr, img_result, 0xFFFFFFFF);
    std::cout << "result 大小： " << img_result.size() << std::endl;
    ret = AlgCell_HeamoClose(ctx_ptr);
    for (int m = 0; m < img_result.size(); m++) {
        std::cout << "item name  结果： " << img_result[m].image_name.data() << std::endl;
        std::cout << "item 图片  大小： " << img_result[m].base64_buff.size() << std::endl;
    }
    std::cout << "result.heamo.size() 结果： " << result.heamo.size() << std::endl;
    int idx =0;
    for (const auto& item : result.heamo) {
        std::cout << std::setw(10) << idx++ << std::setw(10) << item.name.data() << std::setw(10) << item.value << std::setw(10) <<
        item.unit.data() << std::setw(10) << item.open_flag<< std::endl;
    }

    return 0;
}



#include <cstring>
#include <iostream>
int main(int argc, char* argv[])
{
    test_01();
    std::cout << "test_01() "<< std::endl;

    //免疫
    //test_02();
    // std::cout << "test_02() " << std::endl;

    // //聚焦测试
    // if(argc>3){
    //     std::cout << "输入的个数大于3个直接返回 "<<std::endl;
    //     return 0;
    // }
    // else{
    //     std::cout << "Usage: " << argv[0] << "  " << argv[1] << " " <<  argv[2]<<std::endl;
    // }
    // std::cout << "输入的通道是  "<< argv[1]<<" 输入的图片的路径 "<<argv[2]<<std::endl;
    // test_03(atoi(argv[1]), std::string (argv[2]));
    // std::cout << "test_03() 聚焦测试 " << std::endl;



    // test_04();
    // std::cout << "整体测试 test_04() " << std::endl;
    return 0;
}

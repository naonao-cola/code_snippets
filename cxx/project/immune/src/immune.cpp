//
// Created by y on 24-4-19.
//

#include <cstring>
#include <iostream>
#include <math.h>
#include <sstream>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <vector>


#include "alg.h"
#include "algjson.h"
#include "algpeak.h"
#include "immune.h"
#include "lua_port.h"
#include "qr_json.h"

// 算法初始化
Alg_HandleTypeDef handle;

// 获取C算法版本信息
int AlgImmuneGetVersion(char* alg_version, char* qr_json_version, char* lua_version,char* main_versioon,int version_length)
{
    // 参数检查
    if (NULL == alg_version || NULL == qr_json_version) {
        return ALG_ERR;
    }

    char* version;
    version = AlgGetVersion();
    for (int i = 0; i < version_length; ++i) {
        alg_version[i] = version[i];
        if (version[i] == '\0') {
            break;
        }
    }

    version = QrJsonGetVersion();
    for (int i = 0; i < version_length; ++i) {
        qr_json_version[i] = version[i];
        if (version[i] == '\0') {
            break;
        }
    }
    char* l_version = "V1.0.00.20241219";
    for (int i = 0; i < version_length; ++i) {
        lua_version[i] = l_version[i];
        if (l_version[i] == '\0') {
            break;
        }
    }

    char* m_version = "V1.0.00.20250626";

    for (int i = 0; i < version_length; ++i) {
        main_versioon[i] = m_version[i];
        if (m_version[i] == '\0') {
            break;
        }
    }

    return ALG_OK;
}

int AlgImmuneInitOut(const std::string& lua_file)
{
    return AlgInit(&handle, lua_file);
}

int GetCardInfo(uint32_t group, uint32_t mask, char* buf, uint32_t size, int* err, char* card_info)
{
    int ret = 0;
    // 参数检查
    if (NULL == buf || NULL == err || NULL == card_info) {
        ret = -1;
        return ret; // 参数检查出错
    }

    QRJsonCard_t card;
    memset(&card, 0x00, sizeof(QRJsonCard_t));

    // 根据掩码和机型系列查找对应的试剂卡信息
    if (0 != QRJson_SearchCard(&card, group, mask, buf, size, err)) {
        ret = -2;
        return ret; // 未找到与机型匹配的试剂卡信息
    }

    const char* err_string = NULL;
    if (0 != QRJson_EncodeCardInfo(card_info, MAX_DECODE_CARD_INFO, &card, &err_string)) {
        ret = -3;
        return ret; // 试剂卡信息解码出错
    }

    return ret;
}

/*!
 * 将输入数据保存在结果结构体中
 * @param input 输入数据
 * @param line_rst 线结果结构体
 * @return 0表示成功，-1表示失败
 */

int GetInputData(AlgInput* input, LineResult* line_rst)
{
    // 参数检查
    if (NULL == input || NULL == line_rst) {
        return ALG_ERR;
    }

    // 装入输入数据
    line_rst->input_length = input->length;
    for (int i = 0; i < input->length; ++i) {
        line_rst->input_data[i] = (float)input->data[i];
    }

    // 获取线id
    for (int i = 0; i < input->line_cnt; i++) {
        line_rst->single_line_rst[i].line_id = input->line_para.paras[i].line_id;
    }

    return ALG_OK;
}

int AlgImmuneCalculateOut(const std::string& card_info, const std::vector<float>& data_v, char* coef, AlgResultOut* algrstout, char* decoded_card_info)
{
    // 考虑到以后可能的免疫结构变动,虽然decoded_card_info与cardinfo相同,但仍在该函数内进行赋值
    if (NULL == algrstout || NULL == decoded_card_info) {
        return -1;
    }

    char cardinfo_char[MAX_DECODE_CARD_INFO];
    memset(cardinfo_char, 0x00, MAX_DECODE_CARD_INFO);
    strcpy(cardinfo_char, card_info.c_str());

    memcpy(decoded_card_info, cardinfo_char, MAX_DECODE_CARD_INFO);

    // 获取试剂卡信息，刷新输入参数，每一次测试加载
    AlgInput input;
    memset(&input, 0x00, sizeof(AlgInput));
    if (ALG_ERR == InitCardParam(&input, cardinfo_char)) {
        printf("获取试剂卡信息出错\n");
        return -2;
    }

    // 将数据导入input结构体
    int datalen = data_v.size();
    int data[MAX_SAMPLE_CNT];
    for (int i = 0; i < data_v.size(); ++i) {
        data[i] = (int)data_v[i];
    }

    if (ALG_ERR == LoadData(&input, data, datalen)) {
        printf("原始数据导入出错\n");
        return -3;
    }

    // 指针类型转换
    AlgResult* algrst = (AlgResult*)algrstout;
    // di500需要输入数据及line_id 初始化
    if (ALG_ERR == GetInputData(&input, &algrst->line_rst)) {
        printf("输入输出初始化\n");
        return -5;
    }


    // 结构体参数信息转换为json文件
    char json_buf[MAX_INPUT_JSON_BUFFER] = "";
    memset(json_buf, 0x00, MAX_INPUT_JSON_BUFFER);

    if (ALG_ERR == ImmunoCalculate(json_buf, algrst, cardinfo_char, input, &handle, coef)) {
        printf("Failed to execute immune calculate \n");
        return -6;
    }

    std::cout << "json buf:" << json_buf << std::endl;
    return 0;
}

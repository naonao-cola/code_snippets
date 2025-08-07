
#include <cstring>
#include <iostream>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "immune.h"

int main(int argc, char* argv[])
{
    std::cout << "procedure start" << std::endl;

    // 1.初始化原始数据和数据长度
    int data[MAX_SAMPLE_CNT];
    memset(data, 0x00, MAX_SAMPLE_CNT * sizeof(int));
    FILE* file;
    uint32_t v1;
    char line[256];
    int i = 0;
    int datalen = 0;
    file = fopen("./1.txt", "r");
    if (file == NULL) {
        std::cout << "can not open data txt" << std::endl;
        return -1;
    }
    char* end_ptr;
    while (fgets(line, sizeof(line), file)) {
        //		sscanf_s(line, "%d", &v1);//linux 不支持sscanf_s
        v1 = strtol(line, &end_ptr, 10);
        data[i] = v1;
        i++;
    }

    fclose(file);
    datalen = i;

    std::vector<float> data_v(std::begin(data), std::end(data));
    std::cout << "input data" << std::endl;
    for (const auto& iter : data_v) {
        std::cout << iter << " ";
    }
    std::cout << std::endl;

    // 2.读通过读取文本信息获取encodetext相关的文本信息
    char encodetext[MAX_ENCODE_CARD_INFO] = "";
    memset(encodetext, 0x00, MAX_ENCODE_CARD_INFO);

    i = 0;
    file = fopen("./1.card", "r");
    if (file == NULL) {
        std::cout << "can not open card" << std::endl;
        return -1;
    }
    do {
        *(encodetext + i) = fgetc(file);
        i++;
    } while (!feof(file));

    *(encodetext + i - 1) = '\0';
    fclose(file);
    std::cout << encodetext << std::endl;

    // 3.初始化参数
    int version_length = 30;
    char imm_alg_version[version_length];
    char imm_qr_json_version[version_length];
    char lua_alg_version[version_length];
    char main_version[version_length];

    AlgImmuneGetVersion(imm_alg_version, imm_qr_json_version, lua_alg_version, main_version, version_length);

    int ret = 0;
    ret = AlgImmuneInitOut("./alg.lua");
    if (ret) {
        std::cout << "Failed to init alg immune" << std::endl;
        return 0;
    }
    std::cout << "Alg version " << imm_alg_version << std::endl;
    std::cout << "Qr_json version " << imm_qr_json_version << std::endl;

    char* coef = "1.0";
    AlgResultOut algrstout;
    // 4. 获取对应机型的试剂卡信息，如果错误，报错“试剂卡信息与机型匹配错误”
    /*  group 和 mask 要跟随后台的设置改变*/
    // int group = 1; // 1  di-50   2 DIH-500
    // int mask = 0x00000001; // 1 AI-50  2 DI-50

    int group = 2; // 1  di-50   2 DIH-500
    int mask = 0x00000004; // 1 AI-50  2 DI-50

    char cardinfo_char[MAX_DECODE_CARD_INFO];
    memset(&cardinfo_char, 0x00, MAX_DECODE_CARD_INFO);
    int err = 0;
    std::cout << "GetCardInfo " << ret << std::endl;
    std::cout << "main GetCardInfo 95 group" << group << " mask: " << mask << std::endl;
    ret = GetCardInfo(group, mask, encodetext, sizeof(encodetext), &err, cardinfo_char);
    if (ret != ALG_OK_O) {
        std::cout << "Failed to get card content " << ret << std::endl;
        return -1; //
    }

    char decoded_card_info[MAX_DECODE_CARD_INFO];
    // 5.计算结果
    ret = AlgImmuneCalculateOut(std::string(cardinfo_char), data_v, coef, &algrstout, decoded_card_info);
    if (ret) {
        std::cout << "Failed to calculate immune" << std::endl;
        return 0;
    }
    std::cout << "Decoded card info " << decoded_card_info << std::endl;
    std::cout << "Processed succeed." << std::endl;
    return 0;
}

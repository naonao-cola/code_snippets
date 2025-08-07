
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>


#include "alg.h"
#include "algjson.h"
#include "algpeak.h"
#include "iostream"
#include "lua_port.h"
#include "string"


///
/// @brief
///     获取算法版本信息
///
/// @param[in]  iversion  算法版本
///
/// @return Alg_State 返回函数处理的成功与否的状态
///
/// @par History:
/// @xielihui，2022年3月28日，新建函数
///
char* AlgGetVersion()
{
    char* version = ALG_VERSION;
    return version;
}

///
/// @brief
///  算法初始化
///
///
/// @return
///
/// @par History:
/// @xielihui，2022年4月13日，新建函数
///
std::string lua_file_path;
int AlgInit(Alg_HandleTypeDef* handle, const std::string& lua_file)
{
    // 参数检查
    if (NULL == handle || lua_file.empty()) {
        return ALG_ERR;
    }

    memset(handle, 0x00, sizeof(Alg_HandleTypeDef));

    // 保存路径
    std::cout << "Use lua path " << lua_file << std::endl;

    lua_file_path = lua_file;

    // 判定文件是否存在
    if (access(lua_file_path.c_str(), F_OK) != 0) {
        std::cout << "Lua file not exist " << lua_file_path << std::endl;
        return ALG_ERR;
    }
    // 高斯模板，高斯参数7.0
    float sigma = 7.0;

    // 产生高斯模板
    GenerateGaussTemp(handle->gausstemp, &handle->gausstempsum, sigma, GAUSSAIN_FILTER_WINDOWN);
    GenerateGaussTemp(handle->comgausstemp, &handle->comgausstempsum, sigma, COM_GAUSSAIN_FILTER_WINDOWN);

    handle->init_flag = handle;
    std::cout << "immune lib init succeed " << lua_file << std::endl;

    return ALG_OK;
}

//  载入数据
int LoadData(AlgInput* input, int* data, int datalen)
{
    if (NULL == input || NULL == data || datalen <= 0) {
        return ALG_ERR;
    }

    input->length = datalen;
    for (int i = 0; i < datalen; i++) {
        input->data[i] = *(data + i);
    }

    return ALG_OK;
}

//  初始化输入参数
int InitCardParam(AlgInput* input, char* cardinfo)
{
    if (NULL == input || NULL == cardinfo) {
        return ALG_ERR;
    }
    // 获取线信息
    GetLineParam(cardinfo, input);

    // 获取方法学模式
    GetMethodParam(cardinfo, input);

    // 获取通道模式以及门限
    GetChannelModeAndGate(cardinfo, input);

    return ALG_OK;
}

// 结果初始化
int RstInit(AlgResult* algrst)
{
    if (NULL == algrst) {
        return ALG_ERR;
    }

    memset(algrst, 0x00, sizeof(AlgResult));

    for (int i = 0; i < MAX_LINE_CNT; i++) {
        strcpy(algrst->line_rst.single_line_rst[i].signal, "0.0000");
    }
    return ALG_OK;
}

// 数据处理
int ImmunoCalculate(char* json_buf, AlgResult* algrst, char* cardinfo, AlgInput input, Alg_HandleTypeDef* handle, char* coef)
{
    if (NULL == json_buf || NULL == algrst || NULL == cardinfo || NULL == handle || NULL == coef) {
        return ALG_ERR;
    }

    // 检测模式赋值（校准1，质控2，常规0）
    algrst->cal_flag = input.testmode;

    // 获取检测线的面积及其检测线的详情
    int ret = AlgGetLineResult(&input, &algrst->line_rst, handle, coef);
    if (ALG_ERR == ret) {
        printf("AlgGetLineResult failed!\n");
        return ALG_ERR;
    }

    // 组织检测线信息及其试剂卡信息
    AlgLuaInput luaipt;
    if (ALG_ERR == ConverLuaInput(&luaipt, input.line_para.cid, input.line_para.cerr, cardinfo, &algrst->line_rst, input.testmode, input.nature)) {
        printf("ConverLuaInput failed!\n");
        return ALG_ERR;
    }

    if (ALG_ERR == EncodeLuaInput(json_buf, MAX_INPUT_JSON_BUFFER, &luaipt)) {
        printf("EncodeLuaInput failed!\n");
        return ALG_ERR;
    }

    // 8.调用lua解析结果 原本在main中,为使用全局变量保存的lua文件名,移动至此处
    char opt_text[MAX_OUTPUT_JSON_BUFFER] = "";
    memset(opt_text, 0x00, MAX_OUTPUT_JSON_BUFFER);

    if (ALG_OK == LuaPort_CallMainDoFile(opt_text, lua_file_path.c_str(), json_buf)) {
        printf("%s\n", opt_text);
        // 检测结果json格式转换为算法结果结构体
        if (0 != DecodeLuaOutput(opt_text, &algrst->channel_rst)) {
            printf("wrong in decode output");
            return ALG_ERR;
        }
    }

    return ALG_OK;
}

// 获取方法学，MEATHOD_IM 0 ，MEATHOD_CG 1
int GetMethod(AlgInput input)
{
    return (input.method);
}

// 获取测试模式
int GetTestMode(AlgInput input)
{
    return (input.testmode);
}

// 获取高低增益，GAIN_LOW 0， GAIN_HIGH 1
int GetGain(AlgInput input)
{
    return (input.gain);
}
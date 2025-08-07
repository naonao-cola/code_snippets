
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>


#include "libalgimm.h"
// #include "DihLog.h"
#include "algLog.h"
#include "utils.h"



typedef struct AlgImmCtx
{
    AlgImmRst_t      rst;
    std::string      cardinfo;
    float            calib_coef;
    AlgImmFuncMask_t func_mask;
} AlgImmCtx_t;

#define ALGIMM_CTX_RST(ctx) ((ctx)->rst)                 // 结果缓存
#define ALGIMM_CTX_CARDINFO(ctx) ((ctx)->cardinfo)       // 试剂卡信息
#define ALGIMM_CTX_CALIB_COEF(ctx) ((ctx)->calib_coef)   // 校准系数
#define ALGIMM_CTX_FUNC_MASK(ctx) ((ctx)->func_mask)     // 功能掩码
#define MAX_IMM_LINE_CNT 5                               // 最大检测线为5（含T+C）


// 获取算法版本号
std::string AlgImm_Version(char* alg_version, char* qr_json_version, char* lua_version, char* main_versioon)
{
    std::ostringstream oss;
    char               imm_alg_version[ALGIMM_LIB_VERSION_LENGTH];
    char               imm_qr_json_version[ALGIMM_LIB_VERSION_LENGTH];
    char               l_version[ALGIMM_LIB_VERSION_LENGTH];
    char               m_version[ALGIMM_LIB_VERSION_LENGTH];

    AlgImmuneGetVersion(imm_alg_version, imm_qr_json_version, l_version, m_version, ALGIMM_LIB_VERSION_LENGTH);

    memcpy(alg_version, imm_alg_version, strlen(imm_alg_version));
    memcpy(qr_json_version, imm_qr_json_version, strlen(imm_qr_json_version));
    memcpy(lua_version, l_version, strlen(l_version));
    memcpy(main_versioon, m_version, strlen(m_version));

    // alg_version=imm_alg_version;
    // qr_json_version = imm_qr_json_version;
    // lua_version     = l_version;
    // main_versioon   = m_version;

    oss << imm_alg_version << " + " << imm_qr_json_version << " + " << l_version << " + " << l_version << " ";
    std::string imm_total_version = oss.str();
    ALGLogInfo << "ALGIMM VERSION " << imm_total_version;
    return imm_total_version;
}

// 获取免疫算法上下文
AlgImmCtxID_t AlgImm_Init(void)
{
    AlgImmCtx_t* ctx = new AlgImmCtx_t;
    return (AlgImmCtxID_t)ctx;
}

// 释放免疫算法上下文，-1表示失败，0表示成功
int AlgImm_DeInit(AlgImmCtxID_t ctx_id)
{
    AlgImmCtx_t* ctx = (AlgImmCtx_t*)ctx_id;
    if (ctx == NULL) {
        return -1;
    }
    delete ctx;
    return 0;
}

// 免疫算法导入配置文件，0表示成功，-1表示参数检查失败，-2表示初始化失败
int AlgImm_RunConfigLoad(AlgImmCtxID_t ctx_id, const char* cfg_path)
{
    AlgImmCtx_t* ctx = (AlgImmCtx_t*)ctx_id;
    // 参数检查
    if (ctx == NULL || nullptr == cfg_path) {
        return -1;
    }

    std::string immPath = std::string().append(cfg_path);
    immPath.append("/alg/lua/alg.lua");
    if (AlgImmuneInitOut(immPath) < 0) {
        ALGLogError << "Failed to init immune and use lua path " << immPath;
        return -1;
    }
    return 0;
}

// 免疫上下文释放
int AlgImm_RunConfigUnload(AlgImmCtxID_t ctx_id)
{
    AlgImmCtx_t* ctx = (AlgImmCtx_t*)ctx_id;
    if (ctx == NULL) {
        return -1;
    }
    return 0;
}
int AlgImm_GetCardInfo(AlgImmCtxID_t ctx_id, uint32_t group, uint32_t mask, char* buf, uint32_t size, char* card_info){

    AlgImmCtx_t* ctx = (AlgImmCtx_t*)ctx_id;
    int err;
    int ret = GetCardInfo(group, mask, buf, size, &err, card_info);
    return ret;
}

int AlgImm_Open(AlgImmCtxID_t ctx_id, AlgImmFuncMask_t func_mask, const std::string& cardinfo, const float calib_coef)
{
    AlgImmCtx_t* ctx = (AlgImmCtx_t*)ctx_id;
    if (ctx == NULL) {
        return -1;
    }
    ALGIMM_CTX_FUNC_MASK(ctx)  = func_mask;
    ALGIMM_CTX_CALIB_COEF(ctx) = calib_coef;
    ALGIMM_CTX_CARDINFO(ctx)   = cardinfo;
    memset(&ALGIMM_CTX_RST(ctx), 0, sizeof(AlgImmRst_t));
    // 增加初始化
    for (int i = 0; i < MAX_IMM_LINE_CNT; i++) {
        strcpy(ALGIMM_CTX_RST(ctx).line_rst.single_line_rst[i].signal, "0.0000");
    }

    ALGLogInfo << "start immune ";

    return 0;
}

int AlgImm_Close(AlgImmCtxID_t ctx_id)
{
    AlgImmCtx_t* ctx = (AlgImmCtx_t*)ctx_id;
    if (ctx == NULL) {
        return -1;   // 免疫结束上下文失败
    }
    ALGIMM_CTX_FUNC_MASK(ctx) = 0;
    memset(&ALGIMM_CTX_RST(ctx), 0, sizeof(AlgImmRst_t));
    return 0;
}

/*!
 * 将字符串中的指定字符串进行替换
 * @param str 需要进行处理的字符串
 * @param old_value 被替换字符串
 * @param new_value 替换为
 * @return
 */

void replace_all_distinct(std::string& str, const std::string& old_value, const std::string& new_value)
{
    for (std::string::size_type pos(0); pos != std::string::npos; pos += new_value.length()) {
        if ((pos = str.find(old_value, pos)) != std::string::npos)
            str.replace(pos, old_value.length(), new_value);
        else
            break;
    }
}

/*!
 * 二维码无法保存整个card，为保持已有代码，在此处增加原有信息
 * @param cardTxt
 * @return
 */

void AddCardHead(std::string& cardTxt)
{
    replace_all_distinct(cardTxt, "\"", "\\\"");
    cardTxt = "{\"reagent_card\":[{\"infomation\":\"" + cardTxt + "\"}]}";
}

static int AlgImm_Calculate(AlgImmRst_t& result, std::string& cardinfo, std::vector<AlgImmData_t>& data, const float& calib_coef)
{
    // 参数检查
    if (true == cardinfo.empty() || true == data.empty()) {
        return -1;
    }

    char coef_string[16] = "";
    snprintf(coef_string, 16, "%.4f", calib_coef);

    char decoded_card_info[MAX_DECODE_CARD_INFO];
    int  ret = AlgImmuneCalculateOut(cardinfo, data, coef_string, &result, decoded_card_info);

    if (ret) {
        ALGLogError << "Failed to calculate immune";
        for (const auto& iter : data) {
            ALGLogError << iter;
        }
        return -2;
    }
    ALGLogInfo << "Decoded card info: " << decoded_card_info;

    std::cout << "Decoded card info: " << decoded_card_info<<std::endl;
    return 0;
}


int AlgImm_PushData(AlgImmCtxID_t ctx_id, std::vector<AlgImmData_t>& data, uint32_t group_idx, uint32_t samp_idx)
{
    AlgImmCtx_t* ctx = (AlgImmCtx_t*)ctx_id;
    int          ret = 0;
    if (ctx == NULL) {
        ret = 0x30001;   // 免疫检测出错
        return ret;
    }
    int validflag = 0;


    if (ALGIMM_CTX_FUNC_MASK(ctx) & ALGIMM_FUNC_CALIB) {
        // std::cout << "AlgImm_PushData AlgImm_Calculate" << std::endl;
        if (0 == AlgImm_Calculate(ALGIMM_CTX_RST(ctx), ALGIMM_CTX_CARDINFO(ctx), data, ALGIMM_CTX_CALIB_COEF(ctx))) {
            // 计算成功，判定validflag 的值
            for (unsigned int i = 0; i < ctx->rst.channel_rst.channel_cnt; i++) {
                validflag = ctx->rst.channel_rst.single_channel_rst[i].validflag;
                if (0 == validflag) {
                    continue;
                }
                else if (-3 == validflag || -4 == validflag || -5 == validflag) {
                    return 0x30002  ; // 免疫检测卡失效
                }
                else if (-20 == validflag || -21 == validflag || -22 == validflag || -23 == validflag || -7 == validflag) {
                    return 0x30003;   // 校准不通过
                }
                else if (-1 == validflag || -2 == validflag || -6 == validflag || -8 == validflag || -9 == validflag || -10 == validflag ||
                         -11 == validflag || -12 == validflag) {
                    return 0x30004;   // 试剂卡信息解析出错
                }
            }
        }
        else {
            // 计算失败，直接返回-1
            ret = 0x30001;
            return ret;   // 免疫检测出错
        }
    }
    return ret;   // 返回成功0
}


void PrintImmune(const AlgImmRst_t& immune_result)
{
    ALGLogInfo << "Print immune result";
    std::cout<< "Print immune result"<<"\n";
    float immune_return_coef = 0.f;   // 各个通道校准系数相同
    ALGLogInfo << "Channel cnt " << immune_result.channel_rst.channel_cnt << std::endl;
    std::cout << "Channel cnt " << immune_result.channel_rst.channel_cnt << std::endl;
    for (int i = 0; i < immune_result.channel_rst.channel_cnt; ++i) {
        auto        channel_rst = immune_result.channel_rst.single_channel_rst[i];
        char        cdata[20];
        std::string scmode = channel_rst.mode;
        int         k      = 0;
        // mode 内含特殊字符,将引发上层报错,需要替换
        const char ad[2] = "/";
        for (k = 0; k < scmode.length(); k++) {
            if (cdata[k] == ad[0])
                cdata[k] = ad[0];
            else
                cdata[k] = scmode[k];
        }

        std::stringstream oss;
        cdata[k] = '\0';

        oss << " mode= " << cdata;
        memset(cdata, 0, sizeof(cdata));
        sprintf(cdata, "%.2f", float(channel_rst.validflag));
        oss << " validflag= " << cdata;

        memset(cdata, 0, sizeof(cdata));
        sprintf(cdata, "%.2f", strtod(channel_rst.signal, nullptr));
        oss << " signal= " << cdata;

        oss << " concentration= " << channel_rst.concentration;
        oss << " name= " << channel_rst.channel_name;

        // 替换result 为unit
        std::string unit{channel_rst.unit};
        ReplaceAllDistinct(unit, "/", "/");
        oss << " unit= " << unit;
        ALGLogInfo << oss.str();
        std::cout  << oss.str()<<"\n";

        immune_return_coef = float(channel_rst.coef);
    }

    ALGLogInfo << "Line cnt " << immune_result.line_rst.line_cnt << std::endl;
    std::cout << "Line cnt " << immune_result.line_rst.line_cnt << std::endl;
    // 线结果
    for (unsigned int i = 0; i < immune_result.line_rst.line_cnt; ++i) {
        std::stringstream oss;
        auto              line_rst = immune_result.line_rst.single_line_rst[i];
        char              cdata[20];
        sprintf(cdata, "%.2f", float(line_rst.line_id));
        oss << " line_id= " << cdata;

        memset(cdata, 0, sizeof(cdata));
        sprintf(cdata, "%.2f", float(line_rst.max_point));
        oss << " max_point= " << cdata;

        memset(cdata, 0, sizeof(cdata));
        sprintf(cdata, "%.2f", float(line_rst.max_value));
        oss << " max_value= " << cdata;

        memset(cdata, 0, sizeof(cdata));
        sprintf(cdata, "%.2f", float(line_rst.signal_start));
        oss << " signal_start= " << cdata;

        memset(cdata, 0, sizeof(cdata));
        sprintf(cdata, "%.2f", float(line_rst.signal_end));
        oss << " signal_end= " << cdata;

        memset(cdata, 0, sizeof(cdata));
        sprintf(cdata, "%.2f", float(line_rst.area));
        oss << " area= " << cdata;

        memset(cdata, 0, sizeof(cdata));
        sprintf(cdata, "%.2f", float(line_rst.base_line));
        oss << " base_line= " << cdata;

        memset(cdata, 0, sizeof(cdata));
        sprintf(cdata, "%.2f", float(immune_return_coef));

        oss << " immune_return_coef= " << cdata;
        ALGLogInfo << oss.str();
        std::cout << oss.str()<<std::endl;
        oss.clear();
    }

    std::cout << "input_length 318 " << immune_result.line_rst.input_length << std::endl;
    for (int i = 0; i < immune_result.line_rst.input_length;i++) {
        std::cout << immune_result.line_rst.input_data [i]<< ",";
    }
    std::cout << std::endl;
    std::cout << "filter_data 323 " << immune_result.line_rst.input_length << std::endl;
    for (int i = 0; i < immune_result.line_rst.input_length; i++) {
        std::cout << immune_result.line_rst.filter_data[i] << ",";
    }
    std::cout << std::endl;
}


int AlgImm_GetResult(AlgImmCtxID_t ctx_id, AlgImmRst_t& result)
{
    AlgImmCtx_t* ctx = (AlgImmCtx_t*)ctx_id;
    if (ctx == NULL) {
        return -1;   // 获取结果失败
    }
    std::cout << "PrintImmune"<<std::endl;
    //PrintImmune(ALGIMM_CTX_RST(ctx));
    memcpy(&result, &ALGIMM_CTX_RST(ctx), sizeof(AlgImmRst_t));

    // 指针类型转换
    AlgResult* algrst = (AlgResult*) (&result);

    // 非校准模式，赋值校准系数
    for (int idx = 0; idx < algrst->channel_rst.channel_cnt; ++idx) {
        float& calib_coef = algrst->channel_rst.single_channel_rst[idx].coef;
        if (1 != algrst->cal_flag) {   // 1是校准，2是质控，0是常规
            calib_coef = ALGIMM_CTX_CALIB_COEF(ctx);
        }
    }

    return 0;
}

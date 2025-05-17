
#include <cstring>
#include <string>
#include <iomanip>
#include <iostream>

#include "libalgimm.h"
//#include "DihLog.h"
#include "utils.h"
#include "algLog.h"


#define ALGIMM_WRAP_VERSION                        "V1.4.0"
#define ALGIMM_LIB_VERSION_LENGTH                  30
typedef struct AlgImmCtx {
    AlgImmRst_t rst;
    std::string cardinfo;
    float calib_coef;
    AlgImmFuncMask_t func_mask;
} AlgImmCtx_t;
#define ALGIMM_CTX_RST(ctx)                        ((ctx)->rst)            // 结果缓存
#define ALGIMM_CTX_CARDINFO(ctx)                ((ctx)->cardinfo)        // 试剂卡信息
#define ALGIMM_CTX_CALIB_COEF(ctx)                ((ctx)->calib_coef)        // 校准系数
#define ALGIMM_CTX_FUNC_MASK(ctx)                ((ctx)->func_mask)        // 功能掩码


std::string AlgImm_Version() {
    const char *date_char = {__DATE__};
    const char *time_char = {__TIME__};

    char result[200] = {0};
    char dt[20] = {0};
    sprintf(dt, "%s%s", date_char, time_char);
    int month = 0;
    const char *pMonth[] = {"Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov",
                            "Dec"};
    for (int i = 0; i < 12; i++) {
      if (memcmp(pMonth[i], dt, 3) == 0) {
        month = i + 1;
        break;
      }
    }

    std::ostringstream oss;
    oss << ALGIMM_WRAP_VERSION;
//    char imm_alg_version[ALGIMM_LIB_VERSION_LENGTH];
//    char imm_qr_json_version[ALGIMM_LIB_VERSION_LENGTH];
//
//    AlgImmuneGetVersion(imm_alg_version, imm_qr_json_version, ALGIMM_LIB_VERSION_LENGTH);
//
//    oss << " + "<<imm_alg_version<<" + "<<imm_qr_json_version<<" ";
//
//    oss << dt[7] << dt[8] << dt[9] << dt[10];
//    oss << std::setw(2) << std::setfill('0') << month;
//    auto day = (dt[4] == ' ' ? (dt[5] - '0') : ((dt[4] - '0') * 10) + (dt[5] - '0'));
//    oss << std::setw(2) << std::setfill('0') << day << "_";
//    oss << __TIME__;
//    oss << result;

    std::string imm_total_version = oss.str();
    ALGLogInfo<<"ALGIMM VERSION "<<imm_total_version;
    return imm_total_version;

}

AlgImmCtxID_t AlgImm_Init(void) {
    AlgImmCtx_t *ctx = new AlgImmCtx_t;
	printf("AlgImmCtx_t = 0x%x\r\n",ctx );
    return (AlgImmCtxID_t) ctx;
}

int AlgImm_DeInit(AlgImmCtxID_t ctx_id) {
    AlgImmCtx_t *ctx = (AlgImmCtx_t *) ctx_id;
    if (ctx == NULL) {
        return -1;
    }
    delete ctx;
    return 0;
}

int AlgImm_RunConfigLoad(AlgImmCtxID_t ctx_id, const char *cfg_path) {
    AlgImmCtx_t *ctx = (AlgImmCtx_t * ) ctx_id;

    if (ctx == NULL) {
        return -1;
    }

	if(cfg_path == nullptr){
		printf("cfg_path is null\r\n");
		return -2;
	}
        std::string immPath = std::string().append(cfg_path);
	immPath.append("/alg/lua/alg.lua");
        int ret = AlgImmuneInitOut(immPath);
        if(ret){
          ALGLogError<<"Failed to init immune, use lua path "<<immPath;
          return -3;
        }
    return 0;
    }

int AlgImm_RunConfigUnload(AlgImmCtxID_t ctx_id) {
    AlgImmCtx_t *ctx = (AlgImmCtx_t *) ctx_id;
    if (ctx == NULL) {
        return -1;
    }
    return 0;
}


int  AlgImm_Open(AlgImmCtxID_t ctx_id, AlgImmFuncMask_t func_mask, const std::string &cardinfo, const float calib_coef) {
      AlgImmCtx_t *ctx = (AlgImmCtx_t *) ctx_id;
      if (ctx == NULL) {
          return -1;
      }
      ALGIMM_CTX_FUNC_MASK(ctx) = func_mask;
      ALGIMM_CTX_CALIB_COEF(ctx) = calib_coef;
      ALGIMM_CTX_CARDINFO(ctx) = cardinfo;
      memset(&ALGIMM_CTX_RST(ctx), 0, sizeof(AlgImmRst_t));
      ALGLogInfo<<"start immune ";
      return 0;
  }

  int AlgImm_Close(AlgImmCtxID_t ctx_id) {
      AlgImmCtx_t *ctx = (AlgImmCtx_t *) ctx_id;
      if (ctx == NULL) {
          return -1;
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

void replace_all_distinct(std::string &str, const std::string &old_value, const std::string &new_value) {
    for (std::string::size_type pos(0); pos != std::string::npos; pos += new_value.length()) {
        if ((pos = str.find(old_value, pos)) != std::string::npos)
            str.replace(pos, old_value.length(), new_value);
        else break;
    }
}

/*!
 * 二维码无法保存整个card，为保持已有代码，在此处增加原有信息
 * @param cardTxt
 * @return
 */

void AddCardHead(std::string &cardTxt) {
    replace_all_distinct(cardTxt, "\"", "\\\"");
    cardTxt = "{\"reagent_card\":[{\"infomation\":\"" + cardTxt + "\"}]}";
}

static int AlgImm_Calculate(AlgImmRst_t &result, std::string &cardinfo, std::vector<AlgImmData_t> &data,
                            const float &calib_coef) {
    if (true == cardinfo.empty() || true == data.empty()) {
        return -1;
    }
    char coef_string[16] = "";
    snprintf(coef_string, 16, "%.2f", calib_coef);

    ALGLogInfo<<"Card info: "<<cardinfo;
    ALGLogInfo<<"Data size: "<<data.size();

    if(data.empty()){
      ALGLogInfo<<"Empty immune data";
      return -2;
    }
    if(data.size()>5){
      ALGLogInfo<<"Print 5 immune data ";
      for(int i=0; i<5; ++i){
        ALGLogInfo<<data[i]<<" ";
      }
    }

    char decoded_card_info[ MAX_DECODE_CARD_INFO];
    int ret = AlgImmuneCalculateOut( cardinfo, data,  coef_string,  &result, decoded_card_info);
    if(ret){
      ALGLogError<<"Failed to calculate immune";
      for(const auto& iter: data){
        ALGLogError<<iter;
      }
      return -3;
    }

    ALGLogInfo<<"Decoded card info: "<<decoded_card_info;

    ALGLogInfo<<"immune succeed";
    return 0;
}



int AlgImm_PushData(AlgImmCtxID_t ctx_id, std::vector<AlgImmData_t> &data, uint32_t group_idx, uint32_t samp_idx) {
    AlgImmCtx_t *ctx = (AlgImmCtx_t *) ctx_id;
    if (ctx == NULL) {
        return -1;
    }


    if (ALGIMM_CTX_FUNC_MASK(ctx) & ALGIMM_FUNC_CALIB) {
        return AlgImm_Calculate(ALGIMM_CTX_RST(ctx), ALGIMM_CTX_CARDINFO(ctx), data, ALGIMM_CTX_CALIB_COEF(ctx));
    } else {
        return AlgImm_Calculate(ALGIMM_CTX_RST(ctx), ALGIMM_CTX_CARDINFO(ctx), data, ALGIMM_CTX_CALIB_COEF(ctx));
    }
}



void PrintImmune(const AlgImmRst_t&  immune_result){
  ALGLogInfo<<"Print immune result";
  //免疫
  //通道结果
  float immune_return_coef = 0.f;//各个通道校准系数相同
  ALGLogInfo<<"Channel cnt "<<immune_result.channel_rst.channel_cnt<<std::endl;
  for(int i=0;i<immune_result.channel_rst.channel_cnt;++i){
    auto channel_rst = immune_result.channel_rst.single_channel_rst[i];
    char cdata[20];
    std::string scmode=channel_rst.mode;
    int k = 0;
    //mode 内含特殊字符,将引发上层报错,需要替换
    const char ad[2]="/";
    for (k = 0; k < scmode.length(); k++)
    {
      if(cdata[k]==ad[0])
        cdata[k]=ad[0];
      else
        cdata[k] =scmode[k] ;
    }
    std::stringstream oss;
    cdata[k] = '\0';

    oss<<" mode= "<<cdata;
    memset(cdata, 0, sizeof(cdata) );
    sprintf(cdata,"%.2f",float(channel_rst.validflag));
    oss<<" validflag= "<<cdata;

    memset(cdata, 0, sizeof(cdata) );
    sprintf(cdata,"%.2f", strtod(channel_rst.signal, nullptr));
    oss<<" signal= "<<cdata;
    //    memset(cdata, 0, sizeof(cdata) );
    //    sprintf(cdata,"%.2f",float(channel_rst.concentration_value));
    //    printf("concentration_value=%s  ",cdata);

    oss<<" concentration= "<<channel_rst.concentration;
    oss<<" name= "<<channel_rst.channel_name;

    //替换result 为unit
    std::string unit{channel_rst.unit};
    ReplaceAllDistinct(unit, "/", "/");
    oss<<" unit= "<<unit;
    ALGLogInfo<<oss.str();

    immune_return_coef = float(channel_rst.coef);

  }
  ALGLogInfo<<"Line cnt "<<immune_result.line_rst.line_cnt<<std::endl;
  //线结果
  for (unsigned int i=0; i<immune_result.line_rst.line_cnt;++i ){
    std::stringstream oss;
    auto line_rst = immune_result.line_rst.single_line_rst[i];
    char cdata[20];
    sprintf(cdata,"%.2f",float(line_rst.line_id));
    oss<<" line_id= "<<cdata;

    memset(cdata, 0, sizeof(cdata) );
    sprintf(cdata,"%.2f",float(line_rst.max_point));
    oss<<" max_point= "<<cdata;

    memset(cdata, 0, sizeof(cdata) );
    sprintf(cdata,"%.2f",float(line_rst.max_value));
    oss<<" max_value= "<<cdata;

    memset(cdata, 0, sizeof(cdata) );
    sprintf(cdata,"%.2f",float(line_rst.signal_start));
    oss<<" signal_start= "<<cdata;

    memset(cdata, 0, sizeof(cdata) );
    sprintf(cdata,"%.2f",float(line_rst.signal_end));
    oss<<" signal_end= "<<cdata;

    memset(cdata, 0, sizeof(cdata) );
    sprintf(cdata,"%.2f",float(line_rst.area));
    oss<<" area= "<<cdata;

    memset(cdata, 0, sizeof(cdata) );
    sprintf(cdata,"%.2f",float(line_rst.base_line));
    oss<<" base_line= "<<cdata;

    memset(cdata, 0, sizeof(cdata) );
    sprintf(cdata,"%.2f",float(immune_return_coef));

    oss<<" immune_return_coef= "<<cdata;
    ALGLogInfo<<oss.str();
    oss.clear();
  }

}


int AlgImm_GetResult(AlgImmCtxID_t ctx_id, AlgImmRst_t &result) {
    AlgImmCtx_t *ctx = (AlgImmCtx_t *) ctx_id;
    if (ctx == NULL) {
        return -1;
    }
    PrintImmune(ALGIMM_CTX_RST(ctx));
    memcpy(&result, &ALGIMM_CTX_RST(ctx), sizeof(AlgImmRst_t));
    ALGLogInfo<<"line 0 id "<<result.line_rst.single_line_rst[0].line_id;
    //非校准模式免疫检测的浓度计算校准系数coef采用传入的值，计算结果r_coef返回0.,校准模式则返回计算的值,
    //为表明该次计算真实所使用校准系数，此处进行修正,
    for (int idx = 0; idx < result.channel_rst.channel_cnt; ++idx) {
        float &calib_coef = result.channel_rst.single_channel_rst[idx].coef;
        if (calib_coef == 0.f) {
            calib_coef = ALGIMM_CTX_CALIB_COEF(ctx);
        }
    }
    return 0;
}


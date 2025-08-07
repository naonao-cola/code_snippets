//
// Created by y on 24-6-24.
//
#include <iostream>
#include <string>

#include "AlarmManager.h"
#include "Calibration.h"
#include "algLog.h"
#include "alg_heamo.h"
#include "replace_std_string.h"
#include "utils.h"
/**********************
 * 用于换算结果的宏定义
 *********************/

// 流道高度
#define VOLUME_RBC 100    //  RBC 流道视野高度单位um
#define VOLUME_WBC 150    //  WBC 单???野体积
#define VOLUME_HGB 1.0    //  HGB 单???野体积
#define VOLUME_BASO 150   //  HGB 单???野体积
#define VOLUME_RET 150    //  HGB 单???野体积

extern bool g_pla_flag;
bool g_cbc_flag;

    std::string heamo_save_dir;
void        Heamo_SetHeamoResultDir(const std::string& save_dir)
{
    heamo_save_dir = save_dir;
}

// 为暴露bug,暂不使用该修正函数
void HeamoResultClip(float& wbc,
                     float& neu_percent,
                     float& lym_percent,
                     float& mono_percent,
                     float& eos_percent,
                     float& baso_percent,
                     float& neu,
                     float& lym,
                     float& mono,
                     float& eos,
                     float& baso,
                     float& rbc,
                     float& hgb,
                     float& hct,
                     float& mcv,
                     float& mch,
                     float& mchc,
                     float& rdw_cv,
                     float& rdw_sd,
                     float& plt,
                     float& pct,
                     float& mpv,
                     float& nrbc,
                     float& ret,
                     float& ret_percent)
{
    ClipNumber(wbc, wbc, 0, 999);
    ClipNumber(neu_percent, neu_percent, 0, 100);
    ClipNumber(lym_percent, lym_percent, 0, 100);
    ClipNumber(mono_percent, mono_percent, 0, 100);
    ClipNumber(eos_percent, eos_percent, 0, 100);
    ClipNumber(baso_percent, baso_percent, 0, 100);

    ClipNumber(neu, neu, 0, 999);
    ClipNumber(lym, lym, 0, 999);
    ClipNumber(mono, mono, 0, 999);
    ClipNumber(eos, eos, 0, 999);
    ClipNumber(baso, baso, 0, 999);

    ClipNumber(rbc, rbc, 0, 999);
    ClipNumber(hgb, hgb, 0, 99999);
    ClipNumber(hct, hct, 0, 100);
    ClipNumber(mcv, mcv, 0, 999);
    ClipNumber(mch, mch, 0, 99999);
    ClipNumber(mchc, mchc, 0, 99999);

    ClipNumber(rdw_cv, rdw_cv, 0, 100);
    ClipNumber(rdw_sd, rdw_sd, 0, 999);

    ClipNumber(plt, plt, 0, 99999);
    ClipNumber(pct, pct, 0, 999);
    ClipNumber(mpv, mpv, 0, 999);
    ClipNumber(nrbc, nrbc, 0, 999);
    ClipNumber(ret, ret, 0, 999);
    ClipNumber(ret_percent, ret_percent, 0, 100);
}

#define TASK_ATT_REQUIRED_SIZE 2
int MakeHeamoUpLoadParams(const bool&               qc,
                          const std::vector<float>& task_att_v,
                          HeamoResultCallback_f     callback,
                          void*                     userdata,
                          const float&              WBC_value,
                          const float&              NEUT_percentage,
                          const float&              LYMPH_percentage,
                          const float&              MONO_percentage,
                          const float&              EOS_percentage,
                          const float&              BASO_percentage,
                          const float&              NEUT_value,
                          const float&              LYMPH_value,
                          const float&              MONO_value,
                          const float&              EOS_value,
                          const float&              BASO_value,
                          const float&              RBC_value,
                          const float&              HGB_value,
                          const float&              HCT_value,
                          const float&              MCV_value,
                          const float&              MCH_value,
                          const float&              MCHC_value,
                          const float&              RDW_CV_value,
                          const float&              RDW_SD_value,
                          const float&              NRBC_value,
                          const float&              NRBC_percentage,
                          const float&              PLT_value,
                          const float&              MPV_value,
                          const float&              PCT_value,
                          const float&              RET_value,
                          const float&              RET_percentage,
                          const float&              PDW_value,
                          const float&              PV_value,
                          const float&              PV1_value,
                          const float&              PV2_value,
                          const float&              Pv3_value,
                          const float&              PV4_value,
                          const float&              PV5_value,
                          const float&              PV6_value,
                          const float&              PLA_R_VALUE,
                          const float&              PLA_W_VALUE,
                          const float&              PV1_PERCENT,
                          const float&              PV2_PERCENT,
                          const float&              PV3_PERCENT,
                          const float&              PV4_PERCENT,
                          const float&              PV5_PERCENT,
                          const float&              PV6_PERCENT)
{

    if (task_att_v.size() != TASK_ATT_REQUIRED_SIZE) {
        ALGLogError << "Task att must be " << TASK_ATT_REQUIRED_SIZE << " "
                    << "but " << task_att_v.size() << "was given ";
        std::cout << "Task att must be " << TASK_ATT_REQUIRED_SIZE << " "
        << "but " << task_att_v.size() << "was given \n";
        return -1;
    }
    const bool  required_ret  = static_cast<bool>(task_att_v[0]);
    const bool  required_nrbc = static_cast<bool>(task_att_v[1]);
    std::string WBC_value_s, NEUT_percentage_s, LYMPH_percentage_s, MONO_percentage_s, EOS_percentage_s, BASO_percentage_s, NEUT_value_s,
        LYMPH_value_s, MONO_value_s, EOS_value_s, BASO_value_s, RBC_value_s, HGB_value_s, HCT_value_s, MCV_value_s, MCH_value_s, MCHC_value_s,
        RDW_CV_value_s, RDW_SD_value_s, NRBC_value_s, NRBC_percentage_s, PLT_value_s, MPV_value_s, PCT_value_s, RET_value_s, RET_percentage_s,
        PDW_value_s, PV_value_s, PV1_value_s, PV2_value_s, PV3_value_s, PV4_value_s, PV5_value_s, PV6_value_s, PLA_R_value_s, PLA_W_value_s,
        PV1_PERCENT_s, PV2_PERCENT_s, PV3_PERCENT_s, PV4_PERCENT_s, PV5_PERCENT_s, PV6_PERCENT_s;

    // 数据截断
    CutFloatToString(WBC_value, 1, WBC_value_s);
    CutFloatToString(NEUT_percentage, 0, NEUT_percentage_s);
    CutFloatToString(LYMPH_percentage, 0, LYMPH_percentage_s);
    CutFloatToString(MONO_percentage, 1, MONO_percentage_s);
    CutFloatToString(EOS_percentage, 1, EOS_percentage_s);

    CutFloatToString(BASO_percentage, 1, BASO_percentage_s);
    CutFloatToString(NEUT_value, 1, NEUT_value_s);
    CutFloatToString(LYMPH_value, 1, LYMPH_value_s);
    CutFloatToString(MONO_value, 1, MONO_value_s);
    CutFloatToString(EOS_value, 2, EOS_value_s);

    CutFloatToString(BASO_value, 2, BASO_value_s);
    CutFloatToString(RBC_value, 1, RBC_value_s);
    CutFloatToString(HGB_value, 0, HGB_value_s);
    CutFloatToString(HCT_value, 0, HCT_value_s);
    CutFloatToString(MCV_value, 0, MCV_value_s);

    CutFloatToString(MCH_value, 0, MCH_value_s);
    CutFloatToString(MCHC_value, 0, MCHC_value_s);
    CutFloatToString(RDW_CV_value, 2, RDW_CV_value_s);
    CutFloatToString(RDW_SD_value, 2, RDW_SD_value_s);
    CutFloatToString(NRBC_value, 3, NRBC_value_s);
    CutFloatToString(NRBC_percentage, 3, NRBC_percentage_s);

    CutFloatToString(PLT_value, 0, PLT_value_s);
    CutFloatToString(MPV_value, 2, MPV_value_s);
    CutFloatToString(PCT_value, 2, PCT_value_s);
    CutFloatToString(RET_value, 3, RET_value_s);
    CutFloatToString(RET_percentage, 1, RET_percentage_s);
    CutFloatToString(PDW_value, 3, PDW_value_s);
    // 疟原虫
    CutFloatToString(PV_value,  0, PV_value_s);
    CutFloatToString(PV1_value, 0, PV1_value_s);
    CutFloatToString(PV2_value, 0, PV2_value_s);
    CutFloatToString(Pv3_value, 0, PV3_value_s);
    CutFloatToString(PV4_value, 0, PV4_value_s);
    CutFloatToString(PV5_value, 0, PV5_value_s);
    CutFloatToString(PV6_value, 0, PV6_value_s);

    CutFloatToString(PV1_PERCENT, 2, PV1_PERCENT_s);
    CutFloatToString(PV2_PERCENT, 2, PV2_PERCENT_s);
    CutFloatToString(PV3_PERCENT, 2, PV3_PERCENT_s);
    CutFloatToString(PV4_PERCENT, 2, PV4_PERCENT_s);
    CutFloatToString(PV5_PERCENT, 2, PV5_PERCENT_s);
    CutFloatToString(PV6_PERCENT, 2, PV6_PERCENT_s);

    CutFloatToString(PLA_R_VALUE, 0, PLA_R_value_s);
    CutFloatToString(PLA_W_VALUE, 0, PLA_W_value_s);


    // 按是否质控上传不同结果
    if (qc) {
        if (g_cbc_flag)
        {
            HEAMO_RESULT(callback, userdata, WBC_KEY_NUM, "10^9/L", WBC_value_s.c_str())
            HEAMO_RESULT(callback, userdata, NE_KEY_PERCENTAGE, "%", "-")
            HEAMO_RESULT(callback, userdata, LY_KEY_PERCENTAGE, "%", "-")
            HEAMO_RESULT(callback, userdata, MO_KEY_PERCENTAGE, "%", "-")
            HEAMO_RESULT(callback, userdata, EO_KEY_PERCENTAGE, "%", "-")

            HEAMO_RESULT(callback, userdata, BA_KEY_PERCENTAGE, "%", "-")
            HEAMO_RESULT(callback, userdata, NE_KEY_NUM, "10^9/L", "-")
            HEAMO_RESULT(callback, userdata, LY_KEY_NUM, "10^9/L", "-")
            HEAMO_RESULT(callback, userdata, MO_KEY_NUM, "10^9/L", "-")
            HEAMO_RESULT(callback, userdata, EO_KEY_NUM, "10^9/L", "-")

            HEAMO_RESULT(callback, userdata, BA_KEY_NUM, "10^9/L", "-")
            HEAMO_RESULT(callback, userdata, RBC_KEY_NUM, "10^12/L", RBC_value_s.c_str())
            HEAMO_RESULT(callback, userdata, HB_KEY_VALUE, "g/L", HGB_value_s.c_str())
            HEAMO_RESULT(callback, userdata, HCT_KEY_VALUE, "%", HCT_value_s.c_str())
            HEAMO_RESULT(callback, userdata, MCV_KEY_VALUE, "fL", MCV_value_s.c_str())

            HEAMO_RESULT(callback, userdata, MCH_KEY_VALUE, "pg", MCH_value_s.c_str())
            HEAMO_RESULT(callback, userdata, MCHC_KEY_VALUE, "g/L", MCHC_value_s.c_str())
            HEAMO_RESULT(callback, userdata, RDW_CV_KEY_VALUE, "%", "-")
            HEAMO_RESULT(callback, userdata, RDW_SD_KEY_VALUE, "fL", "-")
            HEAMO_RESULT(callback, userdata, NRBC_KEY_NUM, "10^9/L", "-")
            HEAMO_RESULT(callback, userdata, NRBC_KEY_PERCENTAGE, "%", "-")

            HEAMO_RESULT(callback, userdata, PLT_KEY_NUM, "10^9/L", PLT_value_s.c_str())
            HEAMO_RESULT(callback, userdata, MPV_KEY_VALUE, "fL", "-")
            HEAMO_RESULT(callback, userdata, PCT_KEY_VALUE, "%", "-")
            HEAMO_RESULT(callback, userdata, "PDW", "fL", PDW_value_s.c_str())
        }
        if (g_pla_flag) {
            HEAMO_RESULT(callback, userdata, "PLS", "/uL", PV_value_s.c_str())

            HEAMO_RESULT(callback, userdata, "Pr#", "/uL", PV1_value_s.c_str())
            HEAMO_RESULT(callback, userdata, "Pr%", "%", PV1_PERCENT_s.c_str())

            HEAMO_RESULT(callback, userdata, "Pt#", "/uL", PV2_value_s.c_str())
            HEAMO_RESULT(callback, userdata, "Pt%", "%",   PV2_PERCENT_s.c_str())

            HEAMO_RESULT(callback, userdata, "Ps#", "/uL", PV3_value_s.c_str())
            HEAMO_RESULT(callback, userdata, "Ps%", "%",  PV3_PERCENT_s.c_str())

            HEAMO_RESULT(callback, userdata, "Pg#", "/uL", PV4_value_s.c_str())
            HEAMO_RESULT(callback, userdata, "Pg%", "%", PV4_PERCENT_s.c_str())

            HEAMO_RESULT(callback, userdata, "Px#", "/uL", PV5_value_s.c_str())
            HEAMO_RESULT(callback, userdata, "Px%", "%",   PV5_PERCENT_s.c_str())
            if (PV6_value>0) {
                HEAMO_RESULT(callback, userdata, "E-rosetting#", "/uL", PV6_value_s.c_str())
                HEAMO_RESULT(callback, userdata, "E-rosetting%", "%", PV6_PERCENT_s.c_str())
            }

            HEAMO_RESULT(callback, userdata, "PLS/5m RBC", "/uL", PLA_R_value_s.c_str())
            HEAMO_RESULT(callback, userdata, "PLS/8k WBC", "/uL", PLA_W_value_s.c_str())
        }

        if (required_ret) {
            HEAMO_RESULT(callback, userdata, RET_KEY_NUM, "10^12/L", "-")
            HEAMO_RESULT(callback, userdata, RET_KEY_PERCENTAGE, "%", "-")
            // TODO 需要确认是否有问题
            HEAMO_RESULT(callback, userdata, NRBC_KEY_NUM, "10^9/L", NRBC_value_s.c_str())
            HEAMO_RESULT(callback, userdata, NRBC_KEY_PERCENTAGE, "%", NRBC_percentage_s.c_str())
        }
    }
    else {
        if (g_cbc_flag){
            HEAMO_RESULT(callback, userdata, WBC_KEY_NUM, "10^9/L", WBC_value_s.c_str())
            HEAMO_RESULT(callback, userdata, NE_KEY_PERCENTAGE, "%", NEUT_percentage_s.c_str())
            HEAMO_RESULT(callback, userdata, LY_KEY_PERCENTAGE, "%", LYMPH_percentage_s.c_str())
            HEAMO_RESULT(callback, userdata, MO_KEY_PERCENTAGE, "%", MONO_percentage_s.c_str())
            HEAMO_RESULT(callback, userdata, EO_KEY_PERCENTAGE, "%", EOS_percentage_s.c_str())

            HEAMO_RESULT(callback, userdata, BA_KEY_PERCENTAGE, "%", BASO_percentage_s.c_str())
            HEAMO_RESULT(callback, userdata, NE_KEY_NUM, "10^9/L", NEUT_value_s.c_str())
            HEAMO_RESULT(callback, userdata, LY_KEY_NUM, "10^9/L", LYMPH_value_s.c_str())
            HEAMO_RESULT(callback, userdata, MO_KEY_NUM, "10^9/L", MONO_value_s.c_str())
            HEAMO_RESULT(callback, userdata, EO_KEY_NUM, "10^9/L", EOS_value_s.c_str())

            HEAMO_RESULT(callback, userdata, BA_KEY_NUM, "10^9/L", BASO_value_s.c_str())
            HEAMO_RESULT(callback, userdata, RBC_KEY_NUM, "10^12/L", RBC_value_s.c_str())
            HEAMO_RESULT(callback, userdata, HB_KEY_VALUE, "g/L", HGB_value_s.c_str())
            HEAMO_RESULT(callback, userdata, HCT_KEY_VALUE, "%", HCT_value_s.c_str())
            HEAMO_RESULT(callback, userdata, MCV_KEY_VALUE, "fL", MCV_value_s.c_str())

            HEAMO_RESULT(callback, userdata, MCH_KEY_VALUE, "pg", MCH_value_s.c_str())
            HEAMO_RESULT(callback, userdata, MCHC_KEY_VALUE, "g/L", MCHC_value_s.c_str())
            HEAMO_RESULT(callback, userdata, RDW_CV_KEY_VALUE, "%", RDW_CV_value_s.c_str())
            HEAMO_RESULT(callback, userdata, RDW_SD_KEY_VALUE, "fL", RDW_SD_value_s.c_str())

            ALGLogInfo << "Force to display nrbc cause not decided to make nrbc optional";
            // HEAMO_RESULT(callback, userdata, NRBC_KEY_NUM, "10^9/L", NRBC_value_s.c_str())
            // HEAMO_RESULT(callback, userdata, NRBC_KEY_PERCENTAGE, "%", NRBC_percentage_s.c_str())

            HEAMO_RESULT(callback, userdata, PLT_KEY_NUM, "10^9/L", PLT_value_s.c_str())
            HEAMO_RESULT(callback, userdata, MPV_KEY_VALUE, "fL", MPV_value_s.c_str())
            HEAMO_RESULT(callback, userdata, PCT_KEY_VALUE, "%", PCT_value_s.c_str())
            HEAMO_RESULT(callback, userdata, "PDW", "fL", PDW_value_s.c_str())
        }

        if (g_pla_flag) {
            HEAMO_RESULT(callback, userdata, "PLS", "/uL", PV_value_s.c_str())
            HEAMO_RESULT(callback, userdata, "Pr#", "/uL", PV1_value_s.c_str())
            HEAMO_RESULT(callback, userdata, "Pr%", "%", PV1_PERCENT_s.c_str())
            HEAMO_RESULT(callback, userdata, "Pt#", "/uL", PV2_value_s.c_str())
            HEAMO_RESULT(callback, userdata, "Pt%", "%", PV2_PERCENT_s.c_str())
            HEAMO_RESULT(callback, userdata, "Ps#", "/uL", PV3_value_s.c_str())
            HEAMO_RESULT(callback, userdata, "Ps%", "%", PV3_PERCENT_s.c_str())
            HEAMO_RESULT(callback, userdata, "Pg#", "/uL", PV4_value_s.c_str())
            HEAMO_RESULT(callback, userdata, "Pg%", "%", PV4_PERCENT_s.c_str())
            HEAMO_RESULT(callback, userdata, "Px#", "/uL", PV5_value_s.c_str())
            HEAMO_RESULT(callback, userdata, "Px%", "%", PV5_PERCENT_s.c_str())
            if (PV6_value > 0) {
                HEAMO_RESULT(callback, userdata, "E-rosetting#", "/uL", PV6_value_s.c_str())
                HEAMO_RESULT(callback, userdata, "E-rosetting%", "%", PV6_PERCENT_s.c_str())
            }
            HEAMO_RESULT(callback, userdata, "PLS/5m RBC", "/uL", PLA_R_value_s.c_str())
            HEAMO_RESULT(callback, userdata, "PLS/8k WBC", "/uL", PLA_W_value_s.c_str())
        }

        if (required_ret) {
            HEAMO_RESULT(callback, userdata, RET_KEY_NUM, "10^12/L", RET_value_s.c_str())
            HEAMO_RESULT(callback, userdata, RET_KEY_PERCENTAGE, "%", RET_percentage_s.c_str())
            //TODO 需要确认是否有问题
            HEAMO_RESULT(callback, userdata, NRBC_KEY_NUM, "10^9/L", NRBC_value_s.c_str())
            HEAMO_RESULT(callback, userdata, NRBC_KEY_PERCENTAGE, "%", NRBC_percentage_s.c_str())
        }
    }
    return 0;
}

// cnt->WBC, cnt->NEU, cnt->EOS, cnt->MONO, cnt->LYM, cnt->BASO
// 后续可能需要报错
int CountWbcFour(float& wbc, float& neu, const float& eos, const float& mono, const float& lym, const float& baso)
{
    // 白细胞通道将neu与baso均视为neu, baso在嗜碱流道检测,
    // 因此真实neu需要减去baso,
    neu = neu - baso;
    if (neu < 0) {
        neu = 0;
        ALGLogWarning << "neu has fewer nums than baso, so the result might be wrong.";
    }
    wbc = neu + eos + mono + lym + baso;
    return 0;
}
template<class T>
void FindMaxValue(std::vector<T> nums_v, T& max)
{
    for (const auto& iter : nums_v) {
        if (iter > max) {
            max = iter;
        }
    }
}

template<class T>
void WriteDataToCsv(const std::vector<std::vector<T>>& datas, const int& max_nums, std::ofstream& out_file)
{
    for (int i = 0; i < max_nums; ++i) {
        for (const auto& iter : datas) {
            if (i < iter.size()) {
                out_file << iter[i] << ",";
            }
            else {
                out_file << ",";
            }
        }
        out_file << std::endl;
    }
}
#include <chrono>
#include <future>
void write_to_file(const std::string& csv_save_path, int max_img_nums, std::vector<std::vector<int>> datas)
{
    try {
        std::ofstream out_file(csv_save_path, std::ios::out);
        out_file << "rbc" << ","
                 << "plt" << ","
                 << "ret" << ","
                 << "wbc" << ","
                 << "neu" << ","
                 << "lym" << ","
                 << "mono" << ","
                 << "eos" << ","
                 << "baso" << ","
                 << "nrbc" << ","
                 << "plas" << ","
                 << "pr" << ","
                 << "pt" << ","
                 << "ps" << ","
                 << "pg" << ","
                 << "px" << ","
                 << "E-rosetting" << "," << std::endl;
        WriteDataToCsv(datas, max_img_nums, out_file);
        out_file.flush();
        out_file.close();
        return;
    }
    catch (std::exception& e) {
        ALGLogError << "failed to write csv data in " << csv_save_path;
        return;
    }
}
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::seconds;
using std::chrono::system_clock;
/*!
 * 临时保存计数结果至csv文件中
 * @param germ_nums_per_img
 * @return
 */
int TempHeamoWriteResult(const HeamoCnt_t& cnt)
{

    const std::vector<int>& heamo_rbc_nums_v = cnt.heamo_rbc_nums_v;
    const std::vector<int>& heamo_plt_nums_v = cnt.heamo_plt_nums_v;
    const std::vector<int>& heamo_ret_nums_v = cnt.heamo_ret_nums_v;
    const std::vector<int>& heamo_wbc_nums_v = cnt.heamo_wbc_nums_v;
    const std::vector<int>& heamo_neu_nums_v = cnt.heamo_neu_nums_v;

    const std::vector<int>& heamo_lym_nums_v  = cnt.heamo_lym_nums_v;
    const std::vector<int>& heamo_mono_nums_v = cnt.heamo_mono_nums_v;
    const std::vector<int>& heamo_eos_nums_v  = cnt.heamo_eos_nums_v;
    const std::vector<int>& heamo_baso_nums_v = cnt.heamo_baso_nums_v;
    const std::vector<int>& heamo_nrbc_nums_v = cnt.heamo_nrbc_nums_v;

    const std::vector<int>& heamo_pla_nums_v = cnt.heamo_pla_nums_v;
    const std::vector<int>& heamo_pv1_nums_v = cnt.heamo_pv1_nums_v;
    const std::vector<int>& heamo_pv2_nums_v = cnt.heamo_pv2_nums_v;
    const std::vector<int>& heamo_pv3_nums_v = cnt.heamo_pv3_nums_v;
    const std::vector<int>& heamo_pv4_nums_v = cnt.heamo_pv4_nums_v;
    const std::vector<int>& heamo_pv5_nums_v = cnt.heamo_pv5_nums_v;
    const std::vector<int>& heamo_pv6_nums_v = cnt.heamo_pv6_nums_v;

    int max_img_nums = 0;
    FindMaxValue(std::vector<int>{(int)heamo_rbc_nums_v.size(),
                                  (int)heamo_plt_nums_v.size(),
                                  (int)heamo_ret_nums_v.size(),
                                  (int)heamo_wbc_nums_v.size(),
                                  (int)heamo_neu_nums_v.size(),
                                  (int)heamo_lym_nums_v.size(),
                                  (int)heamo_mono_nums_v.size(),
                                  (int)heamo_eos_nums_v.size(),
                                  (int)heamo_baso_nums_v.size(),
                                  (int)heamo_nrbc_nums_v.size(),
                                  (int)heamo_pla_nums_v.size(),
                                  (int)heamo_pv1_nums_v.size(),
                                  (int)heamo_pv2_nums_v.size(),
                                  (int)heamo_pv3_nums_v.size(),
                                  (int)heamo_pv4_nums_v.size(),
                                  (int)heamo_pv5_nums_v.size(),
                                  (int)heamo_pv6_nums_v.size()

                 },
                 max_img_nums);

    if (max_img_nums == 0) {
        ALGLogError << "Heamo csv file will not be created cause zero germ imgs accepted.";
        return 0;
    }

    auto        time_now = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
    std::string csv_save_dir(heamo_save_dir);
    std::string csv_save_path = csv_save_dir.append(std::to_string(time_now)).append(".csv");
    std::vector<std::vector<int>> datas{heamo_rbc_nums_v,
                                        heamo_plt_nums_v,
                                        heamo_ret_nums_v,
                                        heamo_wbc_nums_v,
                                        heamo_neu_nums_v,
                                        heamo_lym_nums_v,
                                        heamo_mono_nums_v,
                                        heamo_eos_nums_v,
                                        heamo_baso_nums_v,
                                        heamo_nrbc_nums_v,
                                        heamo_pla_nums_v,
                                        heamo_pv1_nums_v,
                                        heamo_pv2_nums_v,
                                        heamo_pv3_nums_v,
                                        heamo_pv4_nums_v,
                                        heamo_pv5_nums_v,
                                        heamo_pv6_nums_v

    };
    auto future = std::async(std::launch::async, write_to_file, csv_save_path, max_img_nums, datas);
    if (future.wait_for(std::chrono::seconds(10)) == std::future_status::timeout)
    {
        ALGLogError << "Write timed out, skipping.\n";
        return 0;
    }
    else{
        ALGLogError << "Write completed in time.\n";
        return 0;
    }
}



#define HEAMO_CHANNEL_NUMS 8
    static int GetHeamoDilutionRatio(const std::vector<float>& dilution_param_v,
                                     const bool&               qc,
                                     const int&                rbc_channel_nums,
                                     const int&                wbc_channel_nums,
                                     const int&                baso_channel_nums,
                                     const int&                ret_channel_nums,
                                     float&                    dilution_ratio_rbc,
                                     float&                    dilution_ratio_wbc,
                                     float&                    dilution_ratio_baso,
                                     float&                    dilution_ratio_ret)
{
    if (dilution_param_v.size() != HEAMO_CHANNEL_NUMS) {
        ALGLogError << "heamo channel should be " << HEAMO_CHANNEL_NUMS << " but " << dilution_param_v.size() << " was given";
        std::cout<< "heamo channel should be " << HEAMO_CHANNEL_NUMS << " but " << dilution_param_v.size() << " was given \n";
        return -1;
    }
    if (qc) {
        dilution_ratio_rbc  = dilution_param_v[4];
        dilution_ratio_wbc  = dilution_param_v[5];
        dilution_ratio_baso = dilution_param_v[6];
        dilution_ratio_ret  = dilution_param_v[7];
    }
    else {
        dilution_ratio_rbc  = dilution_param_v[0];
        dilution_ratio_wbc  = dilution_param_v[1];
        dilution_ratio_baso = dilution_param_v[2];
        dilution_ratio_ret  = dilution_param_v[3];
    }

    int ret = 0;
    ret     = OpenedChannelDilutionIdentify(rbc_channel_nums, dilution_ratio_rbc);
    ret |= OpenedChannelDilutionIdentify(wbc_channel_nums, dilution_ratio_wbc);
    ret |= OpenedChannelDilutionIdentify(baso_channel_nums, dilution_ratio_baso);
    ret |= OpenedChannelDilutionIdentify(ret_channel_nums, dilution_ratio_ret);
    if (ret) {
        ALGLogError << "Opened channel configured dilution rate 0";
        ALGLogError << "channel accepted img nums:" << " configured dilution";
        ALGLogError << rbc_channel_nums << " " << dilution_ratio_rbc;
        ALGLogError << wbc_channel_nums << " " << dilution_ratio_wbc;
        ALGLogError << baso_channel_nums << " " << dilution_ratio_baso;
        ALGLogError << ret_channel_nums << " " << dilution_ratio_ret;
        return -4;
    }

    ALGLogInfo << "dilution ratio for rbc wbc baso ret, " << dilution_ratio_rbc << " " << dilution_ratio_wbc << " " << dilution_ratio_baso << " "
               << dilution_ratio_ret;
    return 0;
}

void CalculateHCTValue(const float& RBC_value, const float& MCV_value, float& HCT_value)
{
    // 单位换算
    float hct_unit_conversion = 1;
    HCT_value                 = RBC_value * MCV_value / 10 / hct_unit_conversion;
}

void CalculateMCHValue(const float& RBC_value, const float& HGB_value, float& MCH_value)
{
    MCH_value = (float)(HGB_value / ((float)RBC_value + ADD_DENOMINATOR));
}

void CalculateMCHCValue(const float& HGB_value, const float& HCT_value, float& MCHC_value)
{
    MCHC_value = (float)(HGB_value / (HCT_value + ADD_DENOMINATOR)) * 100;
}

void CalculatePCTValue(const float& PLT_value, const float& MPV_value, float& PCT_value)
{
    float pct_unit_conversion = 10000;
    PCT_value                 = PLT_value * MPV_value / pct_unit_conversion;
}

void RbcRelevantValueRectify(float& RBC_value, float& MCV_value, float& HGB_value, float& HCT_value, float& MCH_value, float& MCHC_value)
{
    if (RBC_value == 0.f) {
        MCH_value  = 0;
        MCHC_value = 0;
    }
    if (HCT_value == 0.f) {
        MCHC_value = 0;
    }
    if (HGB_value < 0.f) {
        HGB_value  = 0;
        MCH_value  = 0;
        MCHC_value = 0;
    }
}

#define HEAMO_PLT_GAT_THR 0.3
void plt_gat_check(const float& gat_nums, const int& view_nums, float& res)
{
    if (view_nums == 0) {
        res = 0;
    }
    else {
        res = gat_nums / (float)view_nums > HEAMO_PLT_GAT_THR;
    }
}

/*!
 * 获取血球结果
 * @param ctx_id
 * @param curve_rbc
 * @param curve_plt
 * @param callback
 * @param userdata
 * @param view_param
 * @return
 */
int Heamo_GetHeamoResult(HeamoCtxID_t              ctx_id,
                         std::vector<float>&       curve_rbc,
                         std::vector<float>&       curve_plt,
                         HeamoResultCallback_f     callback,
                         void*                     userdata,
                         const ResultViewParam_t&  view_param,
                         std::vector<std::string>& alarm_str_v)
{
    ALGLogInfo << "Get human heamo result";
    std::cout << "Get human heamo result \n";
    HeamoCtx_t* ctx = (HeamoCtx_t*)ctx_id;
    if (ctx == NULL || callback == NULL) {
        return -1;
    }
    // 提取视野数
    const int& rbc_channel_nums = view_param.rbc_channel_nums;
    const int& wbc_channel_nums = view_param.wbc_channel_nums;
    const int& hgb_channel_nums = view_param.baso_channel_nums;
    const int& ret_channel_nums = view_param.ret_channel_nums;

    ALGLogInfo << "channel accepted nums " << rbc_channel_nums << " " << wbc_channel_nums << " " << hgb_channel_nums << " " << ret_channel_nums;
    std::cout << "Get human heamo result 1\n";
    // 提取视野体积
    float volume_rbc  = VOLUME_RBC;
    float volume_wbc  = VOLUME_WBC;
    float volume_baso = VOLUME_BASO;
    float volume_ret  = VOLUME_RBC;

    // 提取稀释倍数
    float dilution_ratio_rbc, dilution_ratio_wbc, dilution_ratio_baso, dilution_ratio_ret;
    if (GetHeamoDilutionRatio(HEAMO_CTX_DILUTION(ctx),
                              HEAMO_CTX_QC(ctx),
                              rbc_channel_nums,
                              wbc_channel_nums,
                              hgb_channel_nums,
                              ret_channel_nums,
                              dilution_ratio_rbc,
                              dilution_ratio_wbc,
                              dilution_ratio_baso,
                              dilution_ratio_ret)) {
        std::cout << "Get human heamo result dilution size" << HEAMO_CTX_DILUTION(ctx).size() << "\n";
        return -1;
    }

    std::cout << "Get human heamo result 2\n";
    // 对视野面积进行校准
    ALG_DEPLOY::CALIBRATION::Calibration<float> calib;
    // 272
    int ret = calib.SetPhysicalSizeCalibration(HEAMO_CTX_IMG_H(ctx), HEAMO_CTX_IMG_W(ctx), HEAMO_CTX_IMG_H_UM(ctx));
    if (ret) {
        return -2;
    }
    float calib_volume_rbc, calib_volume_wbc, clib_volume_ret, calib_volume_ret, calib_volume_baso;

    // 对体积进行校准
    calib.GetVolCalibrationResult(volume_rbc, calib_volume_rbc);
    calib.GetVolCalibrationResult(volume_wbc, calib_volume_wbc);
    calib.GetVolCalibrationResult(volume_ret, calib_volume_ret);
    calib.GetVolCalibrationResult(volume_baso, calib_volume_baso);
    ALGLogInfo << "calib view area volume rbc, wbc, ret, baso " << calib_volume_rbc << " " << calib_volume_wbc << " " << calib_volume_ret << " "
               << calib_volume_baso;
    std::cout << "Get human heamo result 3\n";
    HeamoCnt_t* cnt = HEAMO_CTX_CNT(ctx);
    // 对细胞区域面积相关参数进行校准
    calib.GetAreaCalibrationResult(cnt->rbc_volume_v, cnt->rbc_volume_v);
    calib.GetAreaCalibrationResult(cnt->plt_volume_v, cnt->plt_volume_v);

    // 单位换算
    float RBC_value = (float)(cnt->RBC * dilution_ratio_rbc / (rbc_channel_nums * calib_volume_rbc + ADD_DENOMINATOR) / 1e+3);
    float PLT_value = (float)(cnt->PLT * dilution_ratio_rbc / (rbc_channel_nums * calib_volume_rbc + ADD_DENOMINATOR));
    float RET_value = (float)(cnt->RET * dilution_ratio_ret / (rbc_channel_nums * calib_volume_ret + ADD_DENOMINATOR) / 1e+3);
    //白细胞unknown 的比例
    //TODO unknown 报警上报
    float unknown_percentage = float((cnt->UNKNOWN) / (cnt->WBC+ ADD_DENOMINATOR) *100);
    // 疟原虫
    float PV_VALUE  = 0;
    float PV1_VALUE = 0;
    float PV2_VALUE = 0;
    float PV3_VALUE = 0;
    float PV4_VALUE = 0;
    float PV5_VALUE = 0;
    float PV6_VALUE = 0;

    std::cout << "Get human heamo result 4\n";
    if (g_pla_flag) {
        PV_VALUE  = (float)(cnt->PLA * dilution_ratio_rbc  * 1e+3 / (rbc_channel_nums * calib_volume_rbc + ADD_DENOMINATOR));
        PV1_VALUE = (float)(PV1_VALUE * dilution_ratio_rbc * 1e+3 / (rbc_channel_nums * calib_volume_rbc + ADD_DENOMINATOR));
        PV2_VALUE = (float)(PV2_VALUE * dilution_ratio_rbc * 1e+3 / (rbc_channel_nums * calib_volume_rbc + ADD_DENOMINATOR));
        PV3_VALUE = (float)(PV3_VALUE * dilution_ratio_rbc * 1e+3 / (rbc_channel_nums * calib_volume_rbc + ADD_DENOMINATOR));
        PV4_VALUE = (float)(PV4_VALUE * dilution_ratio_rbc * 1e+3 / (rbc_channel_nums * calib_volume_rbc + ADD_DENOMINATOR));
        PV5_VALUE = (float)(cnt->PV5 * dilution_ratio_rbc  * 1e+3 / (rbc_channel_nums * calib_volume_rbc + ADD_DENOMINATOR));
        PV6_VALUE = (float)(cnt->PV6 * dilution_ratio_rbc *  1e+3 / (rbc_channel_nums * calib_volume_rbc + ADD_DENOMINATOR));
    }
    PV_VALUE    = PV1_VALUE + PV2_VALUE + PV3_VALUE + PV4_VALUE + PV5_VALUE + PV6_VALUE;
    if (PV6_VALUE>0) {
        PV5_VALUE = PV5_VALUE + PV6_VALUE;
    }
    //疟原虫比例
    float PV1_PERCENT = (PV1_VALUE / (PV_VALUE + ADD_DENOMINATOR)) * 100;
    float PV2_PERCENT = (PV2_VALUE / (PV_VALUE + ADD_DENOMINATOR)) * 100;
    float PV3_PERCENT = (PV3_VALUE / (PV_VALUE + ADD_DENOMINATOR)) * 100;
    float PV4_PERCENT = (PV4_VALUE / (PV_VALUE + ADD_DENOMINATOR)) * 100;
    float PV5_PERCENT = (PV5_VALUE / (PV_VALUE + ADD_DENOMINATOR)) * 100;
    float PV6_PERCENT = (PV6_VALUE / (PV_VALUE + ADD_DENOMINATOR)) * 100;


    float PLA_R = PV_VALUE * 5000000 / ((RBC_value+ADD_DENOMINATOR) * 1e+3);



    // 体积统计
    float MCV_value    = 0.0;
    float MPV_value    = 1.0;
    float RDW_CV_value = 0.0;
    float RDW_SD_value = 0.0;

    // 对结果进行拟合

    float NRBC_value  = (float)(cnt->NRBC * dilution_ratio_wbc /
                               (wbc_channel_nums * calib_volume_wbc + ADD_DENOMINATOR));   // 有核红数量极少,正常人通常没有,因此此处以细胞个数进行展示

    float NRBC_key_percentage = (float)(cnt->NRBC / ((float)cnt->WBC + ADD_DENOMINATOR) * 100);

    float NEUT_value =
        (float)(cnt->NEU * dilution_ratio_wbc / (wbc_channel_nums * calib_volume_wbc + ADD_DENOMINATOR));
    float LYMPH_value = (float)(cnt->LYM * dilution_ratio_wbc / (wbc_channel_nums * calib_volume_wbc + ADD_DENOMINATOR));
    float MONO_value  = (float)(cnt->MONO * dilution_ratio_wbc / (wbc_channel_nums * calib_volume_wbc + ADD_DENOMINATOR));
    float EOS_value   = (float)(cnt->EOS * dilution_ratio_wbc / (wbc_channel_nums * calib_volume_wbc + ADD_DENOMINATOR));

    float IG_value = (float)(cnt->IG * dilution_ratio_wbc / (dilution_ratio_wbc * calib_volume_wbc + ADD_DENOMINATOR));

    float BASO_value = (float)(cnt->BASO * dilution_ratio_baso / (hgb_channel_nums * calib_volume_baso + ADD_DENOMINATOR));

    float HGB_value = 0.0;
    std::cout << "Get human heamo result 5\n";
    // 非球形化
    /*    int  ret = HEAMO_CTX_NORM_REAGENT_FIT(ctx).Forward(cnt->rbc_volume_v,
       cnt->incline_cell_nums, cnt->incline_pixels, cnt->plt_volume_v, 1,
                                                                                                      cnt->hgb_data, cnt->hgb_coef,
                                                                                                      MCV_value, RDW_CV_value, RDW_SD_value,
       MPV_value, curve_rbc, curve_plt, HGB_value, RBC_value, RET_value, PLT_value, NEUT_value, LYMPH_value, MONO_value, EOS_value, BASO_value
                                                                                                      );
            ALGLogInfo<<"Doing Norm Fitting";*/

    // 球形化
    ALGLogInfo << "Doing Spherical Fitting";
    ret = HEAMO_CTX_SPHE_REAGENT_FIT(ctx).Forward(cnt->rbc_volume_v,
                                                  cnt->incline_cell_nums,
                                                  cnt->incline_pixels,
                                                  cnt->plt_volume_v,
                                                  1,
                                                  cnt->hgb_data,
                                                  cnt->hgb_coef,
                                                  MCV_value,
                                                  RDW_CV_value,
                                                  RDW_SD_value,
                                                  MPV_value,
                                                  curve_rbc,
                                                  curve_plt,
                                                  HGB_value,
                                                  RBC_value,
                                                  RET_value,
                                                  PLT_value,
                                                  NEUT_value,
                                                  LYMPH_value,
                                                  MONO_value,
                                                  EOS_value,
                                                  BASO_value);
    if (ret) {
        ALGLogError << "Failed to get fitting result";
        std::cout << "Failed to get fitting result \n ";
        return -4;
    }


    float PDW_value = HEAMO_CTX_SPHE_REAGENT_FIT(ctx).pdw;
    // 单位换算
    /*  float hct_unit_conversion = 1;
      float HCT_value = RBC_value * MCV_value / 10 / hct_unit_conversion;


      float pct_unit_conversion = 10000;
      float PCT_value = PLT_value * MPV_value / pct_unit_conversion;

      float MCH_value = (float) (HGB_value / ((float) RBC_value +
      ADD_DENOMINATOR)); float MCHC_value = (float) (HGB_value / (HCT_value +
      ADD_DENOMINATOR))*100;

      //校正除0时产生的非正常值
      if (RBC_value == 0.f) {
            MCH_value = 0;
            MCHC_value = 0;
      }
      if (HCT_value == 0.f) {
            MCHC_value = 0;
      }
      if(HGB_value<0.f){
            HGB_value = 0;
            MCH_value = 0;
            MCHC_value = 0;
      }*/
    float HCT_value, PCT_value, MCH_value, MCHC_value;
    CalculateHCTValue(RBC_value, MCV_value, HCT_value);
    CalculateMCHValue(RBC_value, HGB_value, MCH_value);
    CalculateMCHCValue(HGB_value, HCT_value, MCHC_value);
    RbcRelevantValueRectify(RBC_value, MCV_value, HGB_value, HCT_value, MCH_value, MCHC_value);
    CalculatePCTValue(PLT_value, MPV_value, PCT_value);

    float WBC_value;
    // 白细胞四分类结果统计
    CountWbcFour(WBC_value, NEUT_value, EOS_value, MONO_value, LYMPH_value, BASO_value);



    float NEUT_percentage  = (float)(NEUT_value / (WBC_value + ADD_DENOMINATOR) * 100);
    float LYMPH_percentage = (float)(LYMPH_value / (WBC_value + ADD_DENOMINATOR) * 100);
    float MONO_percentage  = (float)(MONO_value / (WBC_value + ADD_DENOMINATOR) * 100);
    float EOS_percentage   = (float)(EOS_value / (WBC_value + ADD_DENOMINATOR) * 100);
    float BASO_percentage  = (float)(BASO_value / (WBC_value + ADD_DENOMINATOR) * 100);

    float RET_percentage = (float)(cnt->RET / ((float)cnt->RBC + ADD_DENOMINATOR) * 100);


    // 质控品无法进行分类
    if (HEAMO_CTX_QC(ctx)) {
        WBC_value = (float)(cnt->WBC * dilution_ratio_wbc / (wbc_channel_nums * calib_volume_wbc + ADD_DENOMINATOR));
        ALGLogInfo << "Use QC";
    }
    //疟原虫
    float PLA_W = PV_VALUE *8000 / ((WBC_value + ADD_DENOMINATOR) * 1e+3);

    ret = MakeHeamoUpLoadParams(HEAMO_CTX_QC(ctx),
                                HEAMO_CTX_TASK_ATT(ctx),
                                callback,
                                userdata,
                                WBC_value,
                                NEUT_percentage,
                                LYMPH_percentage,
                                MONO_percentage,
                                EOS_percentage,
                                BASO_percentage,
                                NEUT_value,
                                LYMPH_value,
                                MONO_value,
                                EOS_value,
                                BASO_value,
                                RBC_value,
                                HGB_value,
                                HCT_value,
                                MCV_value,
                                MCH_value,
                                MCHC_value,
                                RDW_CV_value,
                                RDW_SD_value,
                                NRBC_value,
                                NRBC_key_percentage,
                                PLT_value,
                                MPV_value,
                                PCT_value,
                                RET_value,
                                RET_percentage,
                                PDW_value,
                                PV_VALUE,
                                PV1_VALUE,
                                PV2_VALUE,
                                PV3_VALUE,
                                PV4_VALUE,
                                PV5_VALUE,
                                PV6_VALUE,
                                PLA_R,
                                PLA_W,
                                PV1_PERCENT,
                                PV2_PERCENT,
                                PV3_PERCENT,
                                PV4_PERCENT,
                                PV5_PERCENT,
                                PV6_PERCENT);
    if (ret) {
        std::cout << "Failed to get MakeHeamoUpLoadParams \n ";
        return -3;
    }

    //
    if (TempHeamoWriteResult(*cnt)) {
        std::cout << "Failed to TempHeamoWriteResult(*cnt) \n ";
        return -3;
    }

    // 报警
    HeamoAlarmManager alarm_manager;
    if (alarm_manager.SetCustomAlarmParams(HEAMO_CTX_ALARM_PARAM(ctx))) {
        return -4;
    }
    float PLT_GAT_value;
    plt_gat_check(cnt->PLT_GAT, rbc_channel_nums, PLT_GAT_value);

    if (alarm_manager.GetAlarmResults(WBC_value,
                                      NEUT_value,
                                      LYMPH_value,
                                      MONO_value,
                                      EOS_value,
                                      BASO_value,
                                      NRBC_value,
                                      IG_value,
                                      0,   // wbc
                                      curve_rbc,
                                      RDW_CV_value,
                                      RDW_SD_value,
                                      MCV_value,
                                      RBC_value,
                                      HGB_value,
                                      MCHC_value,
                                      0,   // rbc
                                      PLT_value,
                                      curve_plt,
                                      PLT_GAT_value,    // plt
                                      RET_percentage,   // ret
                                      PDW_value,
                                      unknown_percentage,
                                      cnt->WBC,
                                      alarm_str_v)) {

        std::cout << "Failed to alarm_manager.GetAlarmResults \n ";
        return -5;
    }

    ALGLogInfo << "Plt gat num " << cnt->PLT_GAT;
    return 0;
}


std::vector<ResultTypeMap_t> heamo_result_map = {
    MODIFY_TYPE_MAP_DEF(ModifyTypeMap_t, WBC_KEY_NUM, "10^9/L", HEAMO_RESULT_QC_KEEP, 0, HEAMO_RESULT_REQUIRED_ALL, 2, HEAMO_MODIFY_TYPE_WBC),
    MODIFY_TYPE_MAP_DEF(ModifyTypeMap_t, NE_KEY_PERCENTAGE, "%", HEAMO_RESULT_QC_ERASE, 0, HEAMO_RESULT_REQUIRED_ALL, 2, HEAMO_MODIFY_TYPE_WBC),
    MODIFY_TYPE_MAP_DEF(ModifyTypeMap_t, LY_KEY_PERCENTAGE, "%", HEAMO_RESULT_QC_ERASE, 0, HEAMO_RESULT_REQUIRED_ALL, 2, HEAMO_MODIFY_TYPE_WBC),
    MODIFY_TYPE_MAP_DEF(ModifyTypeMap_t, MO_KEY_PERCENTAGE, "%", HEAMO_RESULT_QC_ERASE, 0, HEAMO_RESULT_REQUIRED_ALL, 2, HEAMO_MODIFY_TYPE_WBC),
    MODIFY_TYPE_MAP_DEF(ModifyTypeMap_t, EO_KEY_PERCENTAGE, "%", HEAMO_RESULT_QC_ERASE, 0, HEAMO_RESULT_REQUIRED_ALL, 2, HEAMO_MODIFY_TYPE_WBC),
    MODIFY_TYPE_MAP_DEF(ModifyTypeMap_t, BA_KEY_PERCENTAGE, "%", HEAMO_RESULT_QC_ERASE, 0, HEAMO_RESULT_REQUIRED_ALL, 2, HEAMO_MODIFY_TYPE_WBC),
    MODIFY_TYPE_MAP_DEF(ModifyTypeMap_t, NE_KEY_NUM, "%", HEAMO_RESULT_QC_ERASE, 0, HEAMO_RESULT_REQUIRED_ALL, 2, HEAMO_MODIFY_TYPE_WBC),
    MODIFY_TYPE_MAP_DEF(ModifyTypeMap_t, LY_KEY_NUM, "%", HEAMO_RESULT_QC_ERASE, 0, HEAMO_RESULT_REQUIRED_ALL, 2, HEAMO_MODIFY_TYPE_WBC),
    MODIFY_TYPE_MAP_DEF(ModifyTypeMap_t, MO_KEY_NUM, "%", HEAMO_RESULT_QC_ERASE, 0, HEAMO_RESULT_REQUIRED_ALL, 2, HEAMO_MODIFY_TYPE_WBC),
    MODIFY_TYPE_MAP_DEF(ModifyTypeMap_t, EO_KEY_NUM, "%", HEAMO_RESULT_QC_ERASE, 0, HEAMO_RESULT_REQUIRED_ALL, 2, HEAMO_MODIFY_TYPE_WBC),
    MODIFY_TYPE_MAP_DEF(ModifyTypeMap_t, BA_KEY_NUM, "10^9/L", HEAMO_RESULT_QC_ERASE, 0, HEAMO_RESULT_REQUIRED_ALL, 2, HEAMO_MODIFY_TYPE_WBC),

    MODIFY_TYPE_MAP_DEF(ModifyTypeMap_t, RBC_KEY_NUM, "10^9/L", HEAMO_RESULT_QC_KEEP, 0, HEAMO_RESULT_REQUIRED_ALL, 2, HEAMO_MODIFY_TYPE_RBC),
    MODIFY_TYPE_MAP_DEF(ModifyTypeMap_t, HB_KEY_VALUE, "g/L", HEAMO_RESULT_QC_KEEP, 0, HEAMO_RESULT_REQUIRED_ALL, 2, HEAMO_MODIFY_TYPE_RBC),
    MODIFY_TYPE_MAP_DEF(ModifyTypeMap_t, HCT_KEY_VALUE, "%", HEAMO_RESULT_QC_KEEP, 0, HEAMO_RESULT_REQUIRED_ALL, 2, HEAMO_MODIFY_TYPE_NONE),
    MODIFY_TYPE_MAP_DEF(ModifyTypeMap_t, MCV_KEY_VALUE, "fL", HEAMO_RESULT_QC_KEEP, 0, HEAMO_RESULT_REQUIRED_ALL, 2, HEAMO_MODIFY_TYPE_RBC),
    MODIFY_TYPE_MAP_DEF(ModifyTypeMap_t, MCH_KEY_VALUE, "pg", HEAMO_RESULT_QC_ERASE, 0, HEAMO_RESULT_REQUIRED_ALL, 2, HEAMO_MODIFY_TYPE_NONE),
    MODIFY_TYPE_MAP_DEF(ModifyTypeMap_t, MCHC_KEY_VALUE, "g/L", HEAMO_RESULT_QC_ERASE, 0, HEAMO_RESULT_REQUIRED_ALL, 2, HEAMO_MODIFY_TYPE_NONE),
    MODIFY_TYPE_MAP_DEF(ModifyTypeMap_t, RDW_CV_KEY_VALUE, "%", HEAMO_RESULT_QC_ERASE, 0, HEAMO_RESULT_REQUIRED_ALL, 2, HEAMO_MODIFY_TYPE_NONE),
    MODIFY_TYPE_MAP_DEF(ModifyTypeMap_t, RDW_SD_KEY_VALUE, "fL", HEAMO_RESULT_QC_ERASE, 0, HEAMO_RESULT_REQUIRED_ALL, 2, HEAMO_MODIFY_TYPE_NONE),
    MODIFY_TYPE_MAP_DEF(ModifyTypeMap_t, NRBC_KEY_NUM, "10^9/L", HEAMO_RESULT_QC_ERASE, 0, HEAMO_RESULT_REQUIRED_ALL, 3, HEAMO_MODIFY_TYPE_NONE),
    MODIFY_TYPE_MAP_DEF(ModifyTypeMap_t, NRBC_KEY_PERCENTAGE, "%", HEAMO_RESULT_QC_ERASE, 0, HEAMO_RESULT_REQUIRED_ALL, 3, HEAMO_MODIFY_TYPE_NONE),
    MODIFY_TYPE_MAP_DEF(ModifyTypeMap_t, PLT_KEY_NUM, "10^9/L", HEAMO_RESULT_QC_KEEP, 0, HEAMO_RESULT_REQUIRED_ALL, 2, HEAMO_MODIFY_TYPE_PLT),
    MODIFY_TYPE_MAP_DEF(ModifyTypeMap_t, MPV_KEY_VALUE, "fL", HEAMO_RESULT_QC_ERASE, 0, HEAMO_RESULT_REQUIRED_ALL, 2, HEAMO_MODIFY_TYPE_PLT),
    MODIFY_TYPE_MAP_DEF(ModifyTypeMap_t, PCT_KEY_VALUE, "%", HEAMO_RESULT_QC_ERASE, 0, HEAMO_RESULT_REQUIRED_ALL, 2, HEAMO_MODIFY_TYPE_NONE),
    MODIFY_TYPE_MAP_DEF(ModifyTypeMap_t, RET_KEY_NUM, "10^12/L", HEAMO_RESULT_QC_ERASE, 0, HEAMO_RESULT_REQUIRED_RET, 2, HEAMO_MODIFY_TYPE_RBC),
    MODIFY_TYPE_MAP_DEF(ModifyTypeMap_t, RET_KEY_PERCENTAGE, "%", HEAMO_RESULT_QC_ERASE, 0, HEAMO_RESULT_REQUIRED_RET, 2, HEAMO_MODIFY_TYPE_NONE),
    MODIFY_TYPE_MAP_DEF(ModifyTypeMap_t, "PDW", "fL", HEAMO_RESULT_QC_KEEP, 0, HEAMO_RESULT_REQUIRED_ALL, 2, HEAMO_MODIFY_TYPE_PLT),
    MODIFY_TYPE_MAP_DEF(ModifyTypeMap_t, "PLS", "/uL", HEAMO_RESULT_QC_KEEP, 0, HEAMO_RESULT_REQUIRED_ALL, 2, HEAMO_MODIFY_TYPE_RBC),
    MODIFY_TYPE_MAP_DEF(ModifyTypeMap_t, "Pr#", "/uL", HEAMO_RESULT_QC_KEEP, 0, HEAMO_RESULT_REQUIRED_ALL, 2, HEAMO_MODIFY_TYPE_RBC),
    MODIFY_TYPE_MAP_DEF(ModifyTypeMap_t, "Pt#", "/uL", HEAMO_RESULT_QC_KEEP, 0, HEAMO_RESULT_REQUIRED_ALL, 2, HEAMO_MODIFY_TYPE_RBC),
    MODIFY_TYPE_MAP_DEF(ModifyTypeMap_t, "Ps#", "/uL", HEAMO_RESULT_QC_KEEP, 0, HEAMO_RESULT_REQUIRED_ALL, 2, HEAMO_MODIFY_TYPE_RBC),
    MODIFY_TYPE_MAP_DEF(ModifyTypeMap_t, "Pg#", "/uL", HEAMO_RESULT_QC_KEEP, 0, HEAMO_RESULT_REQUIRED_ALL, 2, HEAMO_MODIFY_TYPE_RBC),
    MODIFY_TYPE_MAP_DEF(ModifyTypeMap_t, "Px#", "/uL", HEAMO_RESULT_QC_KEEP, 0, HEAMO_RESULT_REQUIRED_ALL, 2, HEAMO_MODIFY_TYPE_RBC),
    MODIFY_TYPE_MAP_DEF(ModifyTypeMap_t, "E-rosetting#", "/uL", HEAMO_RESULT_QC_KEEP, 0, HEAMO_RESULT_REQUIRED_ALL, 2, HEAMO_MODIFY_TYPE_RBC),
    MODIFY_TYPE_MAP_DEF(ModifyTypeMap_t, "Pr%", "/uL", HEAMO_RESULT_QC_KEEP, 0, HEAMO_RESULT_REQUIRED_ALL, 2, HEAMO_MODIFY_TYPE_RBC),
    MODIFY_TYPE_MAP_DEF(ModifyTypeMap_t, "Pt%", "/uL", HEAMO_RESULT_QC_KEEP, 0, HEAMO_RESULT_REQUIRED_ALL, 2, HEAMO_MODIFY_TYPE_RBC),
    MODIFY_TYPE_MAP_DEF(ModifyTypeMap_t, "Ps%", "/uL", HEAMO_RESULT_QC_KEEP, 0, HEAMO_RESULT_REQUIRED_ALL, 2, HEAMO_MODIFY_TYPE_RBC),
    MODIFY_TYPE_MAP_DEF(ModifyTypeMap_t, "Pg%", "/uL", HEAMO_RESULT_QC_KEEP, 0, HEAMO_RESULT_REQUIRED_ALL, 2, HEAMO_MODIFY_TYPE_RBC),
    MODIFY_TYPE_MAP_DEF(ModifyTypeMap_t, "Px%", "/uL", HEAMO_RESULT_QC_KEEP, 0, HEAMO_RESULT_REQUIRED_ALL, 2, HEAMO_MODIFY_TYPE_RBC),
    MODIFY_TYPE_MAP_DEF(ModifyTypeMap_t, "E-rosetting%", "/uL", HEAMO_RESULT_QC_KEEP, 0, HEAMO_RESULT_REQUIRED_ALL, 2, HEAMO_MODIFY_TYPE_RBC),
    MODIFY_TYPE_MAP_DEF(ModifyTypeMap_t, "PLS/5m RBC", "/uL", HEAMO_RESULT_QC_KEEP, 0, HEAMO_RESULT_REQUIRED_ALL, 2, HEAMO_MODIFY_TYPE_RBC),
    MODIFY_TYPE_MAP_DEF(ModifyTypeMap_t, "PLS/8k WBC", "/uL", HEAMO_RESULT_QC_KEEP, 0, HEAMO_RESULT_REQUIRED_ALL, 2, HEAMO_MODIFY_TYPE_RBC),
};

/*!
 * 将当前传入的一个res解析至map中
 * @param accept_res
 * @param result_map
 * @return
 */
int SetAcceptedItemToResultMap(const AlgCellItem_t& accept_res, std::vector<ResultTypeMap_t>& result_map)
{
    for (auto& one_res : result_map) {
        if (one_res.key == accept_res.name) {
            std::string value = accept_res.value;
            // 对于特殊符号"-"将其转换为0,之后根据map的配置重新组装
            if (value == HEAMO_QC_ERASE_VALUE_COMIC) {
                one_res.value = 0;
                return 0;
            }
            else {
                std::stringstream ss(value);
                float             value_f;
                ss >> value_f;
                if (ss.fail()) {
                    ALGLogError << "Failed to change value type with key, value " << accept_res.name << " " << accept_res.value;
                    return -1;
                }
                else {
                    one_res.value = value_f;
                    return 0;
                }
            }
        }
    }
    ALGLogError << "Do not find key " << accept_res.name;
    return -2;
}

/*!
 * 查找更改的值对应的key, 并将前端下发的值写入结构体中
 * @param changed_param_key
 * @param result_map
 * @param list
 * @param modify_type
 * @return
 */
int ParseModifiedResult(const std::string&              changed_param_key,
                        std::vector<ResultTypeMap_t>&   result_map,
                        const std::list<AlgCellItem_t>& list,
                        HeamoModifyType&                modify_type)
{
    // 搜索更改值的类型
    modify_type = HEAMO_MODIFY_TYPE_NONE;
    for (const auto& one_type : result_map) {
        if (changed_param_key == one_type.key) {
            modify_type = one_type.type;
            break;
        }
    }

    // 解析每一个值
    int ret;
    for (const auto& item : list) {
        ret = SetAcceptedItemToResultMap(item, result_map);
        if (ret) {
            return ret;
        }
    }
    return 0;
}

// 清除全局结果map
void Heamo_ClearResult()
{
    for (auto& one_res : heamo_result_map) {
        one_res.value = 0;
    }
}

int FindValueInResultMap(const std::vector<ResultTypeMap_t>& result_map, const std::string& key, float& value)
{
    for (const auto& one_res : result_map) {
        if (one_res.key == key) {
            value = one_res.value;
            return 0;
        }
    }
    ALGLogError << "Failed to find key " << key << " in result map";
    return -1;
}

int SetValueToResultMap(const std::string& key, const float& value, std::vector<ResultTypeMap_t>& result_map)
{
    for (auto& one_res : result_map) {
        if (one_res.key == key) {
            one_res.value = value;
            return 0;
        }
    }
    ALGLogError << "Failed to set key " << key << " to result map";
    return -1;
}

void CalculateValueAccordPercentage(const float& total_num, const float& percentage, float& value)
{
    value = total_num * percentage * 0.01;
}

/*!
 * 改变RBC或HGB的值
 * @param result_map
 * @return
 */
int ChangeRBCOrHGBOrMCVOrRet(std::vector<ResultTypeMap_t>& result_map)
{
    float RBC_value, MCV_value, HGB_value, HCT_value, MCH_value, MCHC_value, RET_percentage, RET_value;
    int   ret = 0;
    ret |= FindValueInResultMap(result_map, RBC_KEY_NUM, RBC_value);
    ret |= FindValueInResultMap(result_map, MCV_KEY_VALUE, MCV_value);
    ret |= FindValueInResultMap(result_map, HB_KEY_VALUE, HGB_value);
    ret |= FindValueInResultMap(result_map, HCT_KEY_VALUE, HCT_value);
    ret |= FindValueInResultMap(result_map, MCH_KEY_VALUE, MCH_value);
    ret |= FindValueInResultMap(result_map, MCHC_KEY_VALUE, MCHC_value);
    ret |= FindValueInResultMap(result_map, RET_KEY_PERCENTAGE, RET_percentage);
    ret |= FindValueInResultMap(result_map, RET_KEY_NUM, RET_value);
    if (ret) {
        return ret;
    }
    CalculateHCTValue(RBC_value, MCV_value, HCT_value);
    CalculateMCHValue(RBC_value, HGB_value, MCH_value);
    CalculateMCHCValue(HGB_value, HCT_value, MCHC_value);
    RbcRelevantValueRectify(RBC_value, MCV_value, HGB_value, HCT_value, MCH_value, MCHC_value);

    CalculateValueAccordPercentage(RBC_value * 1000, RET_percentage, RET_value);

    ret |= SetValueToResultMap(HCT_KEY_VALUE, HCT_value, result_map);
    ret |= SetValueToResultMap(MCH_KEY_VALUE, MCH_value, result_map);
    ret |= SetValueToResultMap(MCHC_KEY_VALUE, MCHC_value, result_map);
    ret |= SetValueToResultMap(RET_KEY_NUM, RET_value, result_map);
    if (ret) {
        return ret;
    }

    return 0;
}

/*!
 * 改变任一RBC的值
 * @param result_map
 * @return
 */
int ChangeWBC(std::vector<ResultTypeMap_t>& result_map)
{
    float WBC_value, NEUT_percentage, LYMPH_percentage, MONO_percentage, EOS_percentage, BASO_percentage, NEUT_value, LYMPH_value, MONO_value,
        EOS_value, BASO_value;

    int ret = 0;
    ret |= FindValueInResultMap(result_map, WBC_KEY_NUM, WBC_value);
    ret |= FindValueInResultMap(result_map, NE_KEY_PERCENTAGE, NEUT_percentage);
    ret |= FindValueInResultMap(result_map, LY_KEY_PERCENTAGE, LYMPH_percentage);
    ret |= FindValueInResultMap(result_map, MO_KEY_PERCENTAGE, MONO_percentage);
    ret |= FindValueInResultMap(result_map, EO_KEY_PERCENTAGE, EOS_percentage);
    ret |= FindValueInResultMap(result_map, BA_KEY_PERCENTAGE, BASO_percentage);

    ret |= FindValueInResultMap(result_map, NE_KEY_NUM, NEUT_value);
    ret |= FindValueInResultMap(result_map, LY_KEY_NUM, LYMPH_value);
    ret |= FindValueInResultMap(result_map, MO_KEY_NUM, MONO_value);
    ret |= FindValueInResultMap(result_map, EO_KEY_NUM, EOS_value);
    ret |= FindValueInResultMap(result_map, BA_KEY_NUM, BASO_value);
    if (ret) {
        return ret;
    }

    CalculateValueAccordPercentage(WBC_value, NEUT_percentage, NEUT_value);
    CalculateValueAccordPercentage(WBC_value, LYMPH_percentage, LYMPH_value);
    CalculateValueAccordPercentage(WBC_value, MONO_percentage, MONO_value);
    CalculateValueAccordPercentage(WBC_value, EOS_percentage, EOS_value);
    CalculateValueAccordPercentage(WBC_value, BASO_percentage, BASO_value);

    ret |= SetValueToResultMap(NE_KEY_NUM, NEUT_value, result_map);
    ret |= SetValueToResultMap(LY_KEY_NUM, LYMPH_value, result_map);
    ret |= SetValueToResultMap(MO_KEY_NUM, MONO_value, result_map);
    ret |= SetValueToResultMap(EO_KEY_NUM, EOS_value, result_map);
    ret |= SetValueToResultMap(BA_KEY_NUM, BASO_value, result_map);
    if (ret) {
        return ret;
    }

    return 0;
}

int ChangePLTOrMPV(std::vector<ResultTypeMap_t>& result_map)
{
    float PLT_value, MPV_value, PCT_value;
    int   ret = 0;
    ret |= FindValueInResultMap(result_map, PLT_KEY_NUM, PLT_value);
    ret |= FindValueInResultMap(result_map, MPV_KEY_VALUE, MPV_value);
    ret |= FindValueInResultMap(result_map, PCT_KEY_VALUE, PCT_value);
    if (ret) {
        return ret;
    }
    CalculatePCTValue(PLT_value, MPV_value, PCT_value);
    ret |= SetValueToResultMap(PCT_KEY_VALUE, PCT_value, result_map);
    if (ret) {
        return ret;
    }
    return 0;
}

int MakeHeamoUpLoadParams(const std::vector<ResultTypeMap_t>& result_map,
                          const bool&                         qc,
                          const std::vector<float>&           task_att_v,
                          HeamoResultCallback_f               callback,
                          void*                               userdata)
{
    if (task_att_v.size() != TASK_ATT_REQUIRED_SIZE) {
        ALGLogError << "Task att must be " << TASK_ATT_REQUIRED_SIZE << " "
                    << "but " << task_att_v.size() << "was given ";
        return -1;
    }
    const bool required_ret  = static_cast<bool>(task_att_v[0]);
    int        required_type = HEAMO_RESULT_REQUIRED_EMPTY;
    if (required_ret) {
        required_type |= HEAMO_RESULT_REQUIRED_RET;
    }
    for (const auto& one_res : result_map) {
        if (one_res.required_mask & required_type) {
            std::string key  = one_res.key;
            std::string unit = one_res.unit;
            if (qc && one_res.qc_value_type == HEAMO_RESULT_QC_ERASE) {
                unit = HEAMO_QC_ERASE_VALUE_COMIC;
            }
            std::string value_s;
            CutFloatToString(one_res.value, one_res.precision_num, value_s);
            HEAMO_RESULT(callback, userdata, key.c_str(), unit.c_str(), value_s.c_str())
        }
    }

    return 0;
}

void FindQcInfo(const std::vector<ResultTypeMap_t>& result_map, bool& qc)
{
    for (const auto& one_res : result_map) {
        if (one_res.unit == HEAMO_QC_ERASE_VALUE_COMIC) {
            qc = true;
            return;
        }
    }
}

void FindRetInfo(const std::vector<ResultTypeMap_t>& result_map, std::vector<float>& task_att_v)
{
    task_att_v = {0, 0};
    for (const auto& one_res : result_map) {
        if (one_res.key == RET_KEY_PERCENTAGE) {
            task_att_v = {1, 1};
            return;
        }
    }
}

/*!
 * 获取血球结果
 * @param ctx_id
 * @param curve_rbc
 * @param curve_plt
 * @param callback
 * @param userdata
 * @param view_param
 * @return
 */
int Heamo_ModifyHumanResult(const std::string&        changed_param_key,
                            std::vector<float>&       curve_rbc,
                            std::vector<float>&       curve_plt,
                            HeamoResultCallback_f     callback,
                            void*                     userdata,
                            std::vector<std::string>& alarm_str_v)
{
    ALGLogInfo << "key " << changed_param_key;
    Heamo_ClearResult();
    HeamoModifyType modify_type = HEAMO_MODIFY_TYPE_NONE;

    std::list<AlgCellItem_t>* list = (std::list<AlgCellItem_t>*)userdata;
    int                       ret  = ParseModifiedResult(changed_param_key, heamo_result_map, *list, modify_type);
    if (ret) {
        return ret;
    }

    switch (modify_type) {
    case HEAMO_MODIFY_TYPE_WBC:
    {
        ret = ChangeWBC(heamo_result_map);
    }
    case HEAMO_MODIFY_TYPE_RBC:
    {
        ret = ChangeRBCOrHGBOrMCVOrRet(heamo_result_map);
    }
    case HEAMO_MODIFY_TYPE_PLT:
    {
        ret = ChangePLTOrMPV(heamo_result_map);
    }
    default:
        ret = 0;
    }
    if (ret) {
        return ret;
    }
    bool qc;
    FindQcInfo(heamo_result_map, qc);

    std::vector<float> task_att_v;
    FindRetInfo(heamo_result_map, task_att_v);

    list->clear();

    ret = MakeHeamoUpLoadParams(heamo_result_map, qc, task_att_v, callback, userdata);
    if (ret) {
        return ret;
    }
    HeamoAlarmManager alarm_manager;
    float WBC_value, NEUT_value, LYMPH_value, MONO_value, EOS_value, BASO_value, NRBC_value, IG_value, wbc_agglomeration, RDW_CV_value, RDW_SD_value,
        MCV_value, RBC_value, HGB_value, MCHC_value, rbc_agglomeration, PLT_value, plt_agglomeration, RET_percentage, PDW_value, unknown_percentage;

    ret = 0;
    ret |= FindValueInResultMap(heamo_result_map, WBC_KEY_NUM, WBC_value);

    ret |= FindValueInResultMap(heamo_result_map, NE_KEY_NUM, NEUT_value);
    ret |= FindValueInResultMap(heamo_result_map, LY_KEY_NUM, LYMPH_value);
    ret |= FindValueInResultMap(heamo_result_map, MO_KEY_NUM, MONO_value);
    ret |= FindValueInResultMap(heamo_result_map, EO_KEY_NUM, EOS_value);
    ret |= FindValueInResultMap(heamo_result_map, BA_KEY_NUM, BASO_value);

    ret |= FindValueInResultMap(heamo_result_map, RBC_KEY_NUM, RBC_value);
    ret |= FindValueInResultMap(heamo_result_map, MCV_KEY_VALUE, MCV_value);
    ret |= FindValueInResultMap(heamo_result_map, HB_KEY_VALUE, HGB_value);
    ret |= FindValueInResultMap(heamo_result_map, MCHC_KEY_VALUE, MCHC_value);
    ret |= FindValueInResultMap(heamo_result_map, PLT_KEY_NUM, PLT_value);

    ret |= FindValueInResultMap(heamo_result_map, NRBC_KEY_NUM, NRBC_value);
    ret |= FindValueInResultMap(heamo_result_map, RDW_CV_KEY_VALUE, RDW_CV_value);
    ret |= FindValueInResultMap(heamo_result_map, RDW_SD_KEY_VALUE, RDW_SD_value);

    ret |= FindValueInResultMap(heamo_result_map, RET_KEY_PERCENTAGE, RET_percentage);
    ret |= FindValueInResultMap(heamo_result_map, "PDW", PDW_value);

    // TODO  修改的时候不需要 不需要 wbc_unknown_percentage
    unknown_percentage=0.0;
    if (ret)
    {
        return ret;
    }
    // 这4个值未上传给前端,需要特殊处理
    IG_value          = 0;
    wbc_agglomeration = 0;
    rbc_agglomeration = 0;
    plt_agglomeration = 0;

    ret = alarm_manager.ModifyAlarmResults(std::vector<std::string>(alarm_str_v),
                                           WBC_value,
                                           NEUT_value,
                                           LYMPH_value,
                                           MONO_value,
                                           EOS_value,
                                           BASO_value,
                                           NRBC_value,
                                           IG_value,
                                           wbc_agglomeration,
                                           curve_rbc,
                                           RDW_CV_value,
                                           RDW_SD_value,
                                           MCV_value,
                                           RBC_value,
                                           HGB_value,
                                           MCHC_value,
                                           rbc_agglomeration,
                                           PLT_value,
                                           curve_plt,
                                           plt_agglomeration,
                                           RET_percentage,
                                           PDW_value,
                                           unknown_percentage,
                                           300,
                                           alarm_str_v);
    if (ret) {
        return ret;
    }

    return 0;
}

int FindRbcFusionRate(const bool& fused, const int& group_idx, float& fusion_rate)
{
    if (!fused) {
        fusion_rate = 1.0;
        return 0;
    }
    std::vector<AiChlReg_t> list;
    int                     ret = Ai_GetChlReglist(list, Heamo_FindGroup(group_idx));
    if (ret != 0) {
        ALGLogError << "Failed to get chl list";
        return -1;
    }
    for (int chl_idx = 0; chl_idx < list.size(); ++chl_idx) {
        // 像元融合比例在单个通道内相同,因此此处仅取第一个视图
        if (list[chl_idx].chl_type == AI_CHL_TYPE_RBC) {
            for (const auto view : *(list[chl_idx].view_list)) {
                fusion_rate = view.fusion_rate;
                return 0;
            }
        }
    }
    return 0;
}

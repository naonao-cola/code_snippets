/**
 * @FilePath     : /DIH-ALG/libalg/libalgimm.h
 * @Description  :
 * @Author       : naonao
 * @Date         : 2025-06-06 09:29:14
 * @Version      : 0.0.1
 * @LastEditors  : naonao
 * @LastEditTime : 2025-06-26 11:36:51
 * @Copyright (c) 2025 by G, All Rights Reserved.
 **/
#ifndef _LIBIMMUNO_H_
#define _LIBIMMUNO_H_

#include <list>
#include <vector>

#include "immune.h"

#define AlgImmCtxID_t void*
#define AlgImmFuncMask_t uint32_t
#define AlgImmData_t float
#define ALGIMM_LIB_VERSION_LENGTH 30

typedef AlgResultOut AlgImmRst_t;

typedef enum AlgImmFunc
{
    ALGIMM_FUNC_CALIB = (1 << 0),
} AlgImmFunc_e;

std::string AlgImm_Version(char* alg_version, char* qr_json_version, char* lua_version, char* main_versioon);

AlgImmCtxID_t AlgImm_Init(void);

int AlgImm_DeInit(AlgImmCtxID_t ctx_id);
int AlgImm_RunConfigLoad(AlgImmCtxID_t ctx_id, const char* cfg_path);   // 读取初始化文件失败
int AlgImm_RunConfigUnload(AlgImmCtxID_t ctx_id);

int AlgImm_GetCardInfo(AlgImmCtxID_t ctx_id, uint32_t group, uint32_t mask, char* buf, uint32_t size, char* card_info);

int AlgImm_Open(AlgImmCtxID_t ctx_id, AlgImmFuncMask_t func_mask, const std::string& cardinfo, const float calib_coef);
int AlgImm_PushData(AlgImmCtxID_t ctx_id, std::vector<AlgImmData_t>& data, uint32_t group_idx, uint32_t samp_idx);
int AlgImm_GetResult(AlgImmCtxID_t ctx_id, AlgImmRst_t& result);
int AlgImm_Close(AlgImmCtxID_t ctx_id);

#endif /* _ALG_IMMUNO_H_ */

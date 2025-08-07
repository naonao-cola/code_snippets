
#ifndef _ALG_CLARITY_H_
#define _ALG_CLARITY_H_

#include <list>
#include <vector>

#include "ai.h"

#define ClarityCtxID_t void*
#define ClarityValue_t float
#define ClarityImg_t AiImg_t

int Clarity_GetGroupReglist(std::vector<AiGroupReg_t>& list);
int Clarity_GetModReglist(std::vector<AiModReg_t>& list, uint32_t group_idx);
int Clarity_GetChlReglist(std::vector<AiChlReg_t>& list, uint32_t group_idx);
int Clarity_GetViewReglist(std::vector<AiViewReg_t>& list, uint32_t group_idx, uint32_t chl_idx);

ClarityCtxID_t Clarity_Init(AiCtxID_t ai_ctxid);
int            Clarity_DeInit(ClarityCtxID_t ctx_id);
int            Clarity_Open(ClarityCtxID_t ctx_id, const bool& img_fusion, const bool& debug, AiImgCallback_f callback, void* userdata);
int            Clarity_AddImgList(ClarityCtxID_t           ctx_id,
                                  std::list<ClarityImg_t>& list,
                                  uint32_t                 group_idx,
                                  uint32_t                 chl_idx,
                                  uint32_t                 view_idx,
                                  uint8_t*                 img_data,
                                  uint32_t                 width,
                                  uint32_t                 height);
int Clarity_PushImage(ClarityCtxID_t ctx_id, uint32_t group_idx, uint32_t chl_idx, std::list<ClarityImg_t>& img_list, const int& view_pair_idx);
int Clarity_WaitCplt(ClarityCtxID_t ctx_id, uint32_t timeout);
int Clarity_GetValue(ClarityCtxID_t ctx_id, uint32_t view_idx, ClarityValue_t* value);
int Clarity_GetAllValue(ClarityCtxID_t ctx_id, std::vector<ClarityValue_t>& list);
int Clarity_GetLastValue(ClarityCtxID_t ctx_id, uint32_t* index, ClarityValue_t* value);
int            Clarity_GetBestValue(ClarityCtxID_t ctx_id, uint32_t* index, ClarityValue_t* value, int& type);
int Clarity_Close(ClarityCtxID_t ctx_id);


int Clarity_DynamicGenMod(const char* cfg_path, std::string& model_info);

#endif /* _ALG_CLARITY_H_ */

#ifndef _LIBIMMUNO_H_
#define _LIBIMMUNO_H_

#include <list>
#include <vector>

#include "immune.h"

#define AlgImmCtxID_t			void*
#define AlgImmFuncMask_t		uint32_t
#define AlgImmData_t			float


typedef AlgResultOut AlgImmRst_t;

typedef enum AlgImmFunc
{
    ALGIMM_FUNC_CALIB = (1 << 0),
}AlgImmFunc_e;

std::string AlgImm_Version();
AlgImmCtxID_t AlgImm_Init(void);
int AlgImm_DeInit(AlgImmCtxID_t ctx_id);
int AlgImm_RunConfigLoad(AlgImmCtxID_t ctx_id, const char *cfg_path);
int AlgImm_RunConfigUnload(AlgImmCtxID_t ctx_id);
int AlgImm_Open(AlgImmCtxID_t ctx_id, AlgImmFuncMask_t func_mask, const std::string &cardinfo, const float calib_coef);
int AlgImm_PushData(AlgImmCtxID_t ctx_id, std::vector<AlgImmData_t> &data, uint32_t group_idx, uint32_t samp_idx);
int AlgImm_GetResult(AlgImmCtxID_t ctx_id, AlgImmRst_t &result);
int AlgImm_Close(AlgImmCtxID_t ctx_id);

#endif /* _ALG_IMMUNO_H_ */

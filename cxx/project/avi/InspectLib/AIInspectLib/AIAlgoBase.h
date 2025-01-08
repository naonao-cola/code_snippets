#pragma once

#include <thread>
#include <condition_variable>
#include <mutex>
#include <windows.h>

#include "AIRuntime/AIRuntimeDataStruct.h"
#include "AIRuntime/MPMCQueue.h"
#include "AIRuntime/AIRuntimeInterface.h"
#include "AIInpectAlgoInterface.h"
#include "AIAlgoDefines.h"



using InspParamPtr = std::shared_ptr<void>;
using AlgoResultPtr = std::shared_ptr<void>;

enum ENUM_PARA_AI_PARAM {
    E_PARA_AI_MODELID       =  0,       //  Model id
    E_PARA_AI_MAXBATCHSIZE      ,       //  The max batch size during inference
    E_PARA_AI_CONFIDENCETHR     ,       //  Objects confidence threshold
    E_PARA_AI_NMSTHR            ,       //  NMS threshold
    E_PARA_AI_MAXOBJECTNums     ,       //  max object numbers
    E_PARA_AI_USEPINMEM         ,       //  Indicates whether to use locked pages memorys
    E_PARA_AI_WORKSPACESIZE     ,       //  The workspaec size of tensorrt
    E_PARA_AI_GPUCACHESIZE      ,       //  The GPU cache size
    E_PARA_AI_HOSTCACHESIZE     ,       //  The host cache size
    E_PARA_AI_MODELVERSION      ,       //  The AI model version
    E_PARA_AI_PARAMTER1                 //  AI paramter 1
};

class AIAlgoBase : public IModelResultListener, public AIInspectAlgoInterface
{
public:
    AIAlgoBase();
    virtual ~AIAlgoBase();

    eAIErrorCode StartRunAlgo(InspParamPtr spInspParam);
    void Initialize(int preThrdCnt, int preThrdPriority, int postThrdCnt, int postThrdPriority);
    void DeInitialize();
    void RegisterAlgoResultListener(IAlgoResultListener* listener);
    void UnRegisterAlgoResultListener(IAlgoResultListener* listener);

    
protected:
    virtual void OnPreProcess(TaskInfoPtr spTaskInfo) = 0;
    virtual void OnPostProcess(ModelResultPtr spResult, AlgoResultPtr& algoResult, HANDLE& hInspEnd) = 0;
    virtual void OnModelResult(ModelResultPtr spResult);

private:
    void PreProcessWorker();
    void PostProcessWorker();
    void WriteInBlob(ModelResultPtr spResult);
    void DrawResult(ModelResultPtr spResult);

private:
    std::atomic_bool m_bStopped;
    std::vector<std::thread*> m_vecPrepThrds;
    std::vector<std::thread*> m_vecPostThrds;
    rigtorp::MPMCQueue<InspParamPtr> m_queuePrep;
    rigtorp::MPMCQueue<ModelResultPtr> m_queuePost;
    std::vector<IAlgoResultListener*> m_vecResultListener;
    std::mutex m_queLock;
    std::condition_variable cond_;

protected:
    int m_nModelId;
};

#include "../utils/Utils.h"
#include "../utils/logger.h"
#include "AlgoManager.h"
#include "InferenceEngine.h"
#include "ErrorDefine.h"
#include "BaseAlgoGroup.h"


AlgoManager::AlgoManager()
{}

AlgoManager::~AlgoManager()
{}

ErrorCode AlgoManager::ConfigAlgoParams(const json& algo_all_cfg)
{
    //if (!m_algogroup_list.empty()) {
    //    LOGE("Algo params already configured!");
    //    return ErrorCode::WRONG_STATE;
    //}
    LOGE("Algo params start config");
    m_algogroup_list.clear();
    LOGE("Algo params start config 1 {}",m_algogroup_list.size());
    ErrorCode e_code = ErrorCode::OK;
    LOGE("Algo params start  config  1  {} ", Utils::DumpJson(algo_all_cfg));

    try
    {
        for (auto algo_group_cfg : algo_all_cfg)
        {
            LOGE("Algo params start  config  2_1");


            algo_all_cfg.contains("type_id");
            algo_all_cfg.contains("type_name");
            std::string type_id = algo_group_cfg["type_id"];
            std::string type_name = algo_group_cfg["type_name"];

            LOGE("Algo params start  config  2 type id {} ", type_id);
            LOGE("Algo params start  config  2 type name {} ", type_name);
            BaseAlgoGroup* algo_group = GetAlgoGroupByID(type_id);
            if (algo_group == nullptr) {
                algo_group = AlgoManager::CreateAlgoGroup(type_name + ALGO_GROUP_SUFFIX);
                if (algo_group == nullptr) {
                    LOGE("Create Algo Group fail. type_id:{} name:{}", type_id, type_name);
                    return ErrorCode::WRONG_PARAM;
                }
            }
            LOGE("Algo params start  config  3");
            e_code = algo_group->SetParams(algo_group_cfg);
            if (e_code != ErrorCode::OK) return e_code;

            m_algogroup_list.emplace_back(algo_group);
        }
    }
    catch (const std::exception& e )
    {
        LOGE("Algo params error  config  {}",e.what());
    }


    LOGE("Algo params end config");
    return ErrorCode::OK;
}

void AlgoManager::AddAlgo(BaseAlgo* pAlgo)
{
    auto it = std::find(m_algo_list.begin(), m_algo_list.end(), pAlgo);
    if (it == m_algo_list.end()) {
        m_algo_list.emplace_back(pAlgo);
    }
}

void AlgoManager::DeleteAlgo(BaseAlgo* pAlgo)
{
    auto it = std::find(m_algo_list.begin(), m_algo_list.end(), pAlgo);
    if (it != m_algo_list.end()) {
        m_algo_list.erase(it);
        delete (*it);
    }
}

BaseAlgo* AlgoManager::GetAlgo(std::string algo_name)
{
    for (auto algo : m_algo_list)
    {
        if (algo->GetName() == algo_name) {
            return algo;
        }
    }
    return nullptr;
}

BaseAlgoGroup* AlgoManager::GetAlgoGroupByID(std::string type_id)
{
    for (auto algogroup : m_algogroup_list)
    {
        if (algogroup->GetTypeID() == type_id) {
            return algogroup;
        }
    }
    return nullptr;
}

BaseAlgoGroup* AlgoManager::GetALgoGroupByName(std::string type_name)
{
    for (auto algogroup : m_algogroup_list)
    {
        if (algogroup->GetTypeName() == type_name) {
            return algogroup;
        }
    }
    return nullptr;
}

void AlgoManager::Destroy()
{
     for (auto algogroup : m_algogroup_list)
     {
        delete algogroup;
     }
     for (auto algo: m_algo_list)
     {
        delete algo;
     }
     m_algogroup_list.clear();
     m_algo_list.clear();
}






//----------------------------------------------------------------
// ----------- Register class functions below ---------------------
//----------------------------------------------------------------
void AlgoManager::RegisterAlgoGroup(const std::string &class_name, std::function<BaseAlgoGroup *()> constructor)
{
    sAlgoGroupConstructors()[class_name] = constructor;
    std::cout << "RegisterALgoGroup: " << class_name <<std::endl;
}

void AlgoManager::RegisterAlgo(const std::string &class_name, std::function<BaseAlgo *()> constructor)
{
    sAlgoConstructors()[class_name] = constructor;
    std::cout << "RegisterAlgo: " << class_name << std::endl;
}

BaseAlgoGroup *AlgoManager::CreateAlgoGroup(const std::string &class_name)
{
    auto it = sAlgoGroupConstructors().find(class_name);
    if (it != sAlgoGroupConstructors().end())
    {
        return it->second();
    }
    else
    {
        LOGE("Unknown AlgoGroup class name: {}", class_name);
        return nullptr;
    }
}

BaseAlgo *AlgoManager::CreateAlgo(const std::string &class_name)
{
    auto it = sAlgoConstructors().find(class_name);
    if (it != sAlgoConstructors().end())
    {
        return it->second();
    }
    else
    {
        LOGE("Unknown Algo class name: {}", class_name);
        return nullptr;
    }
}

#pragma once
#include <iostream>
#include "Defines.h"

enum class ErrorCode;
class BaseAlgo;
class BaseAlgoGroup;


// 算法管理器，负责管理算法、算法组(对应一张图片的多个算法)的注册，创建、运行等
class AlgoManager
{
public:
    DECLARE_SINGLETON(AlgoManager)

    // 注册AlgoGroup类，自定义算法Group通过REGISTER_ALGO_GROUP(name)进行类的注册，以便Framework自动创建对象
    static void RegisterAlgoGroup(const std::string& class_name, std::function<BaseAlgoGroup*()> constructor);
    // 注册Algo类，自定义算法通过REGISTER_ALGO_GROUP(name)进行类的注册，以便Framework自动创建对象
    static void RegisterAlgo(const std::string& class_name, std::function<BaseAlgo*()> constructor);

    // 创建算法组对象，AlgoEngine在解析算法json参数时候通过该方法创建算法组对象
    static BaseAlgoGroup* CreateAlgoGroup(const std::string& class_name);
    // 创建算法组对象，AlgoEngine在解析算法json参数时候通过该方法创建算法对象
    static BaseAlgo* CreateAlgo(const std::string& class_name);

    // 配置算法参数
    ErrorCode ConfigAlgoParams(const json& algo_cfg);
    // 添加算法对象到列表
    void AddAlgo(BaseAlgo* pAlgo);
    // 从列表中移除算法对象
    void DeleteAlgo(BaseAlgo* pAlgo);
    // 根据算法名字获取算法对象指针
    BaseAlgo* GetAlgo(std::string algo_name);
    // 根据图片type_id获取对应的算法组对象指针
    BaseAlgoGroup* GetAlgoGroupByID(std::string type_id);
    // 根据图片对应的算法组名称获取对应的算法组对象指针
    BaseAlgoGroup* GetALgoGroupByName(std::string algowrapper_name);
    // 释放资源
    void Destroy();


private:
    IMPLEMENT_SINGLETON(AlgoManager)

    // 已注册的AlgoGroup构造函数map
    static std::unordered_map<std::string, std::function<BaseAlgoGroup*()>>& sAlgoGroupConstructors() {
        static std::unordered_map<std::string, std::function<BaseAlgoGroup*()>> instance;
        return instance;
    }
    // 已注册的Algo构造函数map
    static std::unordered_map<std::string, std::function<BaseAlgo*()>>& sAlgoConstructors() {
        static std::unordered_map<std::string, std::function<BaseAlgo*()>> instance;
        return instance;
    }


private:
    std::vector<BaseAlgo*> m_algo_list;             // 算法列表
    std::vector<BaseAlgoGroup*> m_algogroup_list;   // 算法组列表
    json m_common_cfg;      // 系统配置参数
    json m_algo_all_cfg;    // 算法配置参数
};
#pragma once

#include <chrono>
#include <filesystem>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include "nlohmann/json.hpp"

#define PI 3.14159265358979323846

namespace fs = std::filesystem;
using json = nlohmann::json;
using TimePoint = std::chrono::system_clock::time_point;
using ImagePtr = unsigned char*;
typedef void* IntPtr;

#if defined(_WIN32) && !defined(_NODLL)
#  if defined(EXPORT_UTILITY_DLL)
#    define ExportAPI __declspec(dllexport)
#  elif defined(IMPORT_UTILITY_DLL)
#    define ExportAPI __declspec(dllimport)
#  else
#    define ExportAPI
#  endif
#elif defined(__GNUC__) && (__GNUC__ >= 4)
#  define ExportAPI __attribute__((visibility("default")))
#else
#  define ExportAPI
#endif


#ifndef DECLARE_SINGLETON
#define DECLARE_SINGLETON(class_name) \
private: \
    class_name(); \
    ~class_name(); \
    class_name(const class_name&) = delete; \
    class_name& operator=(const class_name&) = delete; \
public: \
    static class_name* get_instance() { \
        static class_name instance; \
        return &instance; \
    }
#endif


#define DCLEAR_TOOL(name) \
private: \
    static const bool name##_registered; \
public: \
    virtual std::string GetName(); \
    virtual ToolCategory GetCategory();


#define IMPLEMENT_TOOL(name, category) \
const bool name::name##_registered = ( \
    CCDAlgo::RegisterTool( \
        stToolClassInfo(#name, category, [](const std::string& tool_key){ return std::make_shared<name>(tool_key); }) \
    ) \
, true); \
std::string name::GetName() { return #name; } \
ToolCategory name::GetCategory() { return category; }

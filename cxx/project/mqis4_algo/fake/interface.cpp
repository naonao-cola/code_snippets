#include "logger.h"
#include "interface.h"
#include <nlohmann/json.hpp>
#include "utils.h"
#include <thread>         // std::this_thread::sleep_for
#include <chrono>         // std::chrono::seconds
using json = nlohmann::json;

class FakeGtmcOcrAlgo {
public:
    FakeGtmcOcrAlgo(std::string tapp_path, int device_id=0);

    bool load();

    void config(const char *config_json_str);

    const char* run(const char *in_param_json_str);

private:
    json m_config;
    std::string m_last_result;
};

FakeGtmcOcrAlgo::FakeGtmcOcrAlgo(std::string tapp_path, int device_id)
{

}

bool FakeGtmcOcrAlgo::load() {
    LOG_INFO("load tapp");
    std::this_thread::sleep_for (std::chrono::seconds(3));;
    return true;
}

void FakeGtmcOcrAlgo::config(const char *config_json_str) {
    LOG_INFO("start parse json: {}", config_json_str);
    std::string utf8_config_json_str = AnsiToUtf8(std::string(config_json_str));
    m_config = json::parse(utf8_config_json_str);
    std::this_thread::sleep_for (std::chrono::seconds(1));;
    LOG_INFO("end parse json: {}", m_config.dump());
}

const char* FakeGtmcOcrAlgo::run(const char *in_param_json_str) {
    json labelset =  json::array();

    json in_param = json::parse(in_param_json_str);
    for (auto img_p: in_param) {
        json out = json::array();

        for (auto task: m_config["task"]) {
            std::string name = task["name"];
            json result;
            std::this_thread::sleep_for (std::chrono::milliseconds(200));;
            if (task["type"] == "static_text") {
                result = {{"text", name}};
            } else if (task["type"] == "dynamic_text") {
                result = {{"text", name}};
            } else if (task["type"] == "barcode") {
                result = {{"text", "2398HDF238HDF"}};
            } else if (task["type"] == "qrcode") {
                result = {{"text", "IASDF3289KJSDF90"}};
            }

            json _out = {
                {"label", name},
                {"shapeType", "polygon"},
                {"points", task["roi"]},
                {"result", result}
            };
            out.push_back(_out);
        }

        json img_result = {
            {"imageName", ""},
            {"imagePath", ""},
            {"status", "OK"},
            {"shapes", out}
        };

        labelset.push_back(img_result);
    }

    json result = {
        {"classList", {}},
        {"labelSet", labelset}
    };
    m_last_result = Utf8ToAnsi(result.dump());
    return m_last_result.c_str();
}

int tapp_model_package(const char *model_path, char *origin_model_dir) {
    return 0;
}

int* tapp_model_open(const char *model_path, int device_id) {
    FakeGtmcOcrAlgo* m = new FakeGtmcOcrAlgo(model_path, device_id);
    if (!m->load()) {
        delete m;
        return nullptr;
    }
    LOG_INFO("handle: {}", (void*)m);
    return (int*)m;
}

void tapp_model_config(int *handle, const char *config_json_str) {
    LOG_INFO("enter ocr algo config: {}", (void*)handle);
    FakeGtmcOcrAlgo* m = (FakeGtmcOcrAlgo*)handle;
    if (m != nullptr) {
        m->config(config_json_str);
    } else {
        LOG_ERROR("handle is nullptr!!");
    }
}

const char* tapp_model_run(int *handle, const char *in_param_json_str) {
    FakeGtmcOcrAlgo* m = (FakeGtmcOcrAlgo*)handle;
    if (m != nullptr) {
        return m->run(in_param_json_str);
    } else {
        LOG_ERROR("handle is nullptr!!");
    }
}

void tapp_model_close(int *handle) {
    FakeGtmcOcrAlgo* m = (FakeGtmcOcrAlgo*)handle;
    if (m != nullptr)
        delete m;
}

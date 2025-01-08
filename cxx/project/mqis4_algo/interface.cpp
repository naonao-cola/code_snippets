#include "logger.h"
#include "interface.h"
#include "gtmc_ocr_algo.h"
#include "TivalCore.h"

// extern int verify_license(int m, int b, int *c);
// extern std::string get_hwid();

// const char* get_hardware_id()
// {
//     return get_hwid().c_str();
// }

// int tapp_license_verify() {
//     long int bbb = static_cast<long int> (std::time(NULL)) && 0xffffff;
//     int ccc;
//     int err = verify_license(0, bbb, &ccc);
//     return err;
// }

static GtmcOcrAlgo* g_m_ptr = nullptr;

int tapp_model_package(const char *model_path, char *origin_model_dir, char *model_key) {

    GtmcOcrAlgo* m = new GtmcOcrAlgo(model_path);
    m->package_model(origin_model_dir, model_key);
    m->save(model_key);
    return 0;
}

int* tapp_model_open(const char *model_path, int device_id) {
    int ret = Tival::TivalCore::Init(2);
    if (ret != 0) {
        LOG_ERROR("tapp_model_open(). License error:{}", ret);
        return nullptr;
    }
    if (!g_m_ptr) {
        g_m_ptr = new GtmcOcrAlgo(model_path, device_id);
    } else{
    
    }
    //GtmcOcrAlgo* m = new GtmcOcrAlgo(model_path, device_id);
    if (!g_m_ptr->load()) {
        delete g_m_ptr;
        return nullptr;
    }
    LOG_INFO("handle: {}", (void *)g_m_ptr);
    return (int *)g_m_ptr;
}

void tapp_model_config(int *handle, const char *config_json_str) {
    GtmcOcrAlgo* m = (GtmcOcrAlgo*)handle;
    if (m != nullptr) {
        m->config(config_json_str);
    } else {
        LOG_ERROR("handle is nullptr!!");
    }
}

const char* tapp_model_run(int *handle, const char *in_param_json_str) {
    GtmcOcrAlgo* m = (GtmcOcrAlgo*)handle;
    if (m != nullptr) {
        return m->run(in_param_json_str);
    } else {
        LOG_ERROR("handle is nullptr!!");
    }
}

void tapp_model_close(int *handle) {
    GtmcOcrAlgo* m = (GtmcOcrAlgo*)handle;
    if (m != nullptr)
        delete m;
}

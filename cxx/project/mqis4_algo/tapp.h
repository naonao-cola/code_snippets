#pragma once

#include <nlohmann/json.hpp>
#include <string.h>
#include "binn.h"

using json = nlohmann::json;

class Tapp {

public:
    Tapp(std::string tapp_path);
    ~Tapp();
    // bool load(std::string model_key);
    bool save(std::string model_key);

    bool set_info(const json &info);

    bool get_blob(std::string key, char **pptr, int *psize);
    bool set_blob(std::string key, char *ptr, int size);
    void free_buffer(binn *binn_obj);

    static int read_file(std::string file_path, char **pptr);

protected:
    json m_info;
    std::string m_tapp_path;
    binn *m_obj;
};

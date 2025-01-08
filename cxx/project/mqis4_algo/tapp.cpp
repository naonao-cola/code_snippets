#include <fstream>
#include "tapp.h"
#include "logger.h"

Tapp::Tapp(std::string tapp_path):
    m_tapp_path(tapp_path),
    m_obj(nullptr)
{
}

Tapp::~Tapp() {
    // if (m_obj != nullptr) {
    //     LOG_INFO("free binn object");
    //     binn_free(m_obj);
    //     LOG_INFO("release binn done");
    // }
}

int Tapp::read_file(std::string file_path, char **pptr) {
    std::ifstream is(file_path, std::ios::binary);
    is.seekg (0, is.end);
    int length = is.tellg();
    is.seekg (0, is.beg);

    char* buffer = new char[length];
    *pptr = buffer;
    is.read(buffer, length);
    return length;
}

void _delete_buf(void* buf) {
    delete buf;
}

void Tapp::free_buffer(binn* binn_obj)
{
    if (binn_obj != nullptr) {
        binn_obj->freefn = _delete_buf;
        binn_free(binn_obj);
        binn_obj = nullptr;
        LOG_INFO("free binn buffer.");
    }
}

// bool Tapp::load(std::string model_key) {
//     if (m_obj == nullptr) {
//         char *buffer;
//         std::string tapp_path = m_tapp_path + "/" + model_key + ".tapp";
//         int length = Tapp::read_file(tapp_path, &buffer);
//         m_obj = binn_open(buffer);
//         m_obj->freefn = _delete_buf;

//         // char *cstr;
//         // binn_object_get_str(m_obj, "info_json", &cstr);
//         // m_info = json::parse(cstr);
//         // LOG_INFO(" TAPP info_json: {}", std::string(cstr));
//     }
//     return true;
// }

bool Tapp::save(std::string model_key) {
    if (m_obj) {
        std::string tapp_path = m_tapp_path + "/" + model_key + ".tapp";
        std::ofstream os(tapp_path, std::ios::binary);
        os.write((char*)binn_ptr(m_obj), binn_size(m_obj));
        LOG_INFO("Save: {}", tapp_path);
        return true;
    }
    return false;
}

bool Tapp::set_info(const json &_info) {
    if (m_obj == nullptr) {
        m_obj = binn_object();
    }
    m_info = _info;
    std::string info_str = _info.dump();
    char * cstr = new char [info_str.length()+1];
    std::strcpy (cstr, info_str.c_str());

    binn_object_set_str(m_obj, "info_json", cstr);
    delete cstr;
    return true;
}

bool Tapp::get_blob(std::string key, char **pptr, int *psize) {
    if (m_obj) {
        binn_object_get_blob(m_obj, key.c_str(), (void**)pptr, psize);
    }
    return false;
}

bool Tapp::set_blob(std::string key, char *ptr, int size) {
    if (m_obj == nullptr) {
        m_obj = binn_object();
    }
    binn_object_set_blob(m_obj, key.c_str(), ptr, size);
    return true;
}

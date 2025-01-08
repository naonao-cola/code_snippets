#include "define_algorithm_pram.h"
#include "logger.h"
Cv_Pram::Cv_Pram() {

}

Cv_Pram::~Cv_Pram() {


}

void Cv_Pram::get_cv_pram(const char* value, json& cv_pram_json) {
    
    const_char_to_json(value, cv_pram_json);
    LOG_INFO("get cv pram ok!!!");
    return;
}

void Cv_Pram::json2cvres(BL_CV_PARM& pram, json cv_pram_json) {
    
    return;
}

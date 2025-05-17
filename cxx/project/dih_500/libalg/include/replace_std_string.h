//
// Created by y on 23-9-14.
//

#ifndef RKNN_ALG_DEMO_REPLACE_STD_STRING_H
#define RKNN_ALG_DEMO_REPLACE_STD_STRING_H
#include <string>
#include <sstream>
namespace std{
template<typename T> std::string to_string(const T&n){
  std::ostringstream stm;
  stm<<n;
  return stm.str();
}
}





#endif  // RKNN_ALG_DEMO_REPLACE_STD_STRING_H

//
// Created by y on 2023/12/7.
//

#ifndef TEST_LIBALG_TEMP_TEST_H
#define TEST_LIBALG_TEMP_TEST_H
#include <iostream>
bool TestInclineNetwork(const std::string& model_path,
                        const std::string& model_label_path,
                        const std::string& img_path);

bool TestSegNetwork(const std::string& model_path,
                        const std::string& model_label_path,
                        const std::string& img_path);
int TestClusterMatrix();
int TestOpencl();
#endif  // TEST_LIBALG_TEMP_TEST_H

cmake_minimum_required(VERSION 3.4.1)
#÷∏∂®linux-android±‡“Î∆˜

set(CMAKE_C_COMPILER "/home/naonao/demo/3rdparty/rknn_tools/aarch64-linux-android-gcc4.9.x/aarch64-linux-android/bin/aarch64-linux-android-gcc")
set(CMAKE_CXX_COMPILER "/home/naonao/demo/3rdparty/rknn_tools/aarch64-linux-android-gcc4.9.x/aarch64-linux-android/bin/aarch64-linux-android-g++")
project(immune_project CXX C)

add_subdirectory(${CMAKE_SOURCE_DIR}/src)
add_subdirectory(${CMAKE_SOURCE_DIR}/process)
add_dependencies(immune_demo immune)
cmake_minimum_required(VERSION 3.10.1)
project(immune_demo CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pie -fPIE -std=c++11  -Wl,--allow-shlib-undefined -DANDROID_STL=c++_static  -D__ANDROID_API__=24  -pthread" )  #����c++�ı���ѡ��
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 " )  #����c�ı���ѡ��
#��������Ŀ¼�µ�Դ�ļ�


set(IMMUNE_DIR ${CMAKE_SOURCE_DIR}/src/include)
message(${IMMUNE_DIR})
message(${CMAKE_SOURCE_DIR})
include_directories(${IMMUNE_DIR})
include_directories(${CMAKE_SOURCE_DIR})
add_executable(${PROJECT_NAME}  main.cpp)


target_link_libraries(${PROJECT_NAME} ${CMAKE_SOURCE_DIR}/third_party/lib/liblua.so)
target_link_libraries(${PROJECT_NAME} immune)

install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_SOURCE_DIR})


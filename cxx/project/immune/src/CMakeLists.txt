cmake_minimum_required(VERSION 3.4.1)
project(immune CXX)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -shared -pie -fPIE  -pthread -std=c++11  -Wl,--allow-shlib-undefined -DANDROID_STL=c++_static  -D__ANDROID_API__=24")  #����c++�ı���ѡ��
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99  ")  #����c�ı���ѡ��
#��������Ŀ¼�µ�Դ�ļ�
set(LUA_DIR ${CMAKE_SOURCE_DIR}/third_party/include)
include_directories(${LUA_DIR})
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/src/include)
aux_source_directory(${CMAKE_SOURCE_DIR}/cJSON sources)
set(LUA_C_JSON_DIR ${CMAKE_SOURCE_DIR}/lua-cjson-2.1.0)

add_library(${PROJECT_NAME} SHARED
        alg.cpp algjson.cpp algpeak.cpp lua_port.c immune.cpp
        ${sources}
        ${LUA_C_JSON_DIR}/lua_cjson.c
        ${LUA_C_JSON_DIR}/fpconv.c
        ${LUA_C_JSON_DIR}/strbuf.c
        include/immune.h
        qr_json.cpp
        crc16.c
)

set(LUA_LIB ${CMAKE_SOURCE_DIR}/third_party/lib/liblua.so)

target_link_libraries(${PROJECT_NAME} ${LUA_LIB})
install(TARGETS ${PROJECT_NAME}  DESTINATION ${CMAKE_SOURCE_DIR}/lib)
install(PROGRAMS ${LUA_LIB} DESTINATION ${CMAKE_SOURCE_DIR}/lib)





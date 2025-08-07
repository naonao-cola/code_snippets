cmake_minimum_required(VERSION 3.10.1)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_COMPILER "/c++/gcc/aarch64-linux-android-gcc4.9.x/aarch64-linux-android/bin/aarch64-linux-android-g++")
#-DANDROID_ABI=arm64-v8a
project(test_libalg CXX)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -pie -fPIE -std=c++11  -Wl,--allow-shlib-undefined -DANDROID_STL=c++_static  -D__ANDROID_API__=24  -pthread " )  #����c++�ı���ѡ��

# rknn_alg_demo
include_directories(${CMAKE_SOURCE_DIR}/libalg/include)
include_directories(${CMAKE_SOURCE_DIR}/libalg)
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/libalg/include/libopencl-stub/include)

include_directories(
        ${CMAKE_SOURCE_DIR}/local_test/
        ${CMAKE_SOURCE_DIR}/local_test/human
        ${CMAKE_SOURCE_DIR}/local_test/local_xml_config
        ${CMAKE_SOURCE_DIR}/libalg/make_result
        ${CMAKE_SOURCE_DIR}/libalg/model_config
        ${CMAKE_SOURCE_DIR}/libalg/opencl_tools
        ${CMAKE_SOURCE_DIR}/libalg/tinyxml2
#        ${CMAKE_SOURCE_DIR}/libalg/include/log4cplus
        ${CMAKE_SOURCE_DIR}/dihLogPlog
        /home/y/c++/rknpu2/examples/third-party
        )

add_executable(app_local_test
        main.cpp
        libalg/alg_task_flow_queue.cpp
        libalg/neural_network.cpp
        libalg/alg_clarity.cpp
        libalg/alg_heamo.cpp
        libalg/ai.cpp
        libalg/imgprocess.cpp
        libalg/utils.cpp
        libalg/libalgcell.cpp
        libalg/libalgimm.cpp
        local_test/human/DetectHuman.cpp
        local_test/UnitTest.cpp
        local_test/ProjectManager.cpp
        local_test/IntegratedCode.cpp
        local_test/IntegratedCodeClarity.cpp
        local_test/IntegratedCodeHeamo.cpp
        local_test/IntegratedCodeImmune.cpp
        local_test/local_xml_config/ParseXML.cpp
        libalg/tinyxml2/tinyxml2.cpp
        local_test/IntegratedCodeHybrid.cpp
        local_test/temp_test.cpp
        libalg/make_result/NormalNetwork.cpp
        libalg/make_result/ParamFitting.cpp
        libalg/make_result/Calibration.h
        libalg/make_result/Calibration.cpp
        libalg/make_result/SphericalNetwork.cpp
        libalg/make_result/TinnyNetwork.cpp
        libalg/make_result/NormalFitting.cpp
        libalg/make_result/SphericalFitting.cpp
        libalg/alg_heamo_human_model_call.cpp
        libalg/alg_heamo_human.cpp
        libalg/alg_heamo_milk.cpp
        libalg/make_result/AlarmManager.cpp
        libalg/model_config/ModelConfig.cpp
        libalg/timecnt.cpp
        local_test/test_opencl_nms.cpp
        libalg/opencl_tools/NmsCl.cpp
        libalg/alg_heamo_milk_model_call.cpp
)



#公共so
set(RGA_LIB ${CMAKE_SOURCE_DIR}/app/link_lib/librga.so)
set(RKNN_RT_LIB     ${CMAKE_SOURCE_DIR}/app/link_lib/librknnrt.so)
set(OPENCV_CORE_LIB   ${CMAKE_SOURCE_DIR}/app/link_lib/libopencv_core.so)
set(OPENCV_CODECS_LIB  ${CMAKE_SOURCE_DIR}/app/link_lib/libopencv_imgcodecs.so)
set(OPENCV_IMGPROC_LIB ${CMAKE_SOURCE_DIR}/app/link_lib/libopencv_imgproc.so)
set(LOG_LIB ${CMAKE_SOURCE_DIR}/app/link_lib/libDIHLog.so)
#set(LOG4PLUS ${CMAKE_SOURCE_DIR}/app/lib/liblog4cplus.so)
set(OPENCL_LIB ${CMAKE_SOURCE_DIR}/app/link_lib/libOpenCL.so)

#编译so
set(LUA_LIB ${CMAKE_SOURCE_DIR}/app/link_lib/liblua.so)
set(IMMUNE_LIB ${CMAKE_SOURCE_DIR}/app/link_lib/libimmune.so)
target_link_libraries(app_local_test
        ${LUA_LIB}
        ${IMMUNE_LIB}
        ${PROTOBUF_LIB}
        ${RGA_LIB}
        ${RKNN_RT_LIB}
        ${OPENCV_CORE_LIB}
        ${OPENCV_CODECS_LIB}
        ${OPENCV_IMGPROC_LIB}
        ${OPENCV_LIB}
        ${LOG_LIB}
        ${OPENCL_LIB}
        )

# install target and libraries
set(TEST_XML_DOCUMENT_PATH ./alg_config.xml)
set(TEST_LUA_DOCUMENT_PATH ${CMAKE_SOURCE_DIR}/app/lua/alg.lua)
#set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/rknn_alg_demo_${CMAKE_SYSTEM_NAME})

install(TARGETS app_local_test DESTINATION ${CMAKE_SOURCE_DIR})

#install(PROGRAMS ${RKNN_RT_LIB} DESTINATION lib)
#install(PROGRAMS ${RGA_LIB} DESTINATION lib)
#install(PROGRAMS ${PROTOBUF_LIB} DESTINATION lib)
#install(PROGRAMS ${LUA_LIB} DESTINATION lib)
#install(PROGRAMS ${IMMUNE_LIB} DESTINATION lib)
#
#install(PROGRAMS ${OPENCV_CORE_LIB} DESTINATION lib)
#install(PROGRAMS ${OPENCV_CODECS_LIB} DESTINATION lib)
#install(PROGRAMS ${OPENCV_IMGPROC_LIB} DESTINATION lib)
#install(PROGRAMS ${LOG_LIB} DESTINATION lib)
#
#
#install(PROGRAMS ${TES_LUA_DOCUMENT_PATH} DESTINATION ./)
#install(PROGRAMS ${TEST_XML_DOCUMENT_PATH} DESTINATION ./)



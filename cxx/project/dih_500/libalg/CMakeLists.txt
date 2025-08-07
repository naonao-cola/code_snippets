project(algLib)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
ADD_COMPILE_OPTIONS(-fPIC -pthread -std=c++11 -DANDROID_ABI=arm64-v8a -DANDROID_STL=c++_static -D__ANDROID_API__=24)
set(PROJECT_AIG_DIR ${PROJECT_SOURCE_DIR} CACHE INTERNAL "AIG算法目录" )
message("PROJECT_AIG_DIR ${PROJECT_AIG_DIR}")
include_directories(
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include/libopencl-stub/include
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/model_config
        ${PROJECT_SOURCE_DIR}/make_result
        ${PROJECT_SOURCE_DIR}/opencl_tools
        ${PROJECT_PLOG_DIR}
        /home/dyno/workspace/rk3588/android/lib/opencv/sdk/native/jni/include
)
link_directories(${PROJECT_LIB_DIR})
message("PROJECT_LIB_DIR ${PROJECT_LIB_DIR}")
set(source_files
        ai.cpp
        alg_clarity.cpp
        alg_heamo.cpp
        imgprocess.cpp
        libalgimm.cpp
        libalgcell.cpp
        neural_network.cpp
        alg_task_flow_queue.cpp
        timecnt.cpp
        utils.cpp
        make_result/Calibration.cpp
        make_result/NormalFitting.cpp
        make_result/NormalNetwork.cpp
        make_result/ParamFitting.cpp
        make_result/SphericalFitting.cpp
        make_result/SphericalNetwork.cpp
        make_result/TinnyNetwork.cpp
        model_config/ModelConfig.cpp
        tinyxml2/tinyxml2.cpp
        alg_heamo_human_model_call.cpp
        alg_heamo_milk_model_call.cpp
        #        dynoLog.cpp

)
add_library(${PROJECT_NAME} SHARED ${source_files})
target_link_libraries(${PROJECT_NAME}
        #        ${OpenCV_LIBS}
        opencv_imgcodecs
        opencv_imgproc
        opencv_core
        DIHLog
        rga
        lua
        immune
        rknnrt
        OpenCL
        eventLib
)
INSTALL(TARGETS ${PROJECT_NAME} DESTINATION lib)

cmake_minimum_required(VERSION 3.10)

SET(CMAKE_TOOLCHAIN_FILE /home/naonao/demo/3rdparty/android-ndk-r17c/build/cmake/android.toolchain.cmake)
SET(ANDROID_ABI "arm64-v8a")
SET(ANDROID_NDK /home/naonao/demo/3rdparty/android-ndk-r17c)
SET(ANDROID_PLATFORM android-26)
SET(ANDROID_STL c++_shared)
SET(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

message(STATUS "${CMAKE_TOOLCHAIN_FILE}")
message(STATUS "${ANDROID_ABI}")
message(STATUS "${ANDROID_NDK}")
message(STATUS "${ANDROID_PLATFORM}")
message(STATUS "${ANDROID_STL}")

project(OpenCVExample)

if(NOT WIN32)
    string(ASCII 27 Esc)
    set(ColourReset "${Esc}[m")
    set(ColourBold "${Esc}[1m")
    set(Red "${Esc}[31m")
    set(Green "${Esc}[32m")
    set(Yellow "${Esc}[33m")
    set(Blue "${Esc}[34m")
    set(Magenta "${Esc}[35m")
    set(Cyan "${Esc}[36m")
    set(White "${Esc}[37m")
    set(BoldRed "${Esc}[1;31m")
    set(BoldGreen "${Esc}[1;32m")
    set(BoldYellow "${Esc}[1;33m")
    set(BoldBlue "${Esc}[1;34m")
    set(BoldMagenta "${Esc}[1;35m")
    set(BoldCyan "${Esc}[1;36m")
    set(BoldWhite "${Esc}[1;37m")
endif()

# install 设置
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_INSTALL_PREFIX "../install")

# opencv
# xxxx目录包含OpenCVConfig.cmake
set(OpenCV_DIR /home/naonao/demo/3rdparty/test/opencv410_android/sdk/native/jni)
find_package(OpenCV REQUIRED)
message(STATUS "${Green} OpenCV_LIBRARIES is: ${OpenCV_LIBRARIES}")

# include
include_directories(${OpenCV_INCLUDE_DIRS})
message(STATUS "${Green} OpenCV_INCLUDE_DIRS is: ${OpenCV_INCLUDE_DIRS}")

# source
FILE(GLOB SRCS ./src/*.cpp)
add_executable(${PROJECT_NAME} ${SRCS}) # *.cpp指要编译的那些源文件
message(STATUS "${Green} Source file is: ${SRCS}")
message(" ${ColourReset}")

# link
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES}) # install

install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin # 可执行文件安装路径
)
message(STATUS "Include directories: ${INCLUDE_DIRECTORIES}")
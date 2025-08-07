#!/bin/sh

# 文件夹路径
folder="build"

# 判断文件夹是否存在
if [ -d "$folder" ]; then
    # 如果存在，则删除
    rm -rf "$folder"
    echo "'$folder' 已删除,新建 build 文件夹"
    mkdir build && cd build
else
    echo "文件夹 '$folder' 不存在"
    mkdir build && cd build
fi

# cmake  -DCMAKE_TOOLCHAIN_FILE=/home/naonao/demo/3rdparty/android-ndk-r17c/build/cmake/android.toolchain.cmake \
#     -DCMAKE_VERBOSE_MAKEFILE=ON \
#     -DANDROID_ABI="arm64-v8a" \
#     -DANDROID_NDK=/home/naonao/demo/3rdparty/android-ndk-r17c \
#     -DANDROID_PLATFORM=android-26 \
#     -DANDROID_STL=c++_shared \
#     ..

cmake  .. 
make -j8







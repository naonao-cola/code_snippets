#lib项目信息
project (testlib)
#
SET(source_h
	${PROJECT_SOURCE_DIR}/metaObject.h
  )
  
SET(source_cpp
	${PROJECT_SOURCE_DIR}/testlib.cpp
	)
  
#头文件目录
include_directories(${PROJECT_SOURCE_DIR})
#

if (${WIN_OS})
#将库文件输出到Debug或Release目录下,文件目录编译时自动创建
set(LIBRARY_OUTPUT_PATH  ${PROJECT_SOURCE_DIR}/../bin)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
add_library(testlibd SHARED ${source_h} ${source_cpp})

else(CMAKE_BUILD_TYPE)
add_library(testlib SHARED ${source_h} ${source_cpp})
endif (CMAKE_BUILD_TYPE)

else(${WIN_OS})

set(LIBRARY_OUTPUT_PATH  ${PROJECT_SOURCE_DIR}/../bin/linux)
# 指定生成目标
add_library(testlib SHARED ${source_h} ${source_cpp})

endif(${WIN_OS})

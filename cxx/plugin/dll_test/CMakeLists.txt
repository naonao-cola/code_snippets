# CMake 最低版本号要求
cmake_minimum_required (VERSION 2.8)
# 项目信息
project (dll_test)
#
if(WIN32)
    message(STATUS "windows compiling...")
    add_definitions(-D_PLATFORM_IS_WINDOWS_)
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
	set(WIN_OS true)
else(WIN32)
    message(STATUS "linux compiling...")
    add_definitions( -D_PLATFORM_IS_LINUX_)
    add_definitions("-Wno-invalid-source-encoding")
	  # add_definitions("-O2")
    set(UNIX_OS true)
    set(_DEBUG true)
    
endif(WIN32)

#
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

# 指定源文件的目录,并将名称保存到变量
SET(source_h_lib
	${PROJECT_SOURCE_DIR}/lib/metaObject.h
  )
  
SET(source_h_src
	${PROJECT_SOURCE_DIR}/src/dlload.h
	${PROJECT_SOURCE_DIR}/src/register.h
  )
  
SET(source_cpp_src
	${PROJECT_SOURCE_DIR}/src/dlload.cpp
	${PROJECT_SOURCE_DIR}/src/register.cpp
	${PROJECT_SOURCE_DIR}/src/test.cpp
  )
  
#头文件目录
include_directories(
	${PROJECT_SOURCE_DIR}/lib
	${PROJECT_SOURCE_DIR}/src
	)
	
add_subdirectory(${PROJECT_SOURCE_DIR}/lib ./lib)

if (${UNIX_OS})

add_definitions(
  "-W"
  "-fPIC"
  "-Wall"
  # "-Wall -g"
  "-Werror"
  "-Wshadow"
  "-Wformat"
  "-Wpointer-arith"
  "-D_REENTRANT"
  "-D_USE_FAST_MACRO"
  "-Wno-long-long"
  "-Wuninitialized"
  "-D_POSIX_PTHREAD_SEMANTICS"
  "-DACL_PREPARE_COMPILE"
  "-Wno-unused-parameter"
  "-fexceptions"
  )
  
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")

link_directories(
	"${EXECUTABLE_OUTPUT_PATH}"/linux
)

# 指定生成目标
add_executable(dll_test ${source_h_lib} ${source_h_src} ${source_cpp_src})
#link
target_link_libraries(dll_test testlib -ldl)

endif(${UNIX_OS})

if (${WIN_OS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4819")

add_definitions(
  "-D_CRT_SECURE_NO_WARNINGS"
  "-D_WINSOCK_DEPRECATED_NO_WARNINGS"
  "-DNO_WARN_MBCS_MFC_DEPRECATION"
  "-DWIN32_LEAN_AND_MEAN"
)

#link_directories()
  
if (CMAKE_BUILD_TYPE STREQUAL "Debug")

# 指定生成目标
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/bin)
# 指定生成目标
add_executable(dll_testd ${source_h_lib} ${source_h_src} ${source_cpp_src})

link_directories(
	"${EXECUTABLE_OUTPUT_PATH}"/Debug
)

else(CMAKE_BUILD_TYPE)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/bin)
# 指定生成目标
add_executable(dll_test ${source_h_lib} ${source_h_src} ${source_cpp_src})

link_directories(
	"${EXECUTABLE_OUTPUT_PATH}"/Release
)

endif (CMAKE_BUILD_TYPE)

endif(${WIN_OS})

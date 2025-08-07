# Makefile for sample program
.PHONY : all clean

# path settings
COMPILE_PATH := /c++/gcc/aarch64-linux-android-gcc4.9.x/aarch64-linux-android/bin/aarch64-linux-android-
PROJECT_PATH := /home/y/c++/alg_git/dih2/alg/DIH-ALG
PROJECT_RUNTIME_LIB_PATH = $(PROJECT_PATH)/app/lib
PROJECT_LINK_LIB_PATH = $(PROJECT_PATH)/app/link_lib

THIRD_PARTY_PATH := /home/y/c++/rknpu2/examples/third-party

# the program to build
NAME := app_local_test

#MKDIR = mkdir
#DIRS  = build
#all:$(DIRS)
#$(DIRS):
	#$(MKDIR) $@

CUR_PATH = ./
#OBJ_PATH = $(CUR_PATH)build/
OBJ_PATH = $(CUR_PATH)/
OPT_PATH = $(CUR_PATH)

# Build tools and flags
CXX := $(COMPILE_PATH)g++
LD := $(COMPILE_PATH)g++

#SRCS := $(wildcard *.cpp)
SRCS += main.cpp ./local_test/human/DetectHuman.cpp ./local_test/ProjectManager.cpp ./local_test/UnitTest.cpp
SRCS += ./local_test/IntegratedCode.cpp ./local_test/IntegratedCodeClarity.cpp  ./local_test/IntegratedCodeHeamo.cpp  ./local_test/IntegratedCodeImmune.cpp
SRCS += ./local_test/temp_test.cpp ./local_test/local_xml_config/ParseXML.cpp ./local_test/test_opencl_nms.cpp
SRCS += ./libalg/neural_network.cpp  ./libalg/utils.cpp ./libalg/imgprocess.cpp ./local_test/IntegratedCodeHybrid.cpp
SRCS += ./libalg/libalgcell.cpp ./libalg/libalgimm.cpp ./libalg/alg_task_flow_queue.cpp  ./libalg/alg_clarity.cpp ./libalg/alg_heamo.cpp
SRCS += ./libalg/ai.cpp ./libalg/timecnt.cpp  ./libalg/alg_heamo_human_model_call.cpp ./libalg/alg_heamo_milk_model_call.cpp
SRCS += ./libalg/alg_heamo_human.cpp ./libalg/alg_heamo_milk.cpp ./libalg/project_utils.cpp ./libalg/FocusControl.cpp
SRCS += ./libalg/make_result/TinnyNetwork.cpp ./libalg/make_result/NormalNetwork.cpp ./libalg/make_result/SphericalNetwork.cpp
SRCS += ./libalg/make_result/ParamFitting.cpp ./libalg/make_result/Calibration.cpp
SRCS += ./libalg/make_result/NormalFitting.cpp ./libalg/make_result/SphericalFitting.cpp ./libalg/make_result/AlarmManager.cpp
SRCS += ./libalg/model_config/ModelConfig.cpp ./libalg/tinyxml2/tinyxml2.cpp
SRCS += ./libalg/opencl_tools/NmsCl.cpp
OBJS := $(patsubst %cpp, %o, $(SRCS))


#CPPFLAGS :=-pie -fPIE -std=c++11  -Wl,--allow-shlib-undefined -DANDROID_STL=c++_static  -D__ANDROID_API__=24  -pthread -fopenmp
CPPFLAGS :=-pie -fPIE -std=c++11  -Wl,--allow-shlib-undefined -DANDROID_STL=c++_static  -D__ANDROID_API__=24  -pthread
CPPFLAGS += -g
CPPFLAGS += -I./local_test -I./local_test/human -I./local_test/local_xml_config
CPPFLAGS += -I ./libalg -I./libalg/include -I./libalg/model_config -I./libalg/tinyxml2 -I./libalg/make_result
CPPFLAGS += -I ./libalg/include/libopencl-stub/include -I./libalg/opencl_tools
CPPFLAGS += -I ./libalg/include/nlohmann/
CPPFLAGS += -I$(THIRD_PARTY_PATH)
#ALL_OBJ_POS = $(addprefix $(OBJ_PATH), $(OBJS))

# OPENCL 宏定义,指定版本信息
DEFS += -D CL_VERSION_1_1=1 -D CL_TARGET_OPENCL_VERSION=110
CPPFLAGS += ${DEFS}


# ����ѡ��
# LDFLAGS += -pie # ����λ���޹صĿ�ִ���ļ�
LDFLAGS += -pthread # ���߳�֧��
LDFLAGS += -Wl,--allow-shlib-undefined # ����̬��δ����
LDFLAGS += -lopencv_core -lopencv_imgcodecs -lopencv_imgproc
LDFLAGS += -lrga -lrknnrt -llua -limmune -lDIHLog
LDFLAGS += -leventLib -lOpenCL
LDFLAGS += -L $(PROJECT_LINK_LIB_PATH)


all : $(NAME)

$(NAME) : $(OBJS)
	$(LD) -o $@ $^ $(CPPFLAGS) $(LDFLAGS)

%.o : %.cpp
	$(CXX) $(CPPFLAGS) -c $< -o $(OBJ_PATH)$@

clean :
	$(RM) $(OBJ_PATH)*.o ./libalg/*.o ./local_test/*.o ./local_test/human/*.o ./libalg/make_result/*.o ./libalg/model_config/*.o ./local_test/local_xml_config/*.o ./libalg/tinyxml2/*.o ./libalg/opencl_tools/*.o



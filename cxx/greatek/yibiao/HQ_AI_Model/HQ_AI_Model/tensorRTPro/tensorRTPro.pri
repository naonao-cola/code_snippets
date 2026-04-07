


CUDA_SOURCES += $$PWD/application/app_yolo/yolo_decode.cu \
                $$PWD/application/app_yolo_pose/yolo_pose_decode.cu \
                $$PWD/application/app_yolo_seg/yolo_seg_decode.cu \
                $$PWD/tensorRT/common/preprocess_kernel.cu \
                $$PWD/application/app_yolo_obb/yolo_obb_decode.cu \
                $$PWD/application/cuosd/cuosd_kernel.cu \
                $$PWD/application/cuosd/gpu_image.cu \
#    $$PWD/tensorRT/onnxplugin/plugins/DCNv2.cu \
#    $$PWD/tensorRT/onnxplugin/plugins/HSigmoid.cu \
#    $$PWD/tensorRT/onnxplugin/plugins/HSwish.cu \
#    $$PWD/tensorRT/onnxplugin/plugins/ScatterND.cu \


CUDA_DIR = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8"


#SYSTEM_NAME = x64
#SYSTEM_TYPE = 64
#CUDA_ARCH = compute_89
#CUDA_CODE = sm_89
#NVCC_OPTIONS = --use_fast_math

SYSTEM_TYPE = 64

NVCC_OPTIONS = --use-local-env


INCLUDEPATH += $$PWD/tensorRT     \
               $$PWD/application     \
               $$PWD/application/cuosd     \
               $$PWD/application/cuosd/textbackend     \


INCLUDEPATH += $$CUDA_DIR/include

CUDA_INC = $$join(INCLUDEPATH,'" -I"','-I"','"')




# 强制使用动态链接（MD/MDd）
QMAKE_CXXFLAGS_RELEASE -= -MD
QMAKE_CXXFLAGS_RELEASE += -MT
QMAKE_CXXFLAGS_DEBUG -= -MTd
QMAKE_CXXFLAGS_DEBUG += -MDd

# 传递给nvcc
CUDA_NVCC_FLAGS_RELEASE = -Xcompiler "/MT"
CUDA_NVCC_FLAGS_DEBUG = -Xcompiler "/MDd"



# Configuration of the Cuda compiler
CONFIG(debug, debug|release) {
    # Debug mode
    CUDA_OBJECTS_DIR = debug
    cuda_d.input = CUDA_SOURCES
    cuda_d.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}.obj
    cuda_d.commands = $$CUDA_DIR/bin/nvcc -D_DEBUG $$NVCC_OPTIONS $$CUDA_INC $$NVCC_LIBS --machine $$SYSTEM_TYPE $$CUDA_NVCC_FLAGS_DEBUG -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda_d.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda_d
}
else {
    # Release mode
    CUDA_OBJECTS_DIR = release
    cuda.input = CUDA_SOURCES
    cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}.obj
    cuda.commands = $$CUDA_DIR/bin/nvcc $$NVCC_OPTIONS $$CUDA_INC $$NVCC_LIBS --machine $$SYSTEM_TYPE $$CUDA_NVCC_FLAGS_RELEASE -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda
}





HEADERS += \
    $$PWD/application/app_yolo/multi_gpu.hpp \
    $$PWD/application/app_yolo/yolo.hpp \
    $$PWD/application/app_yolo_pose/yolo_pose.hpp \
    $$PWD/application/app_yolo_obb/yolo_obb.hpp \
    $$PWD/application/app_yolo_seg/yolo_seg.hpp \
    $$PWD/application/app_yolo_cls/yolo_cls.hpp \
    $$PWD/application/app_deimv2/deim.hpp \
    $$PWD/application/common/face_detector.hpp \
    $$PWD/application/common/object_detector.hpp \
    $$PWD/application/app_ppocr/clipper.hpp \
    $$PWD/application/app_ppocr/ppocr_cls.hpp \
    $$PWD/application/app_ppocr/ppocr_det.hpp \
    $$PWD/application/app_ppocr/ppocr_rec.hpp \
    $$PWD/application/app_ppocr/ppocr.hpp \
    $$PWD/application/app_ppocr/utils.hpp \
    $$PWD/application/cuosd/cuosd.h \
    $$PWD/application/cuosd/cuosd_kernel.h \
    $$PWD/application/cuosd/gpu_image.h \
    $$PWD/application/cuosd/memory.hpp \
    $$PWD/application/cuosd/stb_image_write.h \
    $$PWD/application/cuosd/textbackend/backend.hpp \
    $$PWD/application/cuosd/textbackend/pango-cairo.hpp \
    $$PWD/application/cuosd/textbackend/stb.hpp \
    $$PWD/application/cuosd/textbackend/stb_truetype.h \
#    $$PWD/tensorRT/builder/trt_builder.hpp \
    $$PWD/tensorRT/common/cuda_tools.hpp \
    $$PWD/tensorRT/common/ilogger.hpp \
    $$PWD/tensorRT/common/infer_controller.hpp \
    $$PWD/tensorRT/common/json.hpp \
    $$PWD/tensorRT/common/monopoly_allocator.hpp \
    $$PWD/tensorRT/common/preprocess_kernel.cuh \
    $$PWD/tensorRT/common/trt_tensor.hpp \
    $$PWD/tensorRT/infer/trt_infer.hpp \
#    $$PWD/tensorRT/onnx/onnx-ml.pb.h \
#    $$PWD/tensorRT/onnx/onnx-operators-ml.pb.h \
#    $$PWD/tensorRT/onnx/onnx_pb.h \
#    $$PWD/tensorRT/onnx/onnxifi.h \
#    $$PWD/tensorRT/onnx_parser/ImporterContext.hpp \
#    $$PWD/tensorRT/onnx_parser/LoopHelpers.hpp \
#    $$PWD/tensorRT/onnx_parser/ModelImporter.hpp \
#    $$PWD/tensorRT/onnx_parser/NvOnnxParser.h \
#    $$PWD/tensorRT/onnx_parser/OnnxAttrs.hpp \
#    $$PWD/tensorRT/onnx_parser/RNNHelpers.hpp \
#    $$PWD/tensorRT/onnx_parser/ShapeTensor.hpp \
#    $$PWD/tensorRT/onnx_parser/ShapedWeights.hpp \
#    $$PWD/tensorRT/onnx_parser/Status.hpp \
#    $$PWD/tensorRT/onnx_parser/TensorOrWeights.hpp \
#    $$PWD/tensorRT/onnx_parser/builtin_op_importers.hpp \
#    $$PWD/tensorRT/onnx_parser/onnx2trt.hpp \
#    $$PWD/tensorRT/onnx_parser/onnx2trt_common.hpp \
#    $$PWD/tensorRT/onnx_parser/onnx2trt_runtime.hpp \
#    $$PWD/tensorRT/onnx_parser/onnx2trt_utils.hpp \
#    $$PWD/tensorRT/onnx_parser/onnxErrorRecorder.hpp \
#    $$PWD/tensorRT/onnx_parser/onnx_utils.hpp \
#    $$PWD/tensorRT/onnx_parser/toposort.hpp \
#    $$PWD/tensorRT/onnx_parser/trt_utils.hpp \
#    $$PWD/tensorRT/onnx_parser/utils.hpp \
#    $$PWD/tensorRT/onnxplugin/onnxplugin.hpp \
#    $$PWD/tensorRT/onnxplugin/plugin_binary_io.hpp



SOURCES += $$CUDA_SOURCES \
    $$PWD/application/app_yolo/multi_gpu.cpp \
    $$PWD/application/app_yolo/yolo.cpp \
#    $$PWD/direct/direct_classifier.cpp \
#    $$PWD/direct/direct_mae.cpp \
#    $$PWD/direct/direct_unet.cpp \
#    $$PWD/direct/direct_yolo.cpp \
#    $$PWD/tensorRT/builder/trt_builder.cpp \
    $$PWD/application/app_yolo_pose/yolo_pose.cpp \
    $$PWD/application/app_yolo_obb/yolo_obb.cpp \
    $$PWD/application/app_yolo_seg/yolo_seg.cpp \
    $$PWD/application/app_yolo_cls/yolo_cls.cpp \
    $$PWD/application/app_deimv2/deim.cpp \
    $$PWD/application/app_ppocr/clipper.cpp \
    $$PWD/application/app_ppocr/postprocess_det.cpp \
    $$PWD/application/app_ppocr/ppocr_cls.cpp \
    $$PWD/application/app_ppocr/ppocr_det.cpp \
    $$PWD/application/app_ppocr/ppocr_rec.cpp \
    $$PWD/application/app_ppocr/ppocr.cpp \
    $$PWD/application/app_ppocr/utils.cpp \
    $$PWD/application/cuosd/cuosd.cpp \
    $$PWD/application/cuosd/textbackend/backend.cpp \
    $$PWD/application/cuosd/textbackend/pango-cairo.cpp \
    $$PWD/application/cuosd/textbackend/stb.cpp \
    $$PWD/tensorRT/common/cuda_tools.cpp \
    $$PWD/tensorRT/common/ilogger.cpp \
    $$PWD/tensorRT/common/json.cpp \
    $$PWD/tensorRT/common/trt_tensor.cpp \
    $$PWD/tensorRT/import_lib.cpp \
    $$PWD/tensorRT/infer/trt_infer.cpp \
#    $$PWD/tensorRT/onnx/onnx-ml.pb.cpp \
#    $$PWD/tensorRT/onnx/onnx-operators-ml.pb.cpp \
#    $$PWD/tensorRT/onnx_parser/LoopHelpers.cpp \
#    $$PWD/tensorRT/onnx_parser/ModelImporter.cpp \
#    $$PWD/tensorRT/onnx_parser/NvOnnxParser.cpp \
#    $$PWD/tensorRT/onnx_parser/OnnxAttrs.cpp \
#    $$PWD/tensorRT/onnx_parser/RNNHelpers.cpp \
#    $$PWD/tensorRT/onnx_parser/ShapeTensor.cpp \
#    $$PWD/tensorRT/onnx_parser/ShapedWeights.cpp \
#    $$PWD/tensorRT/onnx_parser/builtin_op_importers.cpp \
#    $$PWD/tensorRT/onnx_parser/onnx2trt_utils.cpp \
#    $$PWD/tensorRT/onnx_parser/onnxErrorRecorder.cpp \
#    $$PWD/tensorRT/onnxplugin/onnxplugin.cpp \
#    $$PWD/tensorRT/onnxplugin/plugin_binary_io.cpp



## Configuration of the Cuda compiler
#CONFIG(debug, debug|release) {
#    # Debug mode
#    cuda_d.input = CUDA_SOURCES
#    cuda_d.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}.obj
#    cuda_d.commands = $$CUDA_DIR/bin/nvcc -D_DEBUG $$NVCC_OPTIONS $$CUDA_INC $$NVCC_LIBS --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
#    cuda_d.dependency_type = TYPE_C
#    QMAKE_EXTRA_COMPILERS += cuda_d
#}
#else {
#    # Release mode
#    cuda.input = CUDA_SOURCES
#    cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}.obj
#    cuda.commands = $$CUDA_DIR/bin/nvcc $$NVCC_OPTIONS $$CUDA_INC $$NVCC_LIBS --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
#    cuda.dependency_type = TYPE_C
#    QMAKE_EXTRA_COMPILERS += cuda
#}

#--use-local-env
#-ccbin "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.40.33807\bin\HostX64\x64"
#-x cu
#-IC:\Program -IC:\Users\15883\Desktop\TensorRTX\tensorrtx\yolov5\Files\NVIDIA
#-IC:\Users\15883\Desktop\TensorRTX\tensorrtx\yolov5\GPU
#-IC:\Users\15883\Desktop\TensorRTX\tensorrtx\yolov5\Computing
#-IC:\Users\15883\Desktop\TensorRTX\tensorrtx\yolov5\Toolkit\CUDA\v11.8\include
#-IC:\Users\15883\Desktop\TensorRTX\tensorrtx\yolov5\src
#-IC:\Users\15883\Desktop\TensorRTX\tensorrtx\yolov5\plugin
#-ID:\beifen\opencv\build\include -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\include"
#--keep-dir myplugins\x64\Release
#-maxrregcount=0
#--machine
#64
#--compile
#-cudart
#static
#-std=c++11
#-Xcompiler="/EHsc -Ob2"
#-D_WINDOWS -DNDEBUG -DAPI_EXPORTS -D"CMAKE_INTDIR=\"Release\"" -Dmyplugins_EXPORTS -D_WINDLL -D_MBCS -D"CMAKE_INTDIR=\"Release\"" -Dmyplugins_EXPORTS -Xcompiler "/EHsc /W3 /nologo /O2 /Fdmyplugins.dir\Release\vc143.pdb /FS   /MD /GR" -o myplugins.dir\Release\yololayer.obj "C:\Users\15883\Desktop\TensorRTX\tensorrtx\yolov5\plugin\yololayer.cu"



## 添加CUDA支持
#CUDA_SOURCES +=  \
#    $$PWD/src/preprocess.cu   \
#    $$PWD/plugin/yololayer.cu

#CUDA_DIR = $$PWD/../../HQ_Share/lib/CUDA/v11.8



## 指定 nvcc 路径（Windows 示例）
#win32 {
#    CUDA_NVCC = $$CUDA_DIR/bin/nvcc.exe
#    QMAKE_EXTRA_COMPILERS += cuda
#}

#CUDA_OBJECTS_DIR = ./
## 强制使用 nvcc 编译 .cu 文件
##$$CUDA_DIR/bin/nvcc -D_DEBUG $$NVCC_OPTIONS $$CUDA_INC $$NVCC_LIBS --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
#cuda.commands = $$CUDA_NVCC  --machine 64 -c ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}
#cuda.dependency_type = TYPE_C
#cuda.input = CUDA_SOURCES
#cuda.output =   $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}.obj
#QMAKE_EXTRA_COMPILERS += cuda



## 强制统一迭代器调试级别
#CONFIG(debug, debug|release) {
#    # Debug 配置
#    DEFINES += _ITERATOR_DEBUG_LEVEL=2
#    CUDA_NVCC_FLAGS += -D_ITERATOR_DEBUG_LEVEL=2
#} else {
#    # Release 配置
#    DEFINES += _ITERATOR_DEBUG_LEVEL=0
#    CUDA_NVCC_FLAGS += -D_ITERATOR_DEBUG_LEVEL=0
#}


## MSVC编译器设置
#win32-msvc {
#    # 强制使用动态链接（MD/MDd）
#    QMAKE_CXXFLAGS_RELEASE -= -MD
#    QMAKE_CXXFLAGS_RELEASE += -MT
#    QMAKE_CXXFLAGS_DEBUG -= -MTd
#    QMAKE_CXXFLAGS_DEBUG += -MDd

#    # 传递给nvcc
#    CUDA_NVCC_FLAGS_RELEASE = -Xcompiler "/MD"
#    CUDA_NVCC_FLAGS_DEBUG = -Xcompiler "/MDd"
#}


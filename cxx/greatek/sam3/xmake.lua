
set_project("sam_test")
set_version("0.0.1",{soname = true})
set_languages("c++17")
add_rules("mode.debug", "mode.release","mode.releasedbg")
add_requires("opencv 4.8.x",{system = false})
add_requires("nlohmann_json",{system = false})
set_policy("build.progress_style", "multirow")
add_rules("plugin.vsxmake.autoupdate")


--trt cudnn
add_includedirs("E:/3rdparty/cudnn-windows-x86_64-8.9.7.29_cuda11-archive/cudnn-windows-x86_64-8.9.7.29_cuda11-archive/include/")
add_linkdirs("E:/3rdparty/cudnn-windows-x86_64-8.9.7.29_cuda11-archive/cudnn-windows-x86_64-8.9.7.29_cuda11-archive/lib/x64/")
add_includedirs("E:/3rdparty/TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8/TensorRT-8.6.1.6/include/")
add_linkdirs("E:/3rdparty/TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8/TensorRT-8.6.1.6/lib/")
add_rpathdirs("E:/3rdparty/TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8/TensorRT-8.6.1.6/lib")
add_rpathdirs("E:/3rdparty/cudnn-windows-x86_64-8.9.7.29_cuda11-archive/cudnn-windows-x86_64-8.9.7.29_cuda11-archive/bin/")
add_links("nvinfer",
"nvinfer_plugin",
"nvparsers",
"nvinfer_vc_plugin",
"nvonnxparser",
"cudnn",
"cudnn_adv_infer",
"cudnn_adv_train",
"cudnn_cnn_infer",
"cudnn_cnn_train",
"cudnn_ops_infer",
"cudnn_ops_train")



target("sam_test")
    set_kind("binary")
    add_packages("opencv")
    add_packages("nlohmann_json")
    add_rules("cuda")
    -- 设置编译路径
    -- 添加文件
    -- Explicitly add only the intended source files to avoid duplicate compilation
    add_files("$(projectdir)/src/main.cpp", "$(projectdir)/src/clip_bpe.cpp", "$(projectdir)/src/utils.cpp")

set_project("yuanba_meter")
set_version("0.0.1")
set_languages("c++17")

add_rules("mode.debug", "mode.release")
set_policy("build.progress_style", "multirow")
set_config("cuda_sdkver", "11.8")
-- 编译链
-- toolchain("my_toolchain")
--     set_kind("standalone")
--     set_toolset("cc", "aarch64-linux-gnu-gcc")
--     set_toolset("cxx", "aarch64-linux-gnu-g++")
--     set_toolset("ld", "aarch64-linux-gnu-g++")
--     set_toolset("sh", "aarch64-linux-gnu-g++")
-- toolchain_end()

if is_mode "release" then
    --set_symbols "hidden"
    --set_optimize "fastest"
    --set_runtimes("MT")
    --调试时打开下面两个
    set_optimize "none"
    set_symbols("debug")
end


rule("rule_display")
     after_build(function (target)
     cprint("${green} BIUD TARGET: %s", target:targetfile())
    end)
rule_end()

add_requires("cpp-httplib")
--系统oepncv
--add_requires("opencv4",{system = true})
add_includedirs("3rdparty/lib/opencv2/include/")
add_linkdirs("3rdparty/lib/opencv2/")
add_links("opencv_world481")





--trt cudnn
add_includedirs("E:/3rdparty/TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8/TensorRT-8.6.1.6/include")
add_includedirs("E:/3rdparty/cudnn-windows-x86_64-8.9.7.29_cuda11-archive/cudnn-windows-x86_64-8.9.7.29_cuda11-archive/include")
add_linkdirs("E:/3rdparty/TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8/TensorRT-8.6.1.6/lib")
add_linkdirs("E:/3rdparty/cudnn-windows-x86_64-8.9.7.29_cuda11-archive/cudnn-windows-x86_64-8.9.7.29_cuda11-archive/lib/x64")
add_links("nvinfer",
"nvinfer_plugin",
"nvparsers",
"nvonnxparser",
"cudnn",
"cudnn_adv_infer",
"cudnn_adv_train",
"cudnn_cnn_infer",
"cudnn_cnn_train",
"cudnn_ops_infer",
"cudnn_ops_train")

target("yuanba_meter")
    set_kind("binary")
    add_packages("cpp-httplib")
    add_rules("rule_display")
    add_rules("cuda")
    add_cuflags("-allow-unsupported-compiler", {force = true})
    add_cugencodes("native")
    add_ldflags("/NODEFAULTLIB:LIBCMT", {force = true})

    --源文件
    add_includedirs("HQ_AI_Model/HQ_AI_Model/")



    --tensorRTPro common
    add_includedirs("HQ_AI_Model/HQ_AI_Model/tensorRTPro/tensorRT/")
    add_includedirs("HQ_AI_Model/HQ_AI_Model/tensorRTPro/tensorRT/common/")
    add_headerfiles("HQ_AI_Model/HQ_AI_Model/tensorRTPro/tensorRT/common/**.hpp")
    add_headerfiles("HQ_AI_Model/HQ_AI_Model/tensorRTPro/tensorRT/common/**.cuh")
    add_files("HQ_AI_Model/HQ_AI_Model/tensorRTPro/tensorRT/common/**.cpp")
    add_files("HQ_AI_Model/HQ_AI_Model/tensorRTPro/tensorRT/common/**.cu")

    --tensorRTPro infer
    add_includedirs("HQ_AI_Model/HQ_AI_Model/tensorRTPro/tensorRT/infer/")
    add_headerfiles("HQ_AI_Model/HQ_AI_Model/tensorRTPro/tensorRT/infer/**.hpp")
    add_files("HQ_AI_Model/HQ_AI_Model/tensorRTPro/tensorRT/infer/**.cpp")
    add_files("HQ_AI_Model/HQ_AI_Model/tensorRTPro/tensorRT/import_lib.cpp")

    --common
    add_includedirs("HQ_AI_Model/HQ_AI_Model/tensorRTPro/application/")
    add_includedirs("HQ_AI_Model/HQ_AI_Model/tensorRTPro/application/common/")
    add_headerfiles("HQ_AI_Model/HQ_AI_Model/tensorRTPro/application/common/**.hpp")

    --deimv2
    add_includedirs("HQ_AI_Model/HQ_AI_Model/tensorRTPro/application/app_deimv2/")
    add_headerfiles("HQ_AI_Model/HQ_AI_Model/tensorRTPro/application/app_deimv2/**.hpp")
    add_files("HQ_AI_Model/HQ_AI_Model/tensorRTPro/application/app_deimv2/**.cpp")
    --yolocls
    add_includedirs("HQ_AI_Model/HQ_AI_Model/tensorRTPro/application/app_yolo_cls/")
    add_headerfiles("HQ_AI_Model/HQ_AI_Model/tensorRTPro/application/app_yolo_cls/**.hpp")
    add_files("HQ_AI_Model/HQ_AI_Model/tensorRTPro/application/app_yolo_cls/**.cpp")
    --yolo
    add_includedirs("HQ_AI_Model/HQ_AI_Model/tensorRTPro/application/app_yolo/")
    add_headerfiles("HQ_AI_Model/HQ_AI_Model/tensorRTPro/application/app_yolo/**.hpp")
    add_files("HQ_AI_Model/HQ_AI_Model/tensorRTPro/application/app_yolo/**.cpp")
    add_files("HQ_AI_Model/HQ_AI_Model/tensorRTPro/application/app_yolo/**.cu")
    --yoloobb
    add_includedirs("HQ_AI_Model/HQ_AI_Model/tensorRTPro/application/app_yolo_obb/")
    add_headerfiles("HQ_AI_Model/HQ_AI_Model/tensorRTPro/application/app_yolo_obb/**.hpp")
    add_files("HQ_AI_Model/HQ_AI_Model/tensorRTPro/application/app_yolo_obb/**.cpp")
    add_files("HQ_AI_Model/HQ_AI_Model/tensorRTPro/application/app_yolo_obb/**.cu")
    --yolopose
    add_includedirs("HQ_AI_Model/HQ_AI_Model/tensorRTPro/application/app_yolo_pose/")
    add_headerfiles("HQ_AI_Model/HQ_AI_Model/tensorRTPro/application/app_yolo_pose/**.hpp")
    add_files("HQ_AI_Model/HQ_AI_Model/tensorRTPro/application/app_yolo_pose/**.cpp")
    add_files("HQ_AI_Model/HQ_AI_Model/tensorRTPro/application/app_yolo_pose/**.cu")
    --yoloseg
    add_includedirs("HQ_AI_Model/HQ_AI_Model/tensorRTPro/application/app_yolo_seg/")
    add_headerfiles("HQ_AI_Model/HQ_AI_Model/tensorRTPro/application/app_yolo_seg/**.hpp")
    add_files("HQ_AI_Model/HQ_AI_Model/tensorRTPro/application/app_yolo_seg/**.cpp")
    add_files("HQ_AI_Model/HQ_AI_Model/tensorRTPro/application/app_yolo_seg/**.cu")


    --ocr ppocr
    add_includedirs("HQ_AI_Model/HQ_AI_Model/tensorRTPro/application/app_ppocr/")
    add_headerfiles("HQ_AI_Model/HQ_AI_Model/tensorRTPro/application/app_ppocr/**.hpp")
    add_files("HQ_AI_Model/HQ_AI_Model/tensorRTPro/application/app_ppocr/**.cpp")


    add_includedirs("HQ_AI_Model/HQ_AI_Model/tensorRTPro/application/cuosd/")
    add_includedirs("HQ_AI_Model/HQ_AI_Model/tensorRTPro/application/cuosd/textbackend/")
    add_headerfiles("HQ_AI_Model/HQ_AI_Model/tensorRTPro/application/cuosd/**.hpp")
    add_headerfiles("HQ_AI_Model/HQ_AI_Model/tensorRTPro/application/cuosd/**.h")
    add_files("HQ_AI_Model/HQ_AI_Model/tensorRTPro/application/cuosd/**.cpp")
    add_files("HQ_AI_Model/HQ_AI_Model/tensorRTPro/application/cuosd/**.cu")


    add_files("src/test.cpp")
    add_files("src/gauge_mask.cpp")
    add_headerfiles("src/gauge_mask.hpp")








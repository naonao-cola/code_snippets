set_project("tvt")
set_version("1.0.1")

--vsStudio 设置
add_rules("plugin.vsxmake.autoupdate")
set_encodings("source:utf-8")

-- cxx设置
set_languages("c++17")
add_cxxflags("-Wall")
set_runtimes("MT")
set_exceptions("cxx", "objc")

--编译模式
add_rules("mode.release","mode.releasedbg")

--opencv 依赖库
if is_plat("windows") then
	add_syslinks("opengl32")
	add_syslinks("gdi32")
	add_syslinks("advapi32")
	add_syslinks("glu32")
	add_syslinks("ws2_32")
	add_syslinks("user32")
	add_syslinks("comdlg32")
end

--三方库
add_requires("openmp")
--本地三方库
add_includedirs("E:/demo/3rdparty/algo_3rdparty/tv_algo_base_3rdparty/opencv4.5.3/include")
add_linkdirs("E:/demo/3rdparty/algo_3rdparty/tv_algo_base_3rdparty/opencv4.5.3/x64/vc16/staticlib")
add_links("ade",
"libjpeg-turbo",
"libpng",
"libprotobuf",
"libtiff",
"libwebp",
"opencv_aruco453",
"opencv_barcode453",
"opencv_bgsegm453",
"opencv_bioinspired453",
"opencv_calib3d453",
"opencv_ccalib453",
"opencv_core453",
"opencv_datasets453",
"opencv_dnn453",
"opencv_dnn_objdetect453",
"opencv_dnn_superres453",
"opencv_dpm453",
"opencv_face453",
"opencv_features2d453",
"opencv_flann453",
"opencv_fuzzy453",
"opencv_gapi453",
"opencv_hfs453",
"opencv_highgui453",
"opencv_imgcodecs453",
"opencv_imgproc453",
"opencv_img_hash453",
"opencv_intensity_transform453",
"opencv_line_descriptor453",
"opencv_mcc453",
"opencv_ml453",
"opencv_objdetect453",
"opencv_optflow453",
"opencv_phase_unwrapping453",
"opencv_photo453",
"opencv_plot453",
"opencv_quality453",
"opencv_rapid453",
"opencv_reg453",
"opencv_rgbd453",
"opencv_saliency453",
"opencv_shape453",
"opencv_stereo453",
"opencv_stitching453",
"opencv_structured_light453",
"opencv_superres453",
"opencv_surface_matching453",
"opencv_text453",
"opencv_tracking453",
"opencv_video453",
"opencv_videoio453",
"opencv_videostab453",
"opencv_wechat_qrcode453",
"opencv_xfeatures2d453",
"opencv_ximgproc453",
"opencv_xobjdetect453",
"opencv_xphoto453",
"quirc",
"zlib")

add_includedirs("E:/demo/3rdparty/algo_3rdparty/tv_algo_base_3rdparty/fmt/include")
add_linkdirs("E:/demo/3rdparty/algo_3rdparty/tv_algo_base_3rdparty/fmt/lib")
add_links("fmt","fmtd")

add_includedirs("./3rdparty/ai_inference/include")
add_linkdirs("./3rdparty/ai_inference/lib")
add_links("AIFramework")
add_rpathdirs("./3rdparty/ai_inference/lib")

add_includedirs("E:/demo/3rdparty/algo_3rdparty/tv_algo_base_3rdparty/tival_utility/include")
add_linkdirs("E:/demo/3rdparty/algo_3rdparty/tv_algo_base_3rdparty/tival_utility/x64/release")
add_links("tival_utility")
add_rpathdirs("E:/demo/3rdparty/algo_3rdparty/tv_algo_base_3rdparty/tival_utility/x64/release")

add_includedirs("E:/demo/3rdparty/algo_3rdparty/tv_algo_base_3rdparty/nlohmann_json/include")

add_includedirs("E:/demo/3rdparty/algo_3rdparty/tv_algo_base_3rdparty/tv_license/include")
add_linkdirs("E:/demo/3rdparty/algo_3rdparty/tv_algo_base_3rdparty/tv_license/lib")
add_links("tv_license")

add_rpathdirs("E:/demo/3rdparty/TensorRT-8.6.1.6/lib")
add_rpathdirs("E:/demo/3rdparty/onnxruntime-win-x64-gpu-1.15.1/lib")

if is_mode "releasedbg" then
    -- set_symbols "hidden"
    -- set_optimize "fastest"
	set_runtimes("MT")
	--调试时打开下面两个
	set_optimize "none"
    set_symbols("debug")
end

--宏定义
add_defines("USE_AI_DETECT")
add_defines("EXPORT_API")

target("tv_algorithm")
	set_kind("shared")
    add_cxxflags("-fPIC",{force = true})
	add_packages("openmp")

    -- 添加 tv_algo_base 库
    add_headerfiles("modules/tv_algo_base/src/framework/*h")
    add_headerfiles("modules/tv_algo_base/src/utils/*h")
    add_headerfiles("modules/tv_algo_base/src/Interface.h")

    add_files("modules/tv_algo_base/src/framework/*cpp")
    add_files("modules/tv_algo_base/src/utils/*cpp")
    add_files("modules/tv_algo_base/src/Interface.cpp","modules/tv_algo_base/src/dllmain.cpp")

	--项目代码
    add_headerfiles("src/project/*h")
    add_files("src/project/*cpp")
    add_files("src/test/fs.cpp")
    local has_executed = false
    --添加宏定义无效，修改一部分代码
    before_build_file(function (target)
        for _, headerfile in ipairs(target:headerfiles()) do
            if headerfile:endswith("Defines.h")  and not has_executed then
                print(headerfile)
                io.gsub(headerfile, "#define USE_TIVAL   0", "#define USE_TIVAL   1")
                has_executed = true
            end
        end
    end)

target_end()


target("test_dll")
    set_kind("binary")
    --添加依赖动态库
	add_deps("tv_algorithm")
    --添加测试代码
    add_headerfiles("src/test/*.h")
	add_files("src/test/*.cpp")

target_end()





set_project("test")
set_version("0.0.1")
set_languages("c++14")
add_rules("mode.release")

--显示构建目标路径
rule("rule_display")
     after_build(function (target)
     cprint("${green}  BIUD TARGET: %s", target:targetfile())
    end)
rule_end()


set_plat("android")
set_arch("arm64-v8a")
set_config("ndk", "/home/naonao/demo/3rdparty/android-ndk-r17c")
set_config("ndk_sdkver", "26")
set_config("runtimes", "c++_shared")

--[[
OpenCVConfig.cmake 文件的路径
方式一，自动添加路径
add_requires("cmake::OpenCV", {alias = "opencv", system = true,configs = {envs = {CMAKE_PREFIX_PATH = "/home/naonao/demo/3rdparty/test/opencv410_android/sdk/native/jni"}}})
add_packages("opencv")
方式二，手动添加opencv 的路径
add_includedirs(
        "$(projectdir)",
        "/home/naonao/demo/3rdparty/test/opencv410_android/sdk/native/jni/include"
    )
    add_linkdirs("/home/naonao/demo/3rdparty/test/opencv410_android/sdk/native/libs/arm64-v8a")
    add_links("opencv_calib3d",
    "opencv_core",
    "opencv_dnn",
    "opencv_features2d",
    "opencv_flann",
    "opencv_gapi",
    "opencv_imgcodecs",
    "opencv_imgproc",
    "opencv_ml",
    "opencv_objdetect",
    "opencv_photo",
    "opencv_stitching",
    "opencv_video",
    "opencv_videoio")
--]]

add_requires("cmake::OpenCV", {alias = "opencv", system = true,configs = {envs = {CMAKE_PREFIX_PATH = "/home/naonao/demo/3rdparty/test/opencv410_android/sdk/native/jni"}}})

target("test02")
     set_kind("binary")
     add_packages("opencv")
     add_includedirs(
         "$(projectdir)"
     )
--     -- add_ldflags(
--     --     "--sysroot /home/naonao/demo/3rdparty/android-ndk-r17c/platforms/android-26/arch-arm64"
--     -- )
     add_files("src/test.cpp")
     add_files("src/main.cpp")
    


-- add_requires("jemalloc",{system = false,configs = {runtimes = "MT"}})

-- target("main_2")
--	set_kind("binary")
--	add_files("src/test_2.cpp")
--	add_packages("jemalloc")

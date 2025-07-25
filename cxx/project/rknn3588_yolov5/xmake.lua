
set_project("test")
set_version("0.0.1")
set_languages("c++14")

add_rules("mode.releasedbg", "mode.release")


--编译模式
if is_mode "releasedbg" then
    -- set_symbols "hidden"
    -- set_optimize "fastest"
	-- set_runtimes("MT")
	--调试时打开下面两个
	set_optimize "none"
    set_symbols("debug")
end


--工具链
toolchain("my_toolchain")
    set_kind("standalone")
    set_sdkdir("/home/naonao/demo/3rdparty/rknn_tools/aarch64-linux-android-gcc4.9.x/aarch64-linux-android")
    set_bindir("/home/naonao/demo/3rdparty/rknn_tools/aarch64-linux-android-gcc4.9.x/aarch64-linux-android/bin")
    set_toolset("cxx", "aarch64-linux-android-g++")
    --set_toolset("cxx", "clang++")
    --set_toolset("c", "aarch64-linux-android-gcc")
toolchain_end()


--三方库
target("toolchain_env")
    set_default(false)
    set_kind("phony")
    on_config(function (target)
        target:set("opencv_dir", "/home/naonao/demo/3rdparty/rknn_tools/third-party/")
        target:set("lib_dir", "/home/naonao/demo/cxx/test03/lib/")

    end)
target_end()


--显示构建目标路径
rule("rule_display")
     after_build(function (target)
     cprint("${green}  BIUD TARGET: %s", target:targetfile())
    end)
rule_end()


-- opencv库规则
rule("package_opencv")
    on_config(function (target)
        import("core.project.project")
        local tool_env = project.target("toolchain_env")
        target:add("includedirs",tool_env:get("opencv_dir"))
        target:add("linkdirs",tool_env:get("lib_dir"))
        target:add("links",
        "opencv_imgproc",
        "opencv_imgcodecs",
        "opencv_core"
        )
    end)
rule_end()


--其他的库
rule("package_other")
    on_config(function (target)
        import("core.project.project")
        local tool_env = project.target("toolchain_env")
        target:add("linkdirs",tool_env:get("lib_dir"))
        target:add("cxxflags","-Wl,--allow-shlib-undefined")
        target:add("links",
        "rga",
        "OpenCL"
        )
    end)
rule_end()


--rknn库
rule("package_rknn")
    on_config(function (target)
        import("core.project.project")
        local tool_env = project.target("toolchain_env")
        target:add("linkdirs",tool_env:get("lib_dir"))
        target:add("links",
        "rknnrt"
        )
    end)
rule_end()


target("test03")

    add_deps("toolchain_env")
    set_toolchains("my_toolchain")
    set_kind("binary")
    set_strip("none")

    --添加三方库
    --add_rules("package_opencv")
    --add_rules("package_rknn")
    add_rules("rule_display")
    add_rules("package_other")
    -- 添加编译文件
    add_files("src/main.cpp")
    add_files("src/NmsCl.cpp")
    add_files("src/utils.cpp")


    -- add_defines("CL_VERSION_1_1=1","CL_TARGET_OPENCL_VERSION=110")
    -- 依赖
    add_cxxflags(
        "-fPIE",
        "-std=c++11",
        "-g",
        "-DANDROID_STL=c++_static",
        "-D__ANDROID_API__=24",
        "-pthread",
        "-D CL_VERSION_1_1=1",
        "-D CL_TARGET_OPENCL_VERSION=110"
    )

    add_ldflags(
        "-Wl,--allow-shlib-undefined",
        "-pie"
        -- "-shared"
    )
    --添加显示头文件
    add_includedirs(
        "$(projectdir)",
        "3rdparty/libopencl-stub/include",
        -- "3rdparty/librga/include",
        -- "3rdparty/stb_image",
        -- "3rdparty/jpeg_turbo/include",
        "src/"
    )
    --依赖
    add_headerfiles("src/utils.h")
    add_headerfiles("src/NmsCl.h")

target_end()


target("test04")

    add_deps("toolchain_env")
    set_toolchains("my_toolchain")
    set_kind("binary")
    set_strip("none")

    --添加三方库
    add_rules("package_opencv")
    add_rules("package_rknn")
    add_rules("rule_display")
    add_rules("package_other")
    -- 添加编译文件
    add_files("src/*.cpp|main.cpp")

    -- 依赖
    add_cxxflags(
        "-fPIE",
        "-std=c++11",
        "-g",
        "-DANDROID_STL=c++_static",
        "-D__ANDROID_API__=24",
        "-pthread",
        "-D CL_VERSION_1_1=1",
        "-D CL_TARGET_OPENCL_VERSION=110"
    )

    add_ldflags(
        "-Wl,--allow-shlib-undefined",
        "-pie",
        "-fPIE"
    )
    --添加显示头文件
    add_includedirs(
        "$(projectdir)",
        "3rdparty/libopencl-stub/include",
        "3rdparty/librga/include",
        "3rdparty/stb_image",
        "3rdparty/jpeg_turbo/include",
        "src/"
    )
    --依赖
    add_headerfiles("src/*.h|utils.h")
    add_cxxflags("-fopenmp")
    add_ldflags("-fopenmp")

target_end()
set_project("dih_alg")
set_version("0.0.1")
set_languages("c++14")

add_rules("mode.release","mode.releasedbg")

--自动更新vs解决方案结构
add_rules("plugin.vsxmake.autoupdate")


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
    --set_toolset("g++")
    set_toolset("cxx", "aarch64-linux-android-g++")
toolchain_end()

--三方库
target("toolchain_env")
    set_default(false)
    set_kind("phony")
    on_config(function (target)
        target:set("opencv_dir", "/home/naonao/demo/3rdparty/rknn_tools/third-party/")
        target:set("pthread_dir", "/home/naonao/demo/3rdparty/rknn_tools/third-party/pthreads4w-3.0/")
        target:set("lib_dir", "/home/naonao/demo/cxx/old_new/DIH-ALG/app/link_lib/")
        target:set("lua_src_dir", "/home/naonao/demo/cxx/old_new/DIH-ALG/app/lua/")
    end)
target_end()


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

-- lua库
rule("package_lua")
    on_config(function (target)
        import("core.project.project")
        local tool_env = project.target("toolchain_env")
        target:add("linkdirs",tool_env:get("lib_dir"))
        target:add("links","lua")
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

--显示构建目标路径
rule("rule_display")
     after_build(function (target)
     cprint("${green}  BIUD TARGET: %s", target:targetfile())
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
        "immune",
        "DIHLog",
        "eventLib",
        "OpenCL"
        --"pthread"
        )
    end)
rule_end()


--目标
target("algLib")
    set_toolchains("my_toolchain")
    add_deps("toolchain_env")
    set_kind("shared")

    set_strip("none")

    --添加三方库
    add_rules("package_opencv")
    add_rules("package_rknn")
    add_rules("package_lua")
    add_rules("package_other")
    add_rules("rule_display")

    add_defines("CL_VERSION_1_1=1","CL_TARGET_OPENCL_VERSION=110")

    add_cxxflags(
        "-fPIC",  -- 编译位置无关代码
        "-g",  -- 调试信息
        "-DANDROID_STL=c++_static",-- Android STL 设置
        "-D__ANDROID_API__=24", -- Android API 版本
        "-fopenmp"
    )


    add_ldflags(
        "-Wl,--allow-shlib-undefined",
        "-shared",
        "-pthread",
        "-fopenmp"
    )
     --添加显示头文件
    add_includedirs(
        "$(projectdir)",
        "libalg",
        "libalg/include",
        "libalg/make_result",
        "libalg/model_config",
        "libalg/tinyxml2",
        "libalg/opencl_tools",
        "libalg/include/libopencl-stub/include",
        "libalg/include/nlohmann/"

    )
    add_headerfiles("libalg/*.h")
    add_headerfiles("libalg/opencl_tools/*.h")
    add_headerfiles("libalg/make_result/*.h")
    add_headerfiles("libalg/model_config/*.h")
    add_headerfiles("libalg/tinyxml2/*.h")
    add_headerfiles("libalg/include/nlohmann/*.hpp")

    -- 添加编译文件
    add_files("libalg/*.cpp")
    add_files("libalg/opencl_tools/*.cpp")
    add_files("libalg/make_result/*.cpp")
    add_files("libalg/model_config/*.cpp")
    add_files("libalg/tinyxml2/*.cpp")


target_end()


target("main2")
    add_deps("toolchain_env")
    add_deps("algLib")

    set_toolchains("my_toolchain")
    set_kind("binary")

    --添加三方库
    add_rules("package_opencv")
    -- add_rules("package_rknn")
    -- add_rules("package_lua")
    -- add_rules("package_other")
    add_rules("rule_display")

    -- 添加编译文件
    add_files("main_2.cpp")


    --依赖
    add_cxxflags(
        "-fPIE",
        "-std=c++11",
        "-g",
        "-DANDROID_STL=c++_static",
        "-D__ANDROID_API__=24",
        "-pthread",
        "-D CL_VERSION_1_1=1",
        "-D CL_TARGET_OPENCL_VERSION=110",
        "-fopenmp"
    )
    add_ldflags(
        "-Wl,--allow-shlib-undefined",
        "-pie",
        "-fopenmp"
    )
    --添加显示头文件
    add_includedirs(
        "$(projectdir)",
        "libalg",
        "libalg/include"
        -- "libalg/make_result",
        -- "libalg/model_config",
        -- "libalg/tinyxml2",
        -- "libalg/opencl_tools",
        -- "libalg/include/libopencl-stub/include",
        -- "local_test",
        -- "local_test/human",
        -- "local_test/local_xml_config"
    )

    --依赖
    -- add_headerfiles("libalg/opencl_tools/*.h")
    -- add_headerfiles("libalg/*.h")
    -- add_headerfiles("libalg/make_result/*.h")
    -- add_headerfiles("libalg/model_config/*.h")
    -- add_headerfiles("libalg/tinyxml2/*.h")
    add_linkdirs("./app/lib")
    add_links("eventLib")

target_end()


target("app_local_test")
    add_deps("toolchain_env")
    set_toolchains("my_toolchain")
    set_kind("binary")
    set_strip("none")

    --添加三方库
    add_rules("package_opencv")
    add_rules("package_rknn")
    add_rules("package_lua")
    add_rules("package_other")
    add_rules("rule_display")

    -- 添加编译文件
    add_files("main.cpp")
    add_files("./local_test/**.cpp")
    add_files("./libalg/*.cpp")
    add_files("./libalg/make_result/*.cpp")
    add_files("./libalg/model_config/*.cpp")
    add_files("./libalg/tinyxml2/*.cpp")
    add_files("./libalg/opencl_tools/*.cpp")


    --依赖
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
    )
    --添加显示头文件
    add_includedirs(
        "$(projectdir)",
        "libalg",
        "libalg/include",
        "libalg/make_result",
        "libalg/model_config",
        "libalg/tinyxml2",
        "libalg/opencl_tools",
        "libalg/include/libopencl-stub/include",
        "local_test",
        "local_test/human",
        "local_test/local_xml_config"
    )

    --依赖
    add_headerfiles("libalg/opencl_tools/*.h")
    add_headerfiles("libalg/*.h")
    add_headerfiles("libalg/make_result/*.h")
    add_headerfiles("libalg/model_config/*.h")
    add_headerfiles("libalg/tinyxml2/*.h")
    add_linkdirs("./app/lib")
    add_links("eventLib")

target_end()

-- includes("@builtin/xpack")
-- xpack("libalgLib")
--     --set_strip("none")
--     set_version("0.0.2")
--     set_formats("zip")
--     set_license("Apache-2.0")
--     add_targets("algLib")
--     set_description("pdw 写入单个明场图的csv.")

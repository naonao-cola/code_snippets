set_project("tvt")
set_version("1.0.0")
set_languages("c++17")
add_rules("mode.debug", "mode.release")

-- 指定三方库目录
local thirdparty = "D:/env/algo_3rdparty/"

-- 宏定义
add_defines("USE_AI_DETECT")

if is_plat("windows") then
	add_syslinks("opengl32")
	add_syslinks("gdi32")
	add_syslinks("advapi32")
	add_syslinks("glu32")
	add_syslinks("ws2_32")
	add_syslinks("user32")
	add_syslinks("comdlg32")
end


if is_mode "debug" then
    add_defines("DEBUG")
    set_symbols "debug"
    set_optimize "none"
    set_runtimes("MT")
end

if is_mode "release" then
    --set_symbols "hidden"
    --set_optimize "fastest"
	set_runtimes("MT")
	--调试时打开下面两个
	set_optimize "none"
    set_symbols("debug")
end

add_includedirs(thirdparty.."/fmt/include")
add_linkdirs(thirdparty.."/fmt/lib")
add_links("fmt","fmtd")

add_includedirs(thirdparty.."/ai_inference/include")
add_linkdirs(thirdparty.."/ai_inference/lib")
add_links("AIFramework")

add_includedirs(thirdparty.."/tival_utility/include")
add_linkdirs(thirdparty.."/tival_utility/x64/release")
add_links("tival_utility")

add_includedirs(thirdparty.."/nlohmann_json/include")

add_includedirs(thirdparty.."/opencv490/include")
add_linkdirs(thirdparty.."/opencv490/x64/vc16/lib")
add_links("opencv_world490")

add_includedirs(thirdparty.."/tv_license/include")
add_linkdirs(thirdparty.."/tv_license/lib")
add_links("tv_license")

-- add_includedirs(thirdparty.."/tvcore/include")
-- add_linkdirs(thirdparty.."/tvcore/lib")
-- add_links("libzbar", "tvcore")

--自动更新vs解决方案结构
add_rules("plugin.vsxmake.autoupdate")

-- 算法dll
target("tv_algorithm")
	set_kind("shared")
    add_headerfiles("src/framework/*h")
	add_files("src/framework/*cpp")
    add_headerfiles("src/custom/*h")
	add_files("src/custom/*cpp")
	add_headerfiles("src/example/*h")
	add_files("src/example/*cpp")
    add_headerfiles("src/utils/*h")
	add_files("src/utils/*cpp")
    add_headerfiles("src/*.h")
	add_files("src/*.cpp")
	--add_defines("DEBUG_ON")
    add_defines("EXPORT_API")

-- 调用算法dll测试程序
target("test_dll")
    set_kind("binary")
	add_deps("tv_algorithm")
    add_defines("EXPORT_API")
	add_files("test/main.cpp")

-- 整体编译exe测试程序
target("test_exe")
	set_kind("binary")
    add_headerfiles("src/framework/*h")
	add_files("src/framework/*cpp")
    add_headerfiles("src/example/*h")
	add_files("src/example/*cpp")
    add_headerfiles("src/utils/*h")
	add_files("src/utils/*cpp")
    add_headerfiles("src/*.h")
	add_files("src/*.cpp")

	add_includedirs(thirdparty.."/co/include")
	add_linkdirs(thirdparty.."/co/lib")
	add_links("co")

    add_files("test/*.cpp|main.cpp")


set_project("imageprocessor")
set_version("0.0.1",{soname = true})
set_languages("c++17")
set_toolchains("msvc", {vs = "2022"})
--set_policy("build.c++.msvc.runtime", "MD")
add_rules("mode.debug", "mode.release", "mode.releasedbg")
add_requires("opencv 4.8.x", {system = false})


set_policy("build.progress_style", "multirow")
add_rules("plugin.vsxmake.autoupdate")


if is_mode "releasedbg" then
    -- set_symbols "hidden"
    -- set_optimize "fastest"
	-- set_runtimes("MT")
	--调试时打开下面两个
	set_optimize "none"
    set_symbols("debug")
end

add_includedirs("E:/3rdparty/jdk-26_windows-x64_bin/jdk-26.0.1/include")
add_includedirs("E:/3rdparty/jdk-26_windows-x64_bin/jdk-26.0.1/include/win32")


target("imageprocessor")
    set_kind("shared")
    add_packages("opencv")

    add_includedirs("./")
    add_headerfiles("./*.h")
    add_files("./*.cpp")




set_project("tvt")
set_version("1.0.0")
set_languages("c++17")
add_rules("mode.debug", "mode.release","mode.releasedbg")


add_requires("lua")
--自动更新vs解决方案结构
add_rules("plugin.vsxmake.autoupdate")
set_policy("check.auto_ignore_flags", false)





-- 调用算法dll测试程序
target("test")
    set_kind("binary")
	add_files("src/main.cpp")
    add_files("src/task.cpp")
    add_ldflags("-luuid")



target("test_xml")
    set_kind("binary")
	add_files("src/test_xml.cpp")
    add_files("src/tinyxml2.cpp")




target("test_lua")
    set_kind("binary")
    add_packages("lua")
    add_rules("c++")
    add_files("lua_test/*cpp")





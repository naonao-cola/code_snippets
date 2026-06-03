

set_version("0.0.1")
set_languages("c++17")
add_rules("mode.debug", "mode.release","mode.releasedbg")
add_requires("opencv 4.8.x",{system = false})




target("test_project")
    add_packages("opencv")
    -- 设置编译路径
    --set_targetdir("$(projectdir)/libs")
    -- 添加文件
    add_files("$(projectdir)/src/**.cpp")

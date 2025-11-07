

set_version("0.0.1")
set_languages("c++17")
add_rules("mode.debug", "mode.release","mode.releasedbg")
add_requires("opencv 4.8.x",{system = false})


-- 添加依赖
--add_requires("python 3.10")
add_requires("pybind11")
--add_requireconfs("pybind11.python",{version = "3.10",override=true})

target("example")
    add_rules("python.module")
    -- 添加依赖
    add_packages("pybind11")
    add_packages("opencv")
    -- 设置编译路径
    --set_targetdir("$(projectdir)/libs")
    -- 添加文件
    add_files("$(projectdir)/src/example.cpp")
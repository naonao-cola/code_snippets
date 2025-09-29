set_project("tvt")
set_version("1.0.1")

--vsStudio 设置
add_rules("plugin.vsxmake.autoupdate")
set_encodings("source:utf-8")
add_rules("mode.release","mode.releasedbg")

-- cxx设置
set_languages("c++17")
add_cxxflags("-Wall")
set_runtimes("MT")

add_includedirs("E:/Env/TensorRT-8.4.1.5/include")
add_linkdirs("E:/Env/TensorRT-8.4.1.5/lib")
add_links("nvinfer","nvinfer_plugin","nvonnxparser","nvparsers")
add_rpathdirs("E:/Env/TensorRT-8.4.1.5/lib")

target("custom-plugin")
	set_kind("shared")
	--项目代码
    add_rules("c++")
    add_headerfiles("src/*hpp")
    add_files("src/*cpp")
    add_files("src/*.cu")
target_end()
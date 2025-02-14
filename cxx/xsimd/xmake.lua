set_project("xsimd_test")

set_version("0.0.1")
set_languages("c++17")
add_rules("mode.debug", "mode.release","mode.releasedbg")
add_vectorexts("avx", "avx2")
add_vectorexts("sse", "sse2", "sse3", "ssse3", "sse4.2")

add_requires("xsimd")
add_requires("openmp")
if is_plat("windows") then
    add_cxxflags("/arch:AVX2")
else
    add_cxxflags("-march=native")
end


--自动更新vs解决方案结构
add_rules("plugin.vsxmake.autoupdate")
set_encodings("source:utf-8")



target("01")
	add_packages("openmp")
	add_packages("xsimd")
	set_kind("binary")
    add_headerfiles("src/**.h")
	add_files("src/**.cpp")
    add_cxxflags("/openmp")
    add_ldflags("-lopenmp")






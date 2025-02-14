
add_requires("myopencv")

target("test_01")
    set_kind("binary")
    add_files("./*.cpp")
    add_packages("myopencv")

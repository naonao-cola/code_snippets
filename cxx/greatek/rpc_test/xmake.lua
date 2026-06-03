add_rules("mode.debug", "mode.release")
add_requires("brpc")
set_languages("c++17")


target("rpc_test")
    set_kind("binary")
    add_packages("brpc")
    add_files("src/*.cpp")
    add_files("src/echo.proto")


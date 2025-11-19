

set_version("1.0.0")
set_languages("c++17")
add_rules("mode.debug", "mode.release")
set_warnings("all", "error")

add_requires("brpc")

target("rpc_server")
    set_kind("binary")

    add_packages("brpc")
    add_files("src/*.cpp|echo_client.cpp")
    add_files("src/*.cc")




target("rpc_client")
    set_kind("binary")

    add_packages("brpc")
    add_files("src/*.cpp|echo_server.cpp")
    add_files("src/*.cc")

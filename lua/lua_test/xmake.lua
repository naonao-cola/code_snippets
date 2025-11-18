add_rules("mode.debug", "mode.release")
add_requires("sol2")
set_languages("c++17")


target("lua_test")
    set_kind("binary")
    add_packages("sol2")
    add_files("src/*.cpp")


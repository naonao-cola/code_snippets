set_project("test_package")
-- version
set_version("2.7.0")
set_languages("c++17")

-- set warning all as error
set_warnings("all")

-- set language: c99, c++11


-- add rules: debug/release
add_rules("mode.debug", "mode.release")

-- for the windows platform (msvc)
if is_plat("windows") then
    -- link libcmt.lib
    add_cxflags("-MT")

end

-- add syslinks
if is_plat("windows") then
    add_syslinks("opengl32")
	add_syslinks("gdi32")
	add_syslinks("advapi32")
	add_syslinks("glu32")
	add_syslinks("ws2_32")
	add_syslinks("user32")
	add_syslinks("comdlg32")
elseif is_plat("android") then
    add_syslinks("m", "c")
elseif is_plat("mingw") then
    add_syslinks("ws2_32", "pthread", "m")
else
    add_syslinks("pthread", "dl", "m", "c")
end

-- add requires
--add_requires("tbox", {configs = {coroutine = true}})

-- add repositories
add_repositories("myrepo repo")

-- include sources
includes("src")
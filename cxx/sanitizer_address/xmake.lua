add_rules("mode.debug", "mode.release")



set_policy("build.sanitizer.address", true)
--set_policy("build.sanitizer.thread", true)
--set_policy("build.sanitizer.memory", true)  --clang only
set_policy("build.sanitizer.leak", true)
set_policy("build.sanitizer.undefined", true)
set_policy("run.autobuild", true)


target("scan_test")
    set_kind("binary")
    add_files("src/*.cpp")



set_project("ocr_llama_demo")
set_version("0.0.1")
set_languages("c++17")

add_rules("mode.debug", "mode.release")
set_policy("build.progress_style", "multirow")

local llama_dir = path.join(os.projectdir(), "sub_repo", "llama.cpp")
local llama_bin_dir = path.join(llama_dir, "build", "bin")

target("ocr_demo")
    set_kind("binary")
    -- 只编译本项目源码（src/*.cpp），llama.cpp 本身不由 xmake 编译；
    -- llama.cpp 使用你已经用 CMake 生成的 build/bin/*.so。
    add_files("src/demo.cpp", "src/test.cpp")
    add_includedirs(
        "src",
        -- llama.cpp 的 public 头文件
        path.join(llama_dir, "include"),
        -- llama.cpp 的 common（部分工具/示例会包含，但这里主要用于兼容 include 路径）
        path.join(llama_dir, "common"),
        -- mtmd 头文件（多模态 C API）
        path.join(llama_dir, "tools", "mtmd"),
        -- ggml 头文件（依赖类型）
        path.join(llama_dir, "ggml", "include")
    )
    -- 链接 llama.cpp 的共享库（来自 CMake build/bin）
    add_linkdirs(llama_bin_dir)
    add_links("mtmd", "llama", "llama-common", "ggml", "ggml-base", "ggml-cpu", "ggml-cuda")
    add_syslinks("pthread", "dl", "m")
    -- 运行时自动找到 build/bin 里的 *.so（避免手动设置 LD_LIBRARY_PATH）
    add_rpathdirs(llama_bin_dir)
    add_rpathdirs(path.join(llama_bin_dir))

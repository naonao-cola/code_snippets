
set_project("pcl_test")
set_version("0.0.1",{soname = true})
set_languages("c++17")
set_toolchains("msvc")
--set_policy("build.c++.msvc.runtime", "MD")
add_rules("mode.debug", "mode.release", "mode.releasedbg")
add_requires("opencv 4.8.x", {system = false,configs = {shared = true}})
set_config("cuda_sdkver","11.8")

--add_requires("onnxruntime")
add_requires("pcl",{configs = {shared = true,vtk = true, visualization = true}})
add_requireconfs("pcl.eigen", {override = true, version = "3.3.7"})

add_requires("nlohmann_json",{system = false})
set_policy("build.progress_style", "multirow")
add_rules("plugin.vsxmake.autoupdate")


if is_mode "releasedbg" then
    -- set_symbols "hidden"
    -- set_optimize "fastest"
	-- set_runtimes("MT")
	--调试时打开下面两个
	set_optimize "none"
    set_symbols("debug")
end

--trt cudnn
-- add_includedirs("E:/3rdparty/cudnn-windows-x86_64-8.9.7.29_cuda11-archive/cudnn-windows-x86_64-8.9.7.29_cuda11-archive/include/")
-- add_linkdirs("E:/3rdparty/cudnn-windows-x86_64-8.9.7.29_cuda11-archive/cudnn-windows-x86_64-8.9.7.29_cuda11-archive/lib/x64/")
-- add_includedirs("E:/3rdparty/TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8/TensorRT-8.6.1.6/include/")
-- add_linkdirs("E:/3rdparty/TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8/TensorRT-8.6.1.6/lib/")
-- add_rpathdirs("E:/3rdparty/TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8/TensorRT-8.6.1.6/lib")
-- add_rpathdirs("E:/3rdparty/cudnn-windows-x86_64-8.9.7.29_cuda11-archive/cudnn-windows-x86_64-8.9.7.29_cuda11-archive/bin/")
-- add_links("nvinfer",
-- "nvinfer_plugin",
-- "nvparsers",
-- "nvinfer_vc_plugin",
-- "nvonnxparser",
-- "cudnn",
-- "cudnn_adv_infer",
-- "cudnn_adv_train",
-- "cudnn_cnn_infer",
-- "cudnn_cnn_train",
-- "cudnn_ops_infer",
-- "cudnn_ops_train")


--onnxruntime
add_includedirs("E:/3rdparty/onnxruntime-win-x64-1.22.1/onnxruntime-win-x64-1.22.1/include")
add_linkdirs("E:/3rdparty/onnxruntime-win-x64-1.22.1/onnxruntime-win-x64-1.22.1/lib")
add_rpathdirs("E:/3rdparty/onnxruntime-win-x64-1.22.1/onnxruntime-win-x64-1.22.1/lib")
add_links("onnxruntime","onnxruntime_providers_shared")

--sensor
add_linkdirs("E:/test/pcl_test/3rdparty/sensor")
add_links("Ldsensorllib")
add_rpathdirs("E:/test/pcl_test/3rdparty/sensor")

target("pcl_test")
    set_version("1.0.2")
    set_toolchains("msvc")
    set_kind("binary")
    add_packages("opencv")
    add_packages("pcl")
    add_rules("cuda")
    add_cugencodes("native")
    add_cuflags("-allow-unsupported-compiler", {force = true})
    --set_policy("build.c++.msvc.runtime", "MD")
    --add_packages("onnxruntime")
    --add_rules("cuda")
    add_includedirs("include")
    add_headerfiles("include/*.h")
    --add_files("src/*.cpp|win_dll_demo.cpp|test.cpp|test_1.cpp|test_2.cpp|test_3.cpp|test_4.cpp|test_5.cpp|test_6.cpp") -- 添加 .cpp 文件编译
    add_files("src/*.cu","src/ox_d.cpp","src/ox_seg.cpp","src/ox.cpp")-- 添加 .cu 文件编译
    add_files("src/test_5.cpp")



    set_configdir(".")
    -- add_configfiles("version.rc.in")
    -- if is_plat("windows") then
    --     add_files("version.rc")
    -- end
    after_install(function (target)
        local installdir = target:installdir()
        local seen = {}
        local function copy_pkg(pkg)
            if seen[pkg:name()] then return end
            seen[pkg:name()] = true
            local root = pkg:installdir()
            if root and os.isdir(root) then
                os.vcp(path.join(root, "**.dll"), path.join(installdir, "bin"))
                os.vcp(path.join(root, "**.lib"), path.join(installdir, "lib"))
                if os.isdir(path.join(root, "include")) then
                    os.vcp(path.join(root, "include/**"), path.join(installdir, "include"))
                end
            end
            for _, dep in ipairs(pkg:deps() or {}) do
                copy_pkg(dep)
            end
        end
        for _, pkg in ipairs(target:pkgs() or {}) do
            copy_pkg(pkg)
        end
    end)





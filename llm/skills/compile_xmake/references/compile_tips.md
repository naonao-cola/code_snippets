# 编译案例与技巧

## 1. CUDA 构建优化
### 静态库中的 Device Link
如果目标是静态库且包含 CUDA 代码，需要设置 `devlink` 策略：
```lua
if is_kind("static") then
    set_policy("build.cuda.devlink", true)
end
```

### 禁用不支持的编译器检查
```lua
add_cuflags("-allow-unsupported-compiler")
```

## 2. 编译模式配置
### 定义 releasedbg
```lua
add_rules("mode.debug", "mode.release", "mode.releasedbg")

if is_mode("releasedbg") then
    set_optimize("none")
    set_symbols("debug")
    set_runtimes("MT")
end
```

## 3. 编译加速优化 (Compilation Speedup)
### 预编译头 (Precompiled Headers)
对于大型项目（如包含 PCL/OpenCV），启用 PCH 可减少 50% 以上的编译时间：
```lua
target("large_project")
    set_pcxxheader("src/pch.h")
    add_files("src/**.cpp")
```

### 编译缓存 (ccache)
```lua
set_policy("build.ccache", true)
```

## 4. 单元测试 (Unit Testing)
xmake 内置了强大的测试支持，可配合 `xrepo` 自动集成测试框架：
```lua
add_requires("gtest")

target("test_suite")
    set_kind("binary")
    add_files("tests/*.cpp")
    add_packages("gtest")
    add_rules("utils.bin2c", {extensions = {".png", ".jpg"}}) -- 示例：将资源转为代码

    -- 注册到 xmake test
    add_tests("test_main")
```
运行命令：`xmake test`

## 5. 跨平台与交叉编译 (Cross-compilation)
### Android 编译
```bash
xmake f -p android --ndk=/path/to/ndk -a arm64-v8a
xmake
```

### 通用交叉编译 (Embedded)
```bash
xmake f -p cross --sdk=/path/to/sdk --cross=arm-linux-gnueabihf-
```

## 6. 配置备份与恢复
```bash
xmake f --save=my_config  # 备份当前配置
xmake f --load=my_config  # 恢复备份的配置
```

## 7. 动态工具检测与 CUDA 版本管理

### 动态检测 MSVC 工具集版本
在 Windows 环境下，当遇到 CUDA 与 MSVC 版本不兼容（如 STL1002 错误）时，可以使用以下逻辑探测并降级：

1. **探测可用工具集**：
   查看 `C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC` 下的文件夹名称。
2. **执行降级配置**：
   ```bash
   # 强制指定旧版工具集（如 14.29）
   xmake f -c --vs=2022 --vs_toolset=14.29.30133
   ```
3. **xmake.lua 中固定版本**：
   ```lua
   set_toolchains("msvc", {vs = "2022", vs_toolset = "14.29.30133"})
   ```

### 多 CUDA 版本检测与切换
如果系统中安装了多个 CUDA 版本，可以通过以下方式管理：

1. **查看当前识别的 CUDA**：
   ```bash
   xmake l detect.sdks.find_cuda
   ```
2. **手动指定 CUDA 路径**：
   ```bash
   # 切换到 CUDA 11.8
   xmake f --cuda="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"

   # 或者全局设置
   xmake g --cuda="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"
   ```
3. **探测环境变量**：
   检查 `CUDA_PATH`, `CUDA_PATH_V11_8`, `CUDA_PATH_V12_1` 等环境变量是否指向正确路径。

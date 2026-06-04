---
name: compile_xmake
description: 处理 xmake 编译相关的问题，包括 CUDA/MSVC 兼容性、本地库加载、编译模式配置（如 releasedbg）等。当用户遇到 xmake 编译报错、环境配置失败或需要特定的编译示例时触发。
---

# compile_xmake

本技能用于解决 xmake 编译过程中的各种疑难杂症，特别是针对 Windows 环境下的 CUDA、MSVC 兼容性问题，以及本地依赖库（如 OpenCV）的配置。

## 核心能力

1. **CUDA/MSVC 兼容性修复**：解决 CUDA 11.8 与过新 MSVC (如 14.51) 不兼容导致的 `STL1002` 错误。
2. **多 CUDA 版本选择**：支持探测并切换系统安装的多个 CUDA 版本。
3. **自动库检索与配置**：自动扫描本地目录（如 TensorRT、OpenCV）并配置 `include`、`lib`、`links` 及 `rpath`。
4. **编译模式配置**：配置 `releasedbg` 等自定义模式，平衡性能与调试需求。
5. **CI/CD 与插件扩展**：提供 GitHub Actions 模板以及自定义插件（Task）开发指导，实现构建流程的全自动化。

## 闭环思维与深度诊断 (P10 Methodology)

当遇到未知的编译错误时，遵循以下深度搜索与闭环路径：

1. **一键清洁验证**：排除缓存干扰。
   - `xmake f -c && xmake g --clean`
2. **符号与依赖溯源**：
   - 使用 `xmake show -t <target>` 查看完整的 `includedirs` 和 `links` 是否包含预期的路径。
   - 在 Windows 上使用 `dumpbin /dependents <exe>` 或在 Linux 上使用 `ldd <exe>` 确认运行时链接是否正确。
3. **Verbose 模式拆解**：
   - 运行 `xmake -vD` 获取完整的命令行参数。
   - 将报错的那个 `nvcc` 或 `cl.exe` 命令提取出来，手动在终端运行，观察是否依然报错，从而隔离是 xmake 配置问题还是编译器本身问题。
4. **版本边界探测**：
   - 如果是工具链问题（如 MSVC），尝试向上/向下跳一个大版本进行交叉对比。

## 常见问题与解决 (Troubleshooting)

### 1. CUDA 11.8 + MSVC 14.5x 冲突 (STL1002)
**症状**：编译 `.cu` 文件报错 `error STL1002: Unexpected compiler version, expected CUDA 13.2 or newer.`
**根因**：`nvcc` 默认使用了系统中最新的 MSVC 头文件，而 CUDA 11.8 不支持新版 STL。
**修复逻辑**：
1. **优先自动探测**：尝试运行 `xmake f -c`。如果编译报错 `STL1002`，则说明当前默认工具集过新。
2. **动态版本查询**：使用 `xmake show -l toolchains` 或查看 `C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC` 目录，列出本地已安装的所有 MSVC 工具集版本。
3. **询问与降级**：
   - 告知用户当前版本不兼容。
   - 列出探测到的旧版工具集（如 `14.29.xxxx`）。
   - 询问用户：“检测到当前 MSVC 版本与 CUDA 11.8 不兼容，是否尝试降级到已安装的 [版本号] 并重新配置？”
4. **执行配置**：根据用户确认的版本，执行：
  ```bash
  xmake f -c --vs=2022 --vs_toolset=14.29.xxxxx -m releasedbg
  ```
- 并在 `xmake.lua` 中同步更新：
  ```lua
  set_toolchains("msvc", {vs = "2022", vs_toolset = "14.29.xxxxx"})
  ```

### 2. 多 CUDA 版本管理与切换
**流程**：
1. **探测已安装版本**：
   - **Windows**: 检查环境变量 `CUDA_PATH_V11_8`, `CUDA_PATH_V12_1` 等，或扫描 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA`。
   - **Linux**: 扫描 `/usr/local/cuda-*` 目录。
2. **显示与选择**：告知用户探测到的所有 CUDA 版本，并询问：“检测到多个 CUDA 版本：[版本列表]，您想使用哪一个？”
3. **执行切换**：
   - 使用 `xmake f --cuda=PATH` 手动指定。
   - 或者使用 `xmake g --cuda=PATH` 进行全局设置，避免频繁切换。
4. **验证**：运行 `xmake l detect.sdks.find_cuda` 确认 xmake 识别的版本。

### 3. 自动检索并添加本地库 (TensorRT/OpenCV/cuDNN 等)
**流程**：
1. **探索结构**：使用 `LS` 或 `Glob` 查看用户提供的目录。寻找 `include`、`inc`、`lib`、`bin`、`build` 等子目录。
2. **提取路径**：
   - `add_includedirs`: 指向包含 `.h` 或 `.hpp` 的目录。
   - `add_linkdirs`: 指向包含 `.lib` (Windows) 或 `.so/.a` (Linux) 的目录。
   - `add_rpathdirs`: 指向包含 `.dll` (Windows) 或 `.so` (Linux) 的目录，确保运行时能找到动态库。
3. **提取链接库**：分析 `lib` 目录下的文件名，提取库名（如 `nvinfer.lib` -> `nvinfer`）。
4. **生成代码**：在 `xmake.lua` 中创建或更新相应的 `target` 配置。

**示例指令**：
"帮我把 D:\libs\TensorRT-8.x 加到 xmake 项目里" -> Agent 自动扫描该目录并生成 `add_includedirs`、`add_linkdirs`、`add_links` 和 `add_rpathdirs`。

### 3. 加载本地依赖 (以 OpenCV 为例)
**方案**：直接在 `xmake.lua` 中指定路径，避免 `add_requires` 自动下载。
```lua
target("your_target")
    add_includedirs("D:/3rdparty/opencv_4.8.1/build/include")
    add_linkdirs("D:/3rdparty/opencv_4.8.1/build/x64/vc16/lib")
    add_links("opencv_world481")
    add_rpathdirs("D:/3rdparty/opencv_4.8.1/build/x64/vc16/bin") -- 增加运行时目录
```

### 4. 下载加速与多代理切换 (Failover)
**场景**：由于网络原因，xrepo 下载 GitHub 包极慢或失败。
**修复逻辑**：
1. **配置 PAC 脚本**：引导用户创建 `pac.lua`，并内置多个常用的 GitHub 镜像站。
2. **多源备选与自动测速**：
   - 引导用户访问 `https://xiake.pro/` 获取最新节点。
   - **动态优化**：Agent 可运行 `scripts/ping_mirrors.mjs` 对已知或新获取的节点进行本地测速。
   - 自动选择延迟最低的节点更新到 `pac.lua`。
3. **全局生效**：使用 `xmake g --proxy_pac=pac.lua` 命令。
4. **清理重试**：切换代理后，务必提醒用户执行 `xmake f -c`。

## 常用命令备忘 (Cheatsheet)

| 命令 | 说明 |
| --- | --- |
| `xmake f -c` | 清除项目本地配置，重新探测环境 |
| `xmake g --clean` | **清除 xmake 全局缓存** (处理工具链检测异常的关键) |
| `xmake clean -a` | 清除项目所有中间文件和生成目标 |
| `xmake g --pkg_searchdirs="DIR"` | **设置本地包搜索目录** (用于离线安装包) |
| `xmake g --proxy_pac="PATH"` | **设置代理 PAC 脚本** (解决下载慢、GitHub 访问难的问题) |
| `xrepo export -o <dir> <pkg>`| **导出已安装的包** (实现跨机器环境迁移的推荐方式) |
| `xrepo import -i <dir> <pkg>`| **导入已导出的包** (配合 export 实现离线环境快速部署) |
| `xmake f -m releasedbg` | 切换到 releasedbg 模式 |
| `xmake show -t <target>` | 显示目标的详细配置信息 |
| `xmake project -k vsxmake2022` | 生成 VS 工程文件 |
| `xmake l find_package opencv` | 检测系统上的包信息 |

## GCC 探测与环境 (GCC Detection)

| 命令 | 说明 |
| --- | --- |
| `gcc -dumpmachine` | 获取当前的 GCC 架构 (如 aarch64-linux-gnu) |
| `ldd --version` | 查看 GLIBC 版本 |
| `which gcc` / `whereis gcc` | 定位 GCC 安装路径 |
| `gcc -v` | 查看详细的版本和编译配置信息 |
| `sudo apt install gcc-aarch64-linux-gnu` | 安装 ARM 架构的交叉编译器 |

## XRepo 常用命令 (XRepo Cheatsheet)

| 命令 | 说明 |
| --- | --- |
| `xrepo install <package>` | 安装指定的包 |
| `xrepo remove <package>` | 卸载指定的包 |
| `xrepo search <package>` | 搜索可用的包 |
| `xrepo info <package>` | 查看包的详细信息（版本、配置项等） |
| `xrepo update-repo` | 更新官方包仓库 |
| `xrepo add-repo <name> <url> [branch]` | **添加自定义仓库** (如 Gitee 镜像) |
| `xrepo env shell` | 进入包含已安装包的环境 Shell |
| `xrepo scan` | 扫描并清理孤立的包 |
| `xrepo remove --all <package>` | 彻底删除指定包的所有版本 |

**添加 Gitee 镜像仓库示例**：
```bash
xrepo add-repo gitee https://gitee.com/tboox/xmake-repo master
```

**设置代理镜像示例**：
```bash
xmake g --proxy_pac=github_mirror.lua
```

## 参考资料
更多详细的编译示例和高级配置，请参考：
- [编译案例与技巧](file:///.trae/skills/compile_xmake/references/compile_tips.md)
- [xmake.lua 模板库](file:///.trae/skills/compile_xmake/references/templates.md) (含 CI/CD 模板)
- [插件开发指导](file:///.trae/skills/compile_xmake/references/plugin_dev.md)
- [进阶疑难杂症排除](file:///.trae/skills/compile_xmake/references/troubleshooting_advanced.md)

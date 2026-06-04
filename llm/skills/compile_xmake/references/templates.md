# xmake.lua 模板库

## 1. 基础 C++ 项目模板
```lua
set_project("MyProject")
set_version("1.0.0")
set_languages("c++17")
add_rules("mode.debug", "mode.release")

target("app")
    set_kind("binary")
    add_files("src/*.cpp")
    add_includedirs("include")
```

## 2. 深度学习/三方库集成模板
```lua
target("AIFramework")
    set_kind("shared")
    -- 规则化管理三方库
    add_rules("package_cuda")
    add_rules("package_tensorrt")
    add_rules("package_opencv")

    add_files("src/**.cpp", "src/**.cu")

    if is_kind("static") then
        set_policy("build.cuda.devlink", true)
    end
```

## 3. 多目标项目模板
```lua
tutorial_list = {"T01_Hello", "T02_Advance"}

for _, v in pairs(tutorial_list) do
    target(v)
        set_kind("binary")
        add_includedirs("src")
        add_files("src/**.cpp", string.format("tutorial/%s.cpp", v))
end
```

## 4. 自定义选项 (Option)
用于通过命令行参数控制编译行为：
```lua
option("tensorrt")
    set_showmenu(true)
    set_description("Enable TensorRT support")
    on_check(function (option)
        if not option:enabled() then
            raise("TensorRT path is not set.")
        end
    end)

target("app")
    if has_config("tensorrt") then
        add_defines("USE_TENSORRT")
    end
```

## 5. 自定义规则 (Rule)
用于封装通用的构建逻辑或库集成：

### 库集成规则 (on_config)
```lua
rule("package_opencv")
    on_config(function (target)
        target:add("includedirs", "3rdparty/opencv/include")
        target:add("linkdirs", "3rdparty/opencv/lib")
        if is_mode("release") then
            target:add("links", "opencv_world481")
        else
            target:add("links", "opencv_world481d")
        end
    end)
rule_end()

target("test")
    add_rules("package_opencv")
```

### 构建后处理规则 (after_build)
```lua
-- 1. 复制文件
rule("rule_copy_output")
    after_build(function (target)
        os.cp(target:targetfile(), "$(projectdir)/install")
        cprint("${green}Copying %s to install dir...", target:filename())
    end)
rule_end()

-- 2. 显示构建目标路径
rule("rule_display")
    after_build(function (target)
        cprint("${green} my output path: %s", target:targetfile())
    end)
rule_end()
```

## 6. OpenMP 集成
展示了如何在不同模式和平台下开启 OpenMP 支持：
```lua
target("omp_test")
    set_kind("binary")
    add_files("src/*.cpp")

    -- 方式 1: 使用 xmake 内置包 (推荐)
    add_requires("openmp")
    add_packages("openmp")

    -- 方式 2: 手动配置编译选项 (针对特定编译器)
    if is_mode("release") then
        if is_plat("windows") then
            add_cxxflags("/openmp")
        else
            add_cxxflags("-fopenmp")
            add_ldflags("-lopenmp") -- 修正：使用用户笔记中的 -lopenmp
        end
    end
```

## 7. 全局加速代理与多源切换 (pac.lua)
用于解决 GitHub 下载包缓慢或失败的问题。建议创建一个 `pac.lua` 并运行 `xmake g --proxy_pac=pac.lua`。

### 自动多镜像切换逻辑
```lua
-- pac.lua
-- 提示：可以运行 scripts/ping_mirrors.mjs 来获取当前最快节点
local mirrors = {
    "https://ghfast.top/",
    "https://v6.gh-proxy.org/",
    "https://v4.gh-proxy.org/",
    "https://cdn.gh-proxy.org/",
    "https://ghfile.geekertao.top/",
    "https://g.blfrp.cn/",
    "https://gh-proxy.org/"
    "https://xiake.pro/" -- 侠客代理入口
}

function mirror(url)
    -- 仅针对 github.com 进行镜像替换
    if url:find("github.com") then
        -- 这里的逻辑由 xmake 内部在下载失败时多次调用或由 Agent 引导切换
        -- 进阶写法：支持在脚本中定义多个候选地址
        return url:gsub("https://github.com/", mirrors[1])
    end
    return url
end

function main(url, host)
    -- 也可以针对特定 host 返回 true 走系统全局代理
    if host:find("google.com") or host:find("github.com") then
        return true
    end
end
```

### 故障切换 (Failover) 与动态发现策略
当下载失败时：
1. **自动测速**：Agent 运行 `node .trae/skills/compile_xmake/scripts/ping_mirrors.mjs`。
2. **节点更新**：从 `https://xiake.pro/` 获取新节点并加入测速列表。
3. **脚本重载**：将测速结果第一名更新到 `pac.lua`。
4. **环境清理**：`xmake f -c` 强制刷新。

## 8. VS 工程自动更新 (vsxmake)
比传统的 `vstudio` 插件更现代，支持文件变动自动同步：
```lua
add_rules("plugin.vsxmake.autoupdate")

target("app")
    set_kind("binary")
    add_files("src/*.cpp")
```
生成工程命令：`xmake project -k vsxmake2022`

## 9. CI/CD 自动化集成 (GitHub Actions)
用于在云端自动构建和测试项目。建议放置在 `.github/workflows/build.yml`。

### 通用构建模板 (Windows/Linux/macOS)
```yaml
name: Build
on: [push, pull_request]

jobs:
  build:
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: xmake-io/github-action-setup-xmake@v1
        with:
          xmake-version: latest
          actions-cache-folder: '.xmake-cache' # 开启缓存加速
          package-cache: true                  # 缓存三方包

      - name: Configure
        run: xmake f -m release -y

      - name: Build
        run: xmake

      - name: Test
        run: xmake test
```

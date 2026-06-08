# xmake 插件开发指导

xmake 的插件系统允许你扩展自定义命令。插件通常放在项目根目录的 `scripts` 目录下，或者全局插件目录。

## 1. 基础插件结构
创建一个名为 `hello` 的插件，可以运行 `xmake hello`。

文件路径：`$(projectdir)/plugins/hello/xmake.lua`
```lua
task("hello")
    set_menu({
        usage = "xmake hello [options]",
        description = "A simple hello plugin",
        options = {
            {'v', "verbose", "k", nil, "Show verbose information."}
        }
    })

    on_run(function ()
        import("core.base.option")
        local verbose = option.get("verbose")
        if verbose then
            cprint("${green}Hello, Xmake! (Verbose mode on)")
        else
            cprint("${green}Hello, Xmake!")
        end
    end)
```

## 2. 常用开发接口
在插件中，你可以使用 xmake 丰富的内部接口：

- `import("core.project.project")`: 获取项目信息（targets, options 等）。
- `import("core.base.option")`: 获取命令行参数。
- `import("core.base.global")`: 获取全局配置。
- `import("core.tool.toolchain")`: 获取工具链信息。
- `os.exec("command")`: 执行外部系统命令。

## 3. 实战示例：自动清理特定后缀文件
```lua
task("clean_ext")
    set_menu({
        usage = "xmake clean_ext [extension]",
        description = "Clean files with specific extension",
    })

    on_run(function ()
        import("core.base.option")
        local ext = option.get("extension") or ".tmp"
        local files = os.files("**" .. ext)
        for _, file in ipairs(files) do
            os.rm(file)
            cprint("${red}Removed: %s", file)
        end
    end)
```

## 4. 如何安装与运行
1. **项目级插件**：将插件目录放在项目根目录，在 `xmake.lua` 中引用：
   ```lua
   add_plugindirs("plugins")
   ```
2. **全局插件**：放在 `~/.xmake/plugins`。
3. **运行**：`xmake [task_name]`。

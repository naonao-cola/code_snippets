# 进阶疑难杂症排除 (Advanced Troubleshooting)

## 1. Windows 工具链污染 (MSYS2 vs MSVC)
**症状**: 在普通 CMD/PowerShell 中运行 `xmake`，却报错找不到 GCC，或者链接时出现 `__msys2_xxx` 符号错误。
**根因**: PATH 环境变量中包含了 MSYS2 或 MinGW 的路径，且优先级高于 MSVC。xmake 探测到了这些环境并尝试混合使用。
**解决方案**:
- **净化环境**: 检查 `PATH`，移除不必要的 MSYS2 路径。
- **显式固定**: 强制 xmake 只看 MSVC：
  ```bash
  xmake f -c --vs=2022 -a x64
  ```
- **xmake.lua 隔离**: 在 `on_load` 中清理环境变量：
  ```lua
  on_load(function (target)
      for _, var in ipairs({"MSYSTEM", "PKG_CONFIG_PATH"}) do
          os.setenv(var, nil)
      end
  end)
  ```

## 2. cl.exe 拒绝访问 / 预处理失败
**症状**: 报错 `nvcc fatal : Failed to preprocess ...` 或 `Access Denied`。
**根因**: 通常是因为 `cu-ccbin` 指向了错误的目录，或者当前的 Terminal 权限不足以调用 VS 的编译驱动。
**解决方案**:
- 不要手动在 `xmake.lua` 里写 `set_toolset("cu-ccbin", "path/to/cl.exe")`。
- 应该在 `xmake f` 命令行中通过 `--vs_toolset` 自动让 xmake 对齐。

## 3. 跨机器包迁移
**症状**: 另一台电脑无法联网，或者下载包太慢。
**方案**: 使用 XRepo 的导出导入功能。
- **机器 A (有网)**: `xrepo export -o E:\pkgs opencv boost`
- **机器 B (无网)**: `xrepo import -i E:\pkgs opencv boost`
这样可以完美替代手动复制 `.xmake` 目录的原始做法，保证包的元数据完整且路径自适应。

## 4. 深度诊断参数
当 `xmake` 运行结果不符合预期时，按顺序尝试：
1. `xmake -v`: 显示编译详细命令（查看 include/link 路径是否正确）。
2. `xmake -vD`: 显示详细的探测和调试信息（查看 xmake 内部逻辑）。
3. `xmake f -c`: 彻底重置本地缓存。
4. `xmake g --clean`: 彻底重置全局缓存。

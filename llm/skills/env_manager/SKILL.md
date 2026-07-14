---
name: env_manager
description: 统一处理 Python 和 C/C++ (xmake) 环境的创建、配置、编译和管理。包括 Python 环境快速搭建、镜像源配置、包名验证，以及 xmake 编译问题修复、CUDA/MSVC 兼容性、本地库配置等。当用户遇到 Python 环境配置、xmake 编译报错、包安装问题时触发。
---

# env_manager

本技能用于解决 Python 和 xmake 编译过程中的环境配置与管理问题。

## 零期准备：全局诊断 SOP (Standard Operating Procedure)
> 作为一个高级 Agent，在直接跳入特定领域前，**必须先执行环境侦察**，拒绝盲人摸象。
1. **查探项目指纹**：使用 `ls` 或 `glob` 扫描当前目录，寻找 `xmake.lua`, `CMakeLists.txt`, `requirements.txt`, `environment.yml`, `setup.py`, `pyproject.toml` 等环境特征文件。
2. **确认当前状态**：若是编译报错，必须先看完整的报错日志；若是环境缺失，先执行 `python -V` 或 `xmake --version` 或 `uv --version` 摸底。
3. **系统与硬件嗅探**：通过 `uname -a`, `nvidia-smi` (若有 GPU) 确认底层硬件与 OS，防止给出错误的架构建议。

由于本技能包含两个庞大的专业领域（Python 环境管理与 C/C++ xmake 编译构建），为了保持工作区清晰且让你能够精准获取所需上下文，请根据用户的具体需求，**点击并阅读对应的参考文件**。

## 路由指南 (Router)

### 👉 场景一：处理 Python 环境与依赖
**触发条件**：用户需要创建 conda 环境、配置 pip/conda 国内源（如清华源）、安装包（如 opencv, torch）、或者遇到 pip 包名冲突、依赖报错等问题。
**你需要阅读的文件**：
- [Python 环境创建、源配置与包名验证指南](file:///.trae/skills/env_manager/references/python.md)

---

### 👉 场景二：处理 C/C++ 与 xmake 编译构建
**触发条件**：用户遇到 xmake 编译报错（如 `STL1002`，MSVC 不兼容）、需要切换 CUDA 版本、需要通过 xmake 链接本地库（如 TensorRT, OpenCV），或者需要加速下载 github 依赖等问题。
**你需要阅读的文件**：
- [xmake 编译管理与疑难杂症修复](file:///.trae/skills/env_manager/references/cpp_xmake.md)

---

### 👉 场景三：处理 Python 与 C/C++ 混合编译环境 (pybind11 / C-Extension)
**触发条件**：项目既包含 Python 脚本，又包含需要编译的 C/C++ 扩展（例如通过 `pybind11` 绑定，或者通过 xmake 编译 `.so`/`.pyd` 给 Python 调用），导致链接报错、找不到 Python.h 等。
**你需要阅读的文件**：
- 综合阅读 [python.md](file:///.trae/skills/env_manager/references/python.md) 和 [cpp_xmake.md](file:///.trae/skills/env_manager/references/cpp_xmake.md)，重点关注如何通过 xmake 的 `add_requires("python 3.x")` 以及本地库链接机制打通两者。

---

## 附加参考资料 (仅在特定 xmake 场景下按需阅读)
如果上面的 `cpp_xmake.md` 无法解决复杂的 xmake 编译问题，请参考以下进阶资料：
- [编译案例与技巧](file:///.trae/skills/env_manager/references/compile_tips.md)
- [xmake.lua 模板库](file:///.trae/skills/env_manager/references/templates.md) (含 CI/CD 模板)
- [插件开发指导](file:///.trae/skills/env_manager/references/plugin_dev.md)
- [进阶疑难杂症排除](file:///.trae/skills/env_manager/references/troubleshooting_advanced.md)

---

## 终局思维：环境固化与闭环沉淀
> 记住，P10 不只是解决眼前的一次编译报错。你的目标是**让同类问题不再发生**。
1. **持久化输出**：如果本次会话你帮用户解决了一个复杂的依赖问题（比如解决了 CUDA 版本冲突，或是写出了正确的 pybind11 构建脚本），**绝对不能只在终端里跑通就结束**。
2. **落地为资产**：必须将成功的配置固化到项目资产中。例如更新项目里的 `xmake.lua`，或者更新 `requirements.txt`/`environment.yml`，又或是主动在项目里创建一个 `build_guide.md` 说明本次排坑过程。没有固化，就不叫闭环。
3. **隔离性验证**：在固化环境后，反问自己：如果将这个工程 clone 到一台全新裸机上，当前的构建脚本能否一键跑通？如果不能，你需要引导用户补充缺失的自动化拉取逻辑。
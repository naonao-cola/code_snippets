# 项目全局记忆与架构约束 (Project Memory & Architecture Constraints)

> **注意**：本文件最初为 Trae 全局级别的 `project_memory.md`。为实现版本控制 (Git) 追踪，现已下放并内聚到 `cpp_toolkit` 技能的 `references/` 目录中。它承载着本项目的核心红线和架构基因。

## 1. Hard Constraints (硬性红线)
- **【最高红线】代码独立生成原则**：严禁生成依赖技能库内部路径的代码，必须物理级提炼逻辑并直接内嵌到输出目录中。
- **【质量红线】Web 搜索二次校验**：涉及高阶 API、CUDA Kernel 或易错算法时，必须强制联网搜索验证。
- **多并发支持**：必须支持多流 (Multi-Stream) 与多上下文 (Multi-Context) 推理，并使用 `std::enable_shared_from_this` 安全管理生命周期。
- **TRT 插件开发**：自定义插件必须实现 `IPluginV2DynamicExt` 和 `IPluginCreator`，且 `enqueue` 需传入 TensorRT 分配的 `cudaStream_t`。
- **C++ 代码规范强制对齐 `cpp-coding-standards`**：必须遵循 RAII，严禁裸 `new`/`delete`。
- **【进化红线】技能反向哺育 (Self-Evolution)**：任务结束前必须复盘，将新发现的 Bug 根因、高阶写法或架构优化作为 Lessons Learned 反向写入 `SKILL.md`、`references/*.md` 及本约束文档。

## 2. Engineering Conventions (工程惯例)
- **工具链约束**：所有工程脚手架生成必须通过 `scripts/cpp_cli.py` 执行，强制执行解耦规则。
- **技能元数据规范**：`SKILL.md` 必须包含符合 `skill-creator` 标准的 YAML Frontmatter。
- **自进化落地行为**：复盘时需提取核心 Snippet 至 `scripts/templates/`，更新 `references/` 文档，并扩充 `SKILL.md` 的路由触发词。

## 3. Lessons Learned (避坑指南与经验沉淀)
- **物理隔离闭环**：曾经出现过工具链脱节的 Tech Debt。通过重写了 `cpp_cli.py` 替代空壳实现，强制要求物理隔离输出目录，解决了这一问题。
- **技能生态集成**：通过将 `web-access` 和 `cpp-coding-standards` 硬编码进触发逻辑，解决了由于大模型记忆带来的代码规范隐患和幻觉问题。

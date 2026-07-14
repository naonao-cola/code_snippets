# 第一部分：C++ 基础与进阶写法 (Basics & Modern C++)

> **定位：** 本模块关注 C++ 的基础语法、面向对象设计、以及现代 C++ (C++11/14/17/20) 的高级/升级写法。

## 1. 常规 C++ 基础与规范
### 编码与编译选项
- 跨 win/linux 项目的编码选择：首选 **UTF-8 With BOM**。
- Visual Studio Code 开发环境建议搭配 `Clangd` + `XMake`。

### 内存对齐 (Memory Alignment)
- 使用 `#pragma pack(1)` 取消默认的 4 字节对齐，强制为 1 字节对齐（如解析位图 BMP 头时非常必要）。

### 宏与小工具 (Macros & Utils)
- **分支预测优化**：`likely(x)` / `unlikely(x)` 宏（基于 GCC/Clang 的 `__builtin_expect`）。
- **类拷贝控制**：`DISALLOW_COPY_AND_ASSIGN` 宏，禁用拷贝构造与赋值。
- **线程安全输出**：基于 `std::ostringstream` 和 `std::mutex` 封装的 `Cout` 类，配合 `XOUT` / `XLOG` 宏使用。
- **跨平台时间封装**：`now_ms()`, `now_us()`, `epoch_ms()`, `sleep_sec()` 等，底层针对 Windows (`QueryPerformanceCounter` / `GetSystemTimeAsFileTime`) 与 Linux (`clock_gettime` / `gettimeofday`) 进行了跨平台封装。
- **其他基础宏**：`CLIP_RANGE`, `SWAP`。

> **💡 代码参考**：有关 `defer`、计时器、线程安全输出以及跨平台时间获取的完整 C++ 宏和源文件实现，请参考模板脚本：
> - 头文件：`../scripts/templates/1_basic/tools.h`
> - 源文件：`../scripts/templates/1_basic/tools.cpp`

### 编译依存性降低 (Compilation Dependencies)
为了将文件间的编译依存关系降至最低，遵循“相依于声明式，不要相依于定义式”的原则 (Effective C++ 条款 31)：
- **Handle Classes**：在头文件中使用前置声明（`class xxx;`）和指针/引用代替变量对象，在 `.cpp` 文件中才 `#include` 具体的头文件。
- **Interface Classes**：父类仅包含虚方法（纯虚函数）和一个静态的 `Create` 函数声明（如工厂模式），将逻辑实现下沉到子类，利用多态调用。

### C++ DLL 导出机制
- 建议放弃普通的 `__declspec(dllexport)` 导出具体类，转而定义只包含**纯虚函数的抽象类**。
- 配合**工厂模式**，将工厂类注册到服务中心，对外仅暴露接口指针，彻底解耦。

### 强制类型转换 (Casting)
- `static_cast`：用于比较自然和低风险的转换（如整型、浮点型互相转换）。
- `reinterpret_cast`：执行逐个比特的复制，风险最高。用于不相关类型指针或引用间的强制转换。
- `const_cast`：专门用于去除变量的 `const` 属性。
- `dynamic_cast`：用于将多态基类的指针/引用安全地向下转型，转型失败返回 `NULL`。

## 2. 现代 C++ 升级写法 (Modern C++)

### 智能指针 (Smart Pointers)
- **避免滥用 `shared_ptr`**：在反复申请和赋值的场景下，`shared_ptr` 因内部的 CAS 校验机制会导致性能下降。当不需要共享所有权时，**优先使用 `std::unique_ptr`**。
- `unique_ptr` 数组支持 `[]` 操作，无 `*` 和 `->`。`shared_ptr` 数组有 `*` 和 `->`，但不支持 `[]`（需自定义 deleter 或使用 C++17+ 特性）。
- **二维数组与自定义 Deleter**：`std::shared_ptr<Student[1024][10]> pt(new Student[1024][10], ...)`。
- **妙用**：将 `shared_ptr` 结合自定义 `deleter` 作为互斥锁（Mutex）的自动释放器，实现 RAII 锁管理。
- 初始化技巧：`auto ptr = std::make_unique<std::array<int, 5>>();`

### 移动语义 (`std::move`)
- 全程传参和赋值中，尽可能采用 `std::move` 和 `emplace` 传递，避免产生中间流程无意义的 Copy。

### Lambda 表达式
- 语法：`[capture](params) opt -> ret { body; }`
- 捕获列表：`[]`（不捕获），`[&]`（按引用），`[=]`（按值），`[this]`（捕获类指针）。
- **`mutable` 关键字**：如果希望修改按值 `[=]` 捕获的外部变量，必须显示声明 `mutable` 以取消 `operator()` 的默认 `const` 属性。
- **泛型 Lambda 与折叠表达式** (C++17)：
  ```cpp
  static auto anyone = [](auto&& k, auto&&... args) -> bool { return ((args == k) || ...); };
  if(anyone(x, 'x', 'X', 'e', 'E', '.')) { work(); }
  ```

### 关键字与特性
- **`volatile`**：防止编译器优化导致从寄存器读取过期的变量值。多用于多线程共享标志、中断服务程序或硬件寄存器映射。
- **`constexpr`**：将函数或变量的计算提前到**编译期**，提升运行期效率。可修饰普通函数或类的构造函数（函数体必须为空，用初始化列表）。
- **内联变量 (Inline Variables, C++17)**：`inline int k = 10;`，头文件直接定义，避免重定义错误。
- **继承构造函数**：`using Base::Base;` 避免子类写繁琐的透传构造函数。

### 其他实用工具 (Tips)
> **💡 代码参考**：有关 C++ 智能指针高阶用法（自定义删除器、2D数组、锁包装器）、降低编译依存性（Interface Classes）的具体实现，以及四种强制类型转换的代码示例，请参考模板脚本：`../scripts/templates/1_basic/cpp_tips_snippets.cpp`。

- **内联 lambda 转函数指针技巧 (+lambda)**:
  ```cpp
  auto CreateLonglink = +[](const std::string& name) -> int8_t { return 0; };
  ```
- **Sleep 的颗粒度**：使用 `std::this_thread::sleep_for(std::chrono::milliseconds(100));` 替代传统的跨平台宏，更加 Modern C++。

## 3. 工程化与性能分析工具 (Profiling & Tools)

> **💡 代码参考**：有关 `perf` 和 `uftrace` 的详细命令，以及配套的性能测试源码，请参考模板脚本：
> - 命令速查脚本：`../scripts/templates/1_basic/perf_uftrace_cmds.sh`
> - C++ 测试源码：`../scripts/templates/1_basic/perf_uftrace_demo.cpp`

### 性能分析神兵：perf & FlameGraph
- **安装坑点 (WSL)**：WSL 环境下经常遇到 `perf not found for kernel` 的问题。可以通过自行拉取 `WSL2-Linux-Kernel` 源码进入 `tools/perf` 进行编译 (`make -j8`)，并拷贝到 `/usr/local/bin`。
- **采样收集**：`perf record -F 99 -a -g ./executable` (99Hz，全 CPU，记录调用栈)。
- **硬件指标统计**：`perf stat -e cache-misses,cpu-clock ./executable` 可用于分析缓存命中率。
- **生成火焰图**：依赖 `FlameGraph` 脚本集，一行管道命令直出可视化：
  ```bash
  perf script -i perf.data | stackcollapse-perf.pl | flamegraph.pl > perf.svg
  ```

### 函数级追踪利器：uftrace
- **核心定位**：当需要明确知道每个 C++ 函数的**调用次数**、**自身耗时 (Self Time)**、甚至**入参与返回值**时，`uftrace` 是首选（需配合 `-pg` 编译选项）。
- **库函数与内核屏蔽**：C++ 中常被 `operator new` 或 `std::ostream` 刷屏，可通过 `--no-libcall` 屏蔽，或利用 `-F` (仅追踪某函数) 和 `-t 1us` (过滤细碎耗时) 实现精准抓手。
- **自动化参数追踪**：`uftrace -a` 或细粒度的 `uftrace -A atoi@arg1/s` 可以直接窥探运行时数据流向。

## 4. 优秀教程与参考资源
- **现代 C++ 教程 (C++ 11/14/17/20)**: [changkun.de](https://changkun.de/modern-cpp/pdf/modern-cpp-tutorial-zh-cn.pdf)
- **C++ 那些事**: [light-city.github.io](https://light-city.github.io/)
- **C++ 模板元编程从易到难**: [知乎专栏](https://zhuanlan.zhihu.com/p/659060939)
- **设计模式的 C++ 实现 (22种)**: [知乎专栏](https://zhuanlan.zhihu.com/p/476220724) / [Gitee 源码](https://gitee.com/naoano/design_pattern)
- **C++ 代码调试的艺术**: [微信读书](https://weread.qq.com/web/reader/423320c07228f7b6423975a)
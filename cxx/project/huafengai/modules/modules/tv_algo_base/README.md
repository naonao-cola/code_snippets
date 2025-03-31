## TVT算法运行框架


### config
配置文件目录，模拟前端传入的json参数

### data
测试数据目录

### test
test代码目录
此目录里面的代码分为两个项目的代码。main.cpp是测试接口dll的项目代码，此项目依赖dll项目。*case.cpp是测试框架内的类的代码，此项目是对框架内的所有文件进行编译。通过宏 EXPORT_API进行区分，详细情况查看xmake.lua文件。

### 工作目录
工作目录是以xmake.lua文件的目录为基准。./就是lua文件的目录。可以在vs2022里面通过项目属性->调试->工作目录 进行查看修改。

### 编译target
- tv_algorithm： 算法包dll
- test_dll: 测试程序（调用dll接口）
- test_exe: 测试程序 (不调用dll，整体编译exe)

### 编译
- xmake b

### 清除编译
- xmake c -a

### 转visual studio工程
xmake project -k "vsxmake2022" -m "release" vs2022 

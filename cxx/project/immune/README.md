# 免疫项目本地测试说明
该文档主要阐释了独立的免疫项目本地测试程序使用方法.

# 使用已编译程序
1.将程序,运行依赖库,运行脚本,测试数据,试剂卡信息,lua脚本上传至
板上:```adb push immune_demo ./lib run.sh 1.txt 1.card alg.lua /data/alg_test/2immune```.其中```/data/alg_test/2immune```
为上传目录,```adb push```为上传命令.  
2.确认上传是否成功,
打开命令行窗口,输入```adb root```切换为root模式,输入```adb shell```进入板子命令行窗口.输入```cd /data/alg_test/2immune ```及```ls```确认上传
文件是否存在,若上传成功,目录结构应当如下:
```
  2immune  
  ├─ immune_demo  
  ├─ lib  
  ├─ run.sh  
  ├─ 1.txt
  ├─ 1.card  
  ├─ alg.lua
  
```
 
# 测试  
进入板子命令行窗口,进入上传的程序目录如```/data/alg_test/2immune```, 输入```./run.sh```即可得到运行结果.


# 使用源码编译
1.将```./CMakeLists.txt```文件中的```CMAKE_C_COMPILER```修改为工具链gcc编译工具路径,```CMAKE_CXX_COMPILER```修改为工具链g++编译工具路径,
工具链在服务器```/home/y/ALG/lf/rknn_tools```目录下.  
2.打开命令行窗口  
1).输入```mkdir build```  
2).输入```cd build```  
3).输入```cmake ..```  
4).输入```make install```  
编译完成后可参考```使用已编译程序```章节上传程序及测试.

# 免疫库
编译完成后,免疫库头文件地址```./src/include/immune.h```,库地址```./lib/libimmune.so```.```./lib```目录了下还存在```liblua.so```库,该库为三方库.
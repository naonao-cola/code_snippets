
#include "DihLogPlog.h"
#include "Format.h"
#include "ProjectManager.h"
#include "event.h"
#include "temp_test.h"
#include "timecnt.h"
#include <iostream>
#include <stdio.h>


using namespace std;
using namespace ALG_LOCAL;
using namespace ALG_DEPLOY;
bool regular_test{true};
// 正常测试脚本
bool RegularTest()
{
    cout << "Run Test." << endl;
    // 初始化日志
    EV_Init(EVFUNC_PRINT, nullptr, nullptr);
    //  auto info = dyno::log::DynoLog::getInstance("/data/alg_test/2reconstruct/log4cplus.properties");
    //  std::cout<<"init log succeed"<<std::endl;
    algPLogConfig("./");
    //  dihPLogConfig(".");
    std::string test = util::Format("This is a nice string with numbers {0} and strings {1} nicely formatted", 123, "hello");
    std::cout << " test "<<test<<std::endl;
    // 读取测试配置
    ProjectManager project{};
    if (!project.GetInitParams("./alg_config.xml")) {
        std::cout << "Fail to read xml." << std::endl;
        return 0;
    }
    // 根据配置进行初始化
    if (!project.Init()) {
        std::cout << "Fail to init." << std::endl;

        return 0;
    }
    // 推理
    project.Forward();
    std::cout << "Test Over." << std::endl;

    TimeCnt_PrintResult();
}

// 临时本地测试
bool TempTest()
{
    EV_Init(EVFUNC_PRINT, nullptr, nullptr);
    //  algPLogConfig("./");
    //  dihPLogConfig("./");
    std::cout << "temp test" << std::endl;


    /*  std::string model_path = "./temp_test/PERSON_INCLINE_RBC_SEG.rknn";
      std::string model_label_path = "./temp_test/PERSON_INCLINE_RBC_SEG.txt";
      std::string img_path = "./temp_test/test.bmp";
      TestSegNetwork(model_path, model_label_path, img_path);*/
    /*  TestClusterMatrix();*/
    TestOpencl();
}

int main()
{
    if (regular_test) {
        RegularTest();
    }
    else {
        TempTest();
    }
    return 0;
}

//
// Created by y on 2023/11/17.
//
//
// Created by y on 2023/10/26.
//

#include <functional>
#include <dirent.h>
#include <thread>
#include <future>

#include "IntegratedCode.h"
#include "replace_std_string.h"
#include "libalgcell.h"
#include "utils.h"

namespace ALG_LOCAL{
namespace INTEGRATE {

bool IntegratedCode::HeamoThread(){

  for (int i=0;i<1000;++i){
    bool thread_result = this->TestAlgSampleField();
    if(!thread_result){
      std::cout<<"Failed to run heamo in hybrid"<<std::endl;
      return false;
    }
  }
  return true;
}

bool IntegratedCode::ClarityThread(){

  for (int i=0;i<1000;++i){
    bool thread_result = this->TestAlgSampleClarity();
    if(!thread_result){
      std::cout<<"Failed to run clarity in hybrid"<<std::endl;
      return false;
    }
  }
  return true;
}


bool IntegratedCode::TestAlgSampleHybrid(){
  //获取多线程执行结果,以轮次为单位放入
  for(int i=0;i<1000;++i){
    std::packaged_task<int(int)> heamo_item(std::bind(&IntegratedCode::TestAlgSampleField, this, std::placeholders::_1));
    std::thread heamo_thread(std::ref(heamo_item), 500);//依次传入函数地址，对象指针，函数参数

    std::packaged_task<int(int)> clarity_item(std::bind(&IntegratedCode::TestAlgSampleClarity, this, std::placeholders::_1));

    std::thread clarity_thread(std::ref(clarity_item), 1);

    heamo_thread.join();
    clarity_thread.join();

    bool heamo_thread_result = heamo_item.get_future().get();
    if(!heamo_thread_result){
      std::cout<<"Failed to run heamo in hybrid"<<std::endl;
      return false;
    }
    bool clarity_thread_result = clarity_item.get_future().get();

    if(!clarity_thread_result){
      std::cout<<"Failed to run clarity in hybrid"<<std::endl;
      return false;
    }
  }

  //同时放入
  /*  std::packaged_task<int()> heamo_item(std::bind(&IntegratedCode::HeamoThread, this));
    std::thread heamo_thread(std::ref(heamo_item));//依次传入函数地址，对象指针，函数参数

    std::packaged_task<int()> clarity_item(std::bind(&IntegratedCode::ClarityThread, this));

    std::thread clarity_thread(std::ref(clarity_item));

    heamo_thread.join();
    clarity_thread.join();

    bool heamo_thread_result = heamo_item.get_future().get();
    if(!heamo_thread_result){
      std::cout<<"Failed to run heamo in hybrid"<<std::endl;
      return false;
    }
    bool clarity_thread_result = clarity_item.get_future().get();

    if(!clarity_thread_result){
      std::cout<<"Failed to run clarity in hybrid"<<std::endl;
      return false;
    }*/



  return true;
}
}
}
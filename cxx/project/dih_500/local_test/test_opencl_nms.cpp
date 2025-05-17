//
// Created by y on 24-9-29.
//
#include "temp_test.h"

#include <CL/cl.h>
#include <iostream>
#include <stdlib.h>
#include <opencv2/core.hpp>

#include "utils.h"
///
//	main() for HelloWorld example
//

//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Timothy Mattson, James Fung, Dan Ginsburg
// ISBN-10:   0-321-74964-2
// ISBN-13:   978-0-321-74964-2
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/9780132488006/
//            http://www.openclprogrammingguide.com
//

// HelloWorld.cpp
//
//    This is a simple example that demonstrates basic OpenCL setup and
//    use.

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string.h>
#include <numeric>
#include <math.h>



#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif


#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))


void ProcImg(const float* boxes, const std::vector<int>& keep_cl, const std::vector<int>& keep_cv){
  std::string img_path("./temp_test_source/nms.bmp");
  cv::Mat img_cl = cv::imread(img_path);
  auto img_cv = img_cl.clone();
  for(int i = 0; i<keep_cl.size(); ++i){
    if(keep_cl[i]==0){
      continue;
    }
    const float* cur_cl_box = boxes+i*4;
    cv::rectangle(img_cl,cv::Point(cur_cl_box[0], cur_cl_box[1]),
                  cv::Point(cur_cl_box[2],cur_cl_box[3]),
                  cv::Scalar(0,0,1), 1);
  }


  for(int i = 0; i<keep_cl.size(); ++i){
    if(keep_cv[i]==0){
      continue;
    }
    const float* cur_cl_box = boxes+i*4;
    cv::rectangle(img_cv,cv::Point(cur_cl_box[0], cur_cl_box[1]),
                  cv::Point(cur_cl_box[2],cur_cl_box[3]),
                  cv::Scalar(0,0,1), 1);
  }


  SaveImage("./temp_test_source/nms_cl.bmp", img_cl);
  SaveImage("./temp_test_source/nms_cv.bmp", img_cv);


}





///
//  Constants
//
const unsigned int ITEM_SIZE = sizeof(cl_ulong)* 8;




///
//  Create an OpenCL context on the first available platform using
//  either a GPU or CPU depending on what is available.
//
cl_context CreateContext()
{
  cl_int errNum;
  cl_uint numPlatforms;
  cl_platform_id firstPlatformId;
  cl_context context = NULL;

  // First, select an OpenCL platform to run on.  For this example, we
  // simply choose the first available platform.  Normally, you would
  // query for all available platforms and select the most appropriate one.
  errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
  if (errNum != CL_SUCCESS || numPlatforms <= 0)
  {
    std::cerr << "Failed to find any OpenCL platforms." << std::endl;
    return NULL;
  }
  // Next, create an OpenCL context on the platform.  Attempt to
  // create a GPU-based context, and if that fails, try to create
  // a CPU-based context.
  cl_context_properties contextProperties[] =
      {
          CL_CONTEXT_PLATFORM,
          (cl_context_properties)firstPlatformId,
          0
      };
  context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU,
                                    NULL, NULL, &errNum);
  if (errNum != CL_SUCCESS)
  {
    std::cout << "Could not create GPU context, trying CPU..." << std::endl;
    context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_CPU,
                                      NULL, NULL, &errNum);
    if (errNum != CL_SUCCESS)
    {
      std::cerr << "Failed to create an OpenCL GPU or CPU context." << std::endl;
      return NULL;
    }
  }

  return context;
}

///
//  Create a command queue on the first device available on the
//  context
//
cl_command_queue CreateCommandQueue(cl_context context, cl_device_id *device)
{
  cl_int errNum;
  cl_device_id *devices;
  cl_command_queue commandQueue = NULL;
  size_t deviceBufferSize = -1;

  // First get the size of the devices buffer
  errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
  std::cout<<deviceBufferSize<<std::endl;
  if (errNum != CL_SUCCESS)
  {
    std::cerr << "Failed call to clGetContextInfo(...,GL_CONTEXT_DEVICES,...)";
    return NULL;
  }

  if (deviceBufferSize <= 0)
  {
    std::cerr << "No devices available.";
    return NULL;
  }

  // Allocate memory for the devices buffer
  devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
  errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);
  if (errNum != CL_SUCCESS)
  {
    delete [] devices;
    std::cerr << "Failed to get device IDs";
    return NULL;
  }

  // In this example, we just choose the first available device.  In a
  // real program, you would likely use all available devices or choose
  // the highest performance device based on OpenCL device queries
  commandQueue = clCreateCommandQueue(context, devices[0], 0, NULL);
  if (commandQueue == NULL)
  {
    delete [] devices;
    std::cerr << "Failed to create commandQueue for device 0";
    return NULL;
  }

  *device = devices[0];
  delete [] devices;
  return commandQueue;
}

///
//  Create an OpenCL program from the kernel source file
//
cl_program CreateProgram(cl_context context, cl_device_id device, const char* fileName)
{
  cl_int errNum;
  cl_program program;

  std::ifstream kernelFile(fileName, std::ios::in);
  if (!kernelFile.is_open())
  {
    std::cerr << "Failed to open file for reading: " << fileName << std::endl;
    return NULL;
  }

  std::ostringstream oss;
  oss << kernelFile.rdbuf();

  std::string srcStdStr = oss.str();
  const char *srcStr = srcStdStr.c_str();
  program = clCreateProgramWithSource(context, 1,
                                      (const char**)&srcStr,
                                      NULL, NULL);
  if (program == NULL)
  {
    std::cerr << "Failed to create CL program from source." << std::endl;
    return NULL;
  }

  errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (errNum != CL_SUCCESS)
  {
    // Determine the reason for the error
    char buildLog[16384];
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                          sizeof(buildLog), buildLog, NULL);

    std::cerr << "Error in kernel: " << std::endl;
    std::cerr << buildLog;
    clReleaseProgram(program);
    return NULL;
  }

  return program;
}

///
//  Create memory objects used as the arguments to the kernel
//  The kernel takes three arguments: result (output), a (input),
//  and b (input)
//
bool CreateMemObjects(cl_context context, cl_mem memObjects[4],
                      float *a, int box_num, int g_w_0, int g_w_1)
{
  cl_int errNum;

  memObjects[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeof(cl_float) * box_num*4, a, &errNum);
  // mask
  memObjects[1] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                 sizeof(cl_ulong) * g_w_0*g_w_1, NULL, &errNum);

  // 存储调试
  memObjects[2] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                 sizeof(cl_uint) * 100, NULL, &errNum);


  // box数目
  if (memObjects[0] == NULL || memObjects[1] == NULL|| memObjects[2] == NULL)
  {
    std::cerr << "Error creating memory objects "<<errNum << std::endl;
    return false;
  }

  return true;
}

///
//  Cleanup any created OpenCL resources
//
void Cleanup(cl_context context, cl_command_queue commandQueue,
             cl_program program, cl_kernel kernel, cl_mem memObjects[3])
{
  for (int i = 0; i < 3; i++)
  {
    if (memObjects[i] != 0)
      clReleaseMemObject(memObjects[i]);
  }
  if (commandQueue != 0)
    clReleaseCommandQueue(commandQueue);

  if (kernel != 0)
    clReleaseKernel(kernel);

  if (program != 0)
    clReleaseProgram(program);

  if (context != 0)
    clReleaseContext(context);

}

bool ReadDataFromFile(const std::string& file_path, std::vector<float>& box_v){

  std::fstream file(file_path);
  if(file.is_open()){
    std::string line;
    while (std::getline(file,line)){
      std::stringstream  ss(line);
      float data;
      while(ss>>data){
        box_v.push_back(data);
      }
    }

    file.close();
  } else{
    std::cout<<"Failed to open file "<<file_path<<std::endl;
    return false;
  }
  for(int i=2,j=3;i<box_v.size();i=i+4, j=j+4){
    box_v[i] = box_v[i]+box_v[i-2];
    box_v[j] = box_v[j]+box_v[j-2];
  }


  return true;

}


static float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1,
                              float ymax1) {
  float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
  float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
  float i = w * h;
  float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) + (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;
  return u <= 0.f ? 0.f : (i / u);
}


void TestCpuTime(float * outputLocations, const int& validCount, std::vector<int>& keep_box){
  std::vector<int> order;
  for(int i = 0;i<validCount;++i){
    order.push_back(i);
  }
  int nums = 0;
  for (int i = 0; i < validCount; ++i) {
    if (order[i] == -1 ) {
      continue;
    }
    nums ++;
    keep_box[i]=1;
    int n = order[i];
    for (int j = i + 1; j < validCount; ++j) {
      int m = order[j];
      if (m == -1  ) {
        continue;
      }
      float xmin0 = outputLocations[n * 4 + 0];
      float ymin0 = outputLocations[n * 4 + 1];
      float xmax0 = outputLocations[n * 4 + 2];
      float ymax0 = outputLocations[n * 4 + 3];

      float xmin1 = outputLocations[m * 4 + 0];
      float ymin1 = outputLocations[m * 4 + 1];
      float xmax1 = outputLocations[m * 4 + 2];
      float ymax1 = outputLocations[m * 4 + 3];

      float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

      if (iou > 0.4) {
        order[j] = -1;
      }

    }
  }
  std::cout<<"Cpu keep nums "<<nums<<std::endl;

}



int TestOpenclNms()
{

  auto tm = TimeMonitor();
  cl_context context = 0;
  cl_command_queue commandQueue = 0;
  cl_program program = 0;
  cl_device_id device = 0;
  cl_kernel kernel = 0;
  cl_mem memObjects[4] = { 0, 0,0,0};
  cl_int errNum;
  std::string file_path ="./temp_test_source/boxes.txt";
  std::vector<float> box_v;
  if(!ReadDataFromFile(file_path, box_v)){
    std::cout<<"Error "<<std::endl;
    return 0;
  }
  //  float boxes[ARRAY_SIZE*4];
  // copy data
  const int temp_lenth = box_v.size()/4;
  float boxes[temp_lenth*4];
  if (!box_v.empty()){
    memcpy(boxes, &box_v[0], temp_lenth*4*sizeof(float));
  }

  std::cout<<"Load data succeed"<<std::endl;
  int box_num =temp_lenth;
  int col_num_64  = DIVUP(box_num, ITEM_SIZE);
  // Create an OpenCL context on first available platform
  context = CreateContext();
  if (context == NULL)
  {
    std::cerr << "Failed to create OpenCL context." << std::endl;
    return 1;
  }

  // Create a command-queue on the first device available
  // on the created context
  commandQueue = CreateCommandQueue(context, &device);
  if (commandQueue == NULL)
  {
    Cleanup(context, commandQueue, program, kernel, memObjects);
    return 1;
  }

  // Create OpenCL program from HelloWorld.cl kernel source
  program = CreateProgram(context, device, "./temp_test_source/nms.cl");
  if (program == NULL)
  {
    Cleanup(context, commandQueue, program, kernel, memObjects);
    return 1;
  }

  // Create OpenCL kernel
  kernel = clCreateKernel(program, "nms_cl", NULL);
  if (kernel == NULL)
  {
    std::cerr << "Failed to create kernel" << std::endl;
    Cleanup(context, commandQueue, program, kernel, memObjects);
    return 1;
  }

  size_t global_work_size[2] = { 0,0};//global work size need to exact division local work size

  //DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
  size_t              workgroup_size;
  size_t              local_work_size[2];
  // Create memory objects that will be used as arguments to
  // kernel.  First create host memory arrays that will be
  // used to store the arguments to the kernel
  clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &workgroup_size, NULL);
  {
    size_t  gsize[2];
    int     w;
    std::cout<<"max work group size "<<workgroup_size<<std::endl;
    if (workgroup_size <= 256)
    {
      gsize[0] = 16;
      gsize[1] = workgroup_size / 16;
    }
    else if (workgroup_size <= 1024)
    {
      gsize[0] = workgroup_size / 16;
      gsize[1] = 16;
    }
    else
    {
      gsize[0] = workgroup_size / 32;
      gsize[1] = 32;
    }

    local_work_size[0] = gsize[0];
    local_work_size[1] = gsize[1];

    int global_work_size_0 = DIVUP(box_num, gsize[0]);
    global_work_size[0] = global_work_size_0*gsize[0];//使大小为local的倍数

    int global_work_size_1 = DIVUP(col_num_64, gsize[1]);
    global_work_size[1] = global_work_size_1*gsize[1];
  }

  auto start = tm.Time();
  std::cout<<"global size "<<global_work_size[0]<<" "<<global_work_size[1]<<std::endl;
  std::cout<<"local size "<<local_work_size[0]<<" "<<local_work_size[1]<<std::endl;

  if (!CreateMemObjects(context, memObjects, boxes,box_num,global_work_size[0],global_work_size[1]))
  {
    Cleanup(context, commandQueue, program, kernel, memObjects);
    return 1;
  }

  // Set the kernel arguments (result, a, b)
  errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &memObjects[0]);

  errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &memObjects[1]);

  errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &memObjects[2]);
  errNum |= clSetKernelArg(kernel, 3, sizeof(cl_uint), &box_num);
  errNum |= clSetKernelArg(kernel, 4, sizeof(cl_uint), &col_num_64);
  if (errNum != CL_SUCCESS)
  {
    std::cerr << "Error setting kernel arguments." << std::endl;
    Cleanup(context, commandQueue, program, kernel, memObjects);
    return 1;
  }




  cl_ulong *result = (cl_ulong *)malloc(sizeof(cl_ulong)*global_work_size[0]*global_work_size[1]) ;

  std::cout<<"mask size "<<global_work_size[0]<<" "<<global_work_size[1]<<std::endl;




  // Queue the kernel up for execution across the array
  errNum = clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL,
                                  global_work_size, local_work_size,
                                  0, NULL, NULL);
  if (errNum != CL_SUCCESS)
  {
    std::cerr << "Error queuing kernel for execution." << std::endl;
    std::cout<<"error code "<<errNum<<std::endl;
    Cleanup(context, commandQueue, program, kernel, memObjects);

    return 1;
  }
  std::cout<<"start read buf "<<std::endl;
  // Read the output buffer back to the Host
  errNum = clEnqueueReadBuffer(commandQueue, memObjects[1], CL_TRUE,
                               0,   sizeof(cl_ulong)*box_num*col_num_64, result,
                               0, NULL, NULL);

  cl_int *result_idx = (cl_int *)malloc(sizeof(cl_int)*100) ;
  errNum |= clEnqueueReadBuffer(commandQueue, memObjects[2], CL_TRUE,
                                0,   sizeof(cl_int)*100, result_idx,
                                0, NULL, NULL);

  if (errNum != CL_SUCCESS)
  {
    std::cerr << "Error reading result buffer. "<<errNum << std::endl;
    Cleanup(context, commandQueue, program, kernel, memObjects);
    return 1;
  }

  std::cout<<"read succeed "<<std::endl;
  auto cost_end = tm.Time();


  // Output the result buffer
  //    for (int i = 0; i < 10000*10; i++)
  //    {
  //        std::cout << result_idx[i] << " ";
  //    }
  std::cout << std::endl;
  std::cout<<"opencl cost time "<<cost_end-start<<std::endl;
  // 第三步：遍历mask，获得最终的keep，在cpu上，因为cpu适合做逻辑密集的循环语句

  std::vector<int> keep_box=std::vector<int> (box_num,0);
  std::vector<cl_ulong> remv(box_num);
  for (int i = 0; i < box_num; i++) {
    int n_block = i/ITEM_SIZE;
    int in_block = i%ITEM_SIZE;
    if(!(remv[n_block]&1UL<<in_block)){//当前box在记录关系表中未被筛去
      keep_box[i] = 1; //保留当前box
      cl_ulong *p = result+i*col_num_64; //定位到i对应的iou行,
      for(int j = n_block; j<col_num_64; j++){//将第i个box对应的iou关系与记录有已确定iou关系的数据做{或},以更新iou表.前n_block中的box已经与当前box比较过,所以不需要比较
        remv[j] |=p[j];// 该步骤的逻辑为一次更新i号box对应的 agg_camp_num个关系,与单个做{或}相比,速度快 1/ITEM_SIZE;remv为总记录表
      }

    } else{
      keep_box[i] = 0; //不保留当前box
    }

  }
  cost_end = tm.Time();
  std::cout<<"total cost time "<<cost_end-start<<std::endl;
  /*  for(int i =0; i< ARRAY_SIZE; i++){
      std::cout << keep_box[i] << " ";
    }*/
  std::cout<<std::endl;
  int keep_nums = std::accumulate(keep_box.begin(), keep_box.end(),0);
  std::cout<<"Opencl keep nums "<<keep_nums<<std::endl;
  std::cout << "Executed program successfully." << std::endl;
  Cleanup(context, commandQueue, program, kernel, memObjects);

  start = tm.Time();
  std::vector<int> keep_box_cv(temp_lenth,0);
  TestCpuTime(boxes,temp_lenth, keep_box_cv);
  cost_end = tm.Time();
  std::cout<<"cpu cost time "<<cost_end-start<<std::endl;
  ProcImg(boxes, keep_box, keep_box_cv);

  free(result);
  free(result_idx);
  return 0;
}









int TestOpencl(){
  TestOpenclNms();
  //  TestOpenclInfo();
  return 0;
}

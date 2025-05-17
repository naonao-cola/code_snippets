//
// Created by y on 24-9-29.
//
#ifndef __CL_ENABLE_EXCEPTIONS
#define __CL_ENABLE_EXCEPTIONS
#endif
#include "NmsCl.h"
#include <CL/cl.h>
#include <fstream>
#include <sstream>


#include "algLog.h"
namespace ALG_CL{
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))

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

void CleanMem(cl_mem memObjects[3])
{
  for (int i = 0; i < 3; i++)
  {
    if (memObjects[i] != 0)
      clReleaseMemObject(memObjects[i]);
  }

}



int NmsCl::Init(const std::string& cl_path){
  ALGLogInfo<<"nms cl path "<<cl_path;
  // Create an OpenCL context on first available platform
  context = CreateContext();
  if (context == NULL)
  {
    ALGLogError<<"Failed to create OpenCL context." ;
    return -1;
  }

  // Create a command-queue on the first device available
  // on the created context
  commandQueue = CreateCommandQueue(context, &device);
  if (commandQueue == NULL)
  {
    Cleanup(context, commandQueue, program, kernel, memObjects);
    ALGLogError<<"Failed to create OpenCL commandQueue." ;
    return -1;
  }

  // Create OpenCL program from HelloWorld.cl kernel source
  program = CreateProgram(context, device, cl_path.c_str());
  if (program == NULL)
  {
    Cleanup(context, commandQueue, program, kernel, memObjects);
    ALGLogError<<"Failed to create OpenCL program." <<cl_path ;
    return -1;
  }

  // Create OpenCL kernel
  kernel = clCreateKernel(program, cl_kernel_name.c_str(), NULL);
  if (kernel == NULL)
  {
    ALGLogError << "Failed to create kernel" <<cl_kernel_name;
    Cleanup(context, commandQueue, program, kernel, memObjects);
    return -1;
  }
  ALGLogInfo<<"Succeed to init cl "<<cl_path<<" with kernel "<<cl_kernel_name;
  return 0;
}


int NmsCl::Forward(float* box, const int& box_nums, const float& iou_thr, std::vector<int>& keep_box){
//  std::cout<<"box num "<<box_nums<<std::endl;
  box_num = box_nums;
  col_num_64 = DIVUP(box_nums, ITEM_SIZE);//ITEM_SIZE位对齐


  size_t global_work_size[2] = { 0,0};//global work size need to exact division local work size

  //DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
  size_t              workgroup_size;
  size_t              local_work_size[2];
  clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &workgroup_size, NULL);
  {
    size_t  gsize[2];
    int     w;
//    std::cout<<"max work group size "<<workgroup_size<<std::endl;
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
    global_work_size[0] = global_work_size_0*gsize[0];//使大小为local的倍数,local size的最大值可使用info查看

    int global_work_size_1 = DIVUP(col_num_64, gsize[1]);
    global_work_size[1] = global_work_size_1*gsize[1];



  }

  // Create memory objects that will be used as arguments to
  // kernel.  First create host memory arrays that will be
  // used to store the arguments to the kernel
  if (!CreateMemObjects(context, memObjects, box, box_num,global_work_size[0],global_work_size[1]))
  {
    Cleanup(context, commandQueue, program, kernel, memObjects);
    ALGLogError << "Error kernel CreateMemObjects.";
    return -1;
  }
  cl_int errNum;
  // Set the kernel arguments (result, a, b)
  errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &memObjects[0]);

  errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &memObjects[1]);

  errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &memObjects[2]);
  errNum |= clSetKernelArg(kernel, 3, sizeof(cl_uint), &box_num); // 真实的box数量
  errNum |= clSetKernelArg(kernel, 4, sizeof(cl_uint), &col_num_64); // 每个item计算的iou数量
  errNum |= clSetKernelArg(kernel, 5, sizeof(cl_float), &iou_thr); // 每个item计算的iou数量

  if (errNum != CL_SUCCESS)
  {
    ALGLogError << "Error setting kernel arguments.";
    Cleanup(context, commandQueue, program, kernel, memObjects);
    return -1;
  }





//  std::cout<<"global size "<<global_work_size[0]<<" "<<global_work_size[1]<<std::endl;
//  std::cout<<"local size "<<local_work_size[0]<<" "<<local_work_size[1]<<std::endl;

  // Queue the kernel up for execution across the array
  errNum = clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL,
                                  global_work_size, local_work_size,
                                  0, NULL, NULL);
  if (errNum != CL_SUCCESS)
  {
    ALGLogError << "Error queuing kernel for execution.";
//    std::cout<<"error code "<<errNum<<std::endl;
    Cleanup(context, commandQueue, program, kernel, memObjects);

    return 1;
  }

  cl_ulong *result = (cl_ulong *)malloc(sizeof(cl_ulong)*global_work_size[0]*global_work_size[1]) ;

//  std::cout<<"start read buf "<<std::endl;
  // Read the output buffer back to the Host
  errNum = clEnqueueReadBuffer(commandQueue, memObjects[1], CL_TRUE,
                               0,   sizeof(cl_ulong)*box_num*col_num_64, result,
                               0, NULL, NULL);

  cl_int *result_idx = (cl_int *)malloc(sizeof(cl_int)*100) ;
  errNum |= clEnqueueReadBuffer(commandQueue, memObjects[2], CL_TRUE,
                                0,   sizeof(cl_int)*100, result_idx,
                                0, NULL, NULL);// 调试用

  if (errNum != CL_SUCCESS)
  {
    ALGLogError << "Error reading result buffer. "<<errNum << std::endl;
    Cleanup(context, commandQueue, program, kernel, memObjects);
    return 1;
  }

  //根据iou矩阵计算是否保留box
  keep_box=std::vector<int> (box_num,0);
  std::vector<cl_ulong> remv(box_num);
  for (int i = 0; i < box_num; i++) {
    int n_block = i/ITEM_SIZE;
    int in_block = i%ITEM_SIZE;
    if(!(remv[n_block]&1UL<<in_block)){//当前box在记录关系表中未被筛去
      keep_box[i] = 1; //保留当前box
      cl_ulong *p = result+i*col_num_64; //定位到i对应的iou行, 该行的所有参数为第i个box与其余box的iou值
      for(int j = n_block; j<col_num_64; j++){//将第i个box对应的iou关系与记录有已确定iou关系的数据做{或},以更新iou表.
        remv[j] |=p[j];// 该步骤的逻辑为一次更新i号box对应的 ITEM_SIZE个关系,与单个做{或}相比,速度快 1/ITEM_SIZE;
      }

    } else{
      keep_box[i] = -1; //剔除当前box
    }

  }

  free(result);
  free(result_idx);
  CleanMem(memObjects);
  return 0;

}

NmsCl::~NmsCl(){
}


void NmsCl::DeInit(){
  Cleanup(context, commandQueue, program, kernel, memObjects);
}

}

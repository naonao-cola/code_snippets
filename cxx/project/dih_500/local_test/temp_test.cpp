//
// Created by y on 2023/12/7.
//

//#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "CL/cl.h"

#include "rknn_api.h"
#include "DetectType.h"
#include "imgprocess.h"
#include "replace_std_string.h"
#include "event.h"
#include "temp_test.h"
#include "utils.h"
static uint8_t *Alg_ReadFile(const char *path,  uint32_t *len)
{
  FILE *fp = NULL;
  char filename[256];
  if(path == NULL)
  {
    return NULL;
  }
  memset(filename, 0, 256);
//  snprintf(filename, 256, "%s/%s.%s", path, keyword, suffix);
  snprintf(filename, 256, "%s", path);
  //DLOG(INFO, "sample process pos = %d x=%d y=%d process=%d (%d/%d)", ctx->pos, ctx->x, ctx->y, process, ctx->currentField, ctx->totalField);
  EVINFO(EVID_INFO, "Use rknn model path: %s", filename);
  fp = fopen(filename, "rb");
  if(fp == NULL)
  {
    printf("[alg/file] unfind:%s\r\n", filename);
    return NULL;
  }
  fseek(fp, 0, SEEK_END);
  uint32_t size = ftell(fp);
  if(fseek(fp, 0, SEEK_SET))
  {
    return NULL;
  }
  uint8_t *data = (uint8_t*)malloc(size);
  if (data == NULL)
  {
    return NULL;
  }
  if(size != fread(data, 1, size, fp))
  {
    free(data);
    return NULL;
  }
  if(len)
  {
    *len = size;
  }
  fclose(fp);
  return data;
}


static int NNet_MakeLabelsList(std::vector<std::string> &list, uint8_t *labels_data, uint32_t labels_size) {
  if (labels_data == NULL || labels_size == 0) {
    return -1;
  }
  for (uint32_t idx = 0; idx < labels_size; idx++) {
    if (labels_data[idx] == '\r' || labels_data[idx] == '\n') {
      labels_data[idx] = 0;
    }
  }
  for (uint32_t idx = 0; idx < labels_size; idx++) {
    if (labels_data[idx] != 0) {
      if (idx == 0 || labels_data[idx - 1] == 0) {
        std::string labels_name = (char *) ((long) labels_data + (long) idx);
        list.push_back(labels_name);
      }
    }
  }
  return 0;
}






//首先画poly看看
//input0:poly box, input1 conf
void PostProcessInclineRbc(float *input0, float  *input1,
                           std::vector<std::string> &labels_v,
                           std::vector<std::vector<int>>& poly_v,
                           std::list<NNetResult_t> &opt_v,
                           const float& conf_thr){
  //21504个anchor, 注该值随输入图像大小改变,调整模型输入图像大小时应当对该值进行调整
  //需要处理的类别idx
  std::vector<int> target_category_v{13,14};
  int anchor_nums = 21504;
  int box_cord_wh_nums = 8;//ppyoloe-r使用4个点,即8个xy保存poly坐标
  for(int a=0; a<anchor_nums; ++a){
    for(int c=0; c<target_category_v.size(); ++c){

      float box_conf = input1[target_category_v[c]*anchor_nums+a];
//      if(box_conf>0.02) {
//        std::cout << "box conf " << box_conf << std::endl;
//      }
      if(box_conf> conf_thr){
        std::cout<<"conf "<<box_conf <<std::endl;
        std::vector<int> one_poly;
        //放入坐标
        for(int xy=0; xy<box_cord_wh_nums; ++xy){
          std::cout<<" "<<input0[a*box_cord_wh_nums+xy];
          one_poly.push_back(int(input0[a*box_cord_wh_nums+xy]));
        }

        poly_v.push_back(one_poly);
      }
    //先看结果是否正常

    }
  }
}
//获取模型结果
void ProcessSegTemp(int8_t *input0, const int& category_nums,
                const int& img_height, const int& img_width,
                std::vector<cv::Mat>& result){
  for(int i=0; i<category_nums; ++i){
    cv::Mat one_result(img_height, img_width, CV_8SC1, input0+i*(img_height*img_width));
    result.push_back(one_result);
  }
}


//获取分割结果
void PostProcessSegRbc(int8_t *input0, const int& category_nums,
                       const int& img_height, const int& img_width,
                       cv::Mat& pred_mask){
  std::vector<cv::Mat> net_result;
  ProcessSegTemp(input0, category_nums, img_height, img_width, net_result);
  //模型含有0,1两个类别
  cv::Mat back_prob(net_result[0]);
  cv::Mat cell_prob(net_result[1]);
  pred_mask = cell_prob>back_prob;
  pred_mask.convertTo(pred_mask, CV_8U, 1);
}


//将数组中的点组织成成对的点
void ConstructPointTemp(const std::vector<int>& xy_v, std::vector<cv::Point>& point){
  if(xy_v.size()%2!=0) return;
  int point_num = xy_v.size()/2;
  for(int i =0; i<point_num; ++i){
    point.emplace_back(xy_v[i*2], xy_v[i*2+1]);
  }
}



using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::seconds;
using std::chrono::system_clock;



/*bool TestInclineNetwork(const std::string& model_path,
                        const std::string& model_label_path,
                        const std::string& img_path){
  // init
  uint32_t mod_size = 0;
  uint32_t label_size = 0;

  uint8_t *mod_data = Alg_ReadFile(model_path.c_str(),  &mod_size);
  uint8_t *label_data = Alg_ReadFile(model_label_path.c_str(), &label_size);

  std::vector<std::string> labels;
  NNet_MakeLabelsList(labels, label_data, label_size);

  std::cout<<"71"<<std::endl;
  int ret;
  rknn_context rknn_ctx;
  ret = rknn_init(&rknn_ctx, mod_data, mod_size, 0, NULL);
  if (ret < 0) {
    printf("rknn_init error ret=%d\r\n", ret);
    return -1;
  }
  std::cout<<"79"<<std::endl;
  rknn_input_output_num io_num;
  ret = rknn_query(rknn_ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  if (ret < 0) {
    printf("rknn_query ionum error ret=%d\r\n", ret);
    return -1;
  }
  std::cout<<"86"<<std::endl;
  rknn_tensor_attr input_attrs;
  input_attrs.index = 0;
  ret = rknn_query(rknn_ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs), sizeof(rknn_tensor_attr));
  if (ret < 0) {
    printf("rknn_query attr error ret=%d\r\n", ret);
    return -1;
  }

  int channel = 0;
  int width = 0;
  int height = 0;
  if (input_attrs.fmt == RKNN_TENSOR_NCHW) {
    channel = input_attrs.dims[1];
    height = input_attrs.dims[2];
    width = input_attrs.dims[3];
  } else {
    height = input_attrs.dims[1];
    width = input_attrs.dims[2];
    channel = input_attrs.dims[3];
  }
  std::cout<<"input fmt "<<input_attrs.fmt<<std::endl;
  std::vector<rknn_tensor_attr> output_attrs;
  for (int i = 0; i < io_num.n_output; i++) {
    rknn_tensor_attr attr;
    attr.index = i;
    ret = rknn_query(rknn_ctx, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(rknn_tensor_attr));
    output_attrs.push_back(attr);
  }

  int modelWidth = width;
  int modelHeight = height;
  std::cout<<"117"<<std::endl;
  //infer
  cv::Mat img = cv::imread(img_path);
  cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
  for(int iter=0; iter<10; ++iter){
    auto start = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
    std::cout<<"model h "<<modelWidth<<std::endl;
    cv::Mat ipt;
    ResizeImg(img, ipt, cv::Size(modelWidth, modelHeight),
              ResizeType::NORMAL, cv::Scalar(0, 0, 0), cv::INTER_LINEAR);
//    cv::resize(img, ipt, cv::Size(modelWidth, modelHeight));
    int ipt_width = ipt.rows;
    int ipt_height = ipt.cols;
    rknn_input inputs[1] = {0};
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = modelWidth * modelHeight * channel;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;
    inputs[0].buf = (void *) ipt.data;

    int netNumInput = io_num.n_input;
    int netNumOutput = io_num.n_output;

    ret = rknn_inputs_set(rknn_ctx, netNumInput, inputs);
    if (ret < 0) {
      printf("rknn_input_set fail! ret=%d\n", ret);
      return -1;
    }

    rknn_output *outputs = new rknn_output[netNumOutput];
    memset(outputs, 0, netNumOutput * sizeof(rknn_output));
    std::cout<<"146"<<std::endl;
    for (int i = 0; i < netNumOutput; i++) {
      outputs[i].want_float = true;
      outputs[i].is_prealloc = false;
    }

    ret = rknn_run(rknn_ctx, NULL);
    if(ret!=0){
      std::cout<<"run err "<<ret<<std::endl;
    }
    std::cout<<"out num "<<netNumOutput<<std::endl;
    ret = rknn_outputs_get(rknn_ctx, netNumOutput, outputs, NULL);
    if(ret!=0){
      std::cout<<"get result err "<<ret<<std::endl;
    }



    auto end =duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
    auto cost_time = end-start;
    std::cout<<"processing time: "<<cost_time<<std::endl;
    //模型已还原出原图大小的rect

    std::list<NNetResult_t> rect_v;
    std::vector<std::vector<int>> poly_v;
    std::list<NNetResult_t> opt_v;
    const float conf_thr = 0.5;

    std::cout<<"output size"<<outputs[0].size<<std::endl
              <<outputs[1].size<<std::endl
              <<std::endl;



    PostProcessInclineRbc((float *) outputs[0].buf, (float *) outputs[1].buf, labels,
                          poly_v, rect_v, conf_thr);

    auto end2 = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
    auto cost_time2 = end2-end;
    std::cout<<"post processing time: "<<cost_time2<<std::endl;

    cv::Mat temp_img(img.clone());

    for(const auto& one_poly:poly_v){
      std::vector<cv::Point> poly_cv_point_v;
      ConstructPointTemp(one_poly, poly_cv_point_v);
//      for(const auto& xy:one_poly){
//        std::cout<<xy<<std::endl;
//      }
      cv::polylines(temp_img, poly_cv_point_v,true,(0,0,255));
    }
    cv::cvtColor(temp_img, temp_img, cv::COLOR_RGB2BGR);
    SaveImage("./save_dir/"+std::to_string(start)+"2.bmp", temp_img);

    ret = rknn_outputs_release(rknn_ctx, netNumOutput, outputs);

    if (outputs) {
      delete (outputs);
    }


  }



}*/


/*bool TestSegNetwork(const std::string& model_path,
                        const std::string& model_label_path,
                        const std::string& img_path){
  // init
  uint32_t mod_size = 0;
  uint32_t label_size = 0;

  uint8_t *mod_data = Alg_ReadFile(model_path.c_str(),  &mod_size);
  uint8_t *label_data = Alg_ReadFile(model_label_path.c_str(), &label_size);

  std::vector<std::string> labels;
  NNet_MakeLabelsList(labels, label_data, label_size);

  std::cout<<"71"<<std::endl;
  int ret;
  rknn_context rknn_ctx;
  ret = rknn_init(&rknn_ctx, mod_data, mod_size, 0, NULL);
  if (ret < 0) {
    printf("rknn_init error ret=%d\r\n", ret);
    return -1;
  }
  std::cout<<"79"<<std::endl;
  rknn_input_output_num io_num;
  ret = rknn_query(rknn_ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  if (ret < 0) {
    printf("rknn_query ionum error ret=%d\r\n", ret);
    return -1;
  }
  std::cout<<"86"<<std::endl;
  rknn_tensor_attr input_attrs;
  input_attrs.index = 0;
  ret = rknn_query(rknn_ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs), sizeof(rknn_tensor_attr));
  if (ret < 0) {
    printf("rknn_query attr error ret=%d\r\n", ret);
    return -1;
  }

  int channel = 0;
  int width = 0;
  int height = 0;
  if (input_attrs.fmt == RKNN_TENSOR_NCHW) {
    channel = input_attrs.dims[1];
    height = input_attrs.dims[2];
    width = input_attrs.dims[3];
  } else {
    height = input_attrs.dims[1];
    width = input_attrs.dims[2];
    channel = input_attrs.dims[3];
  }
  std::cout<<"input fmt "<<input_attrs.fmt<<std::endl;
  std::vector<rknn_tensor_attr> output_attrs;
  for (int i = 0; i < io_num.n_output; i++) {
    rknn_tensor_attr attr;
    attr.index = i;
    ret = rknn_query(rknn_ctx, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(rknn_tensor_attr));
    output_attrs.push_back(attr);
  }

  int modelWidth = width;
  int modelHeight = height;
  std::cout<<"117"<<std::endl;
  //infer
  cv::Mat img = cv::imread(img_path);
  cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
  for(int iter=0; iter<10; ++iter){
    auto start = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
    std::cout<<"model h "<<modelWidth<<std::endl;
    cv::Mat ipt;
    ResizeImg(img, ipt, cv::Size(modelWidth, modelHeight),
              ResizeType::NORMAL, cv::Scalar(0, 0, 0), cv::INTER_LINEAR);
//    cv::resize(img, ipt, cv::Size(modelWidth, modelHeight));
    int ipt_width = ipt.rows;
    int ipt_height = ipt.cols;
    rknn_input inputs[1] = {0};
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = modelWidth * modelHeight * channel;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;
    inputs[0].buf = (void *) ipt.data;

    int netNumInput = io_num.n_input;
    int netNumOutput = io_num.n_output;

    ret = rknn_inputs_set(rknn_ctx, netNumInput, inputs);
    if (ret < 0) {
      printf("rknn_input_set fail! ret=%d\n", ret);
      return -1;
    }

    rknn_output *outputs = new rknn_output[netNumOutput];
    memset(outputs, 0, netNumOutput * sizeof(rknn_output));
    std::cout<<"146"<<std::endl;
    for (int i = 0; i < netNumOutput; i++) {
      outputs[i].want_float = false;
      outputs[i].is_prealloc = false;
    }

    ret = rknn_run(rknn_ctx, NULL);
    if(ret!=0){
      std::cout<<"run err "<<ret<<std::endl;
    }
    std::cout<<"out num "<<netNumOutput<<std::endl;
    ret = rknn_outputs_get(rknn_ctx, netNumOutput, outputs, NULL);
    if(ret!=0){
      std::cout<<"get result err "<<ret<<std::endl;
    }


    auto end =duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
    auto cost_time = end-start;
    std::cout<<"processing time: "<<cost_time<<std::endl;
    //模型已还原出原图大小的rect

    //后处理函数返回mask图
    cv::Mat pred_mask;

    PostProcessSegRbc((int8_t *) outputs[0].buf, 2, modelHeight, modelWidth, pred_mask);
    auto end2 = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
    auto cost_time2 = end2-end;
    std::cout<<"post processing time: "<<cost_time2<<std::endl;
    //view


    cv::Mat pred_merge;
    std::vector<cv::Mat> channels;

    cv::Mat img_b;
    cv::Mat img_g;
    cv::Mat img_r;
    split(img, channels);//分离色彩通道
    img_b = channels.at(0);
    img_g = channels.at(1);
    img_r = channels.at(2);

    cv::Mat ori_size_mask;
    cv::resize(pred_mask, ori_size_mask, cv::Size(img.cols, img.rows));

    img_r = img_r/2 + ori_size_mask*255/2;
    cv::Mat temp_img;
    cv::merge(std::vector<cv::Mat>{img_b, img_g, img_r}, temp_img);

//    std::cout<<"img type "<<temp_img.type()<<" " <<pred_mask.type()<<" "<<pred_merge.type()<<std::endl;

    cv::cvtColor(temp_img, temp_img, cv::COLOR_RGB2BGR);
    SaveImage("./save_dir/"+std::to_string(start)+"2.bmp", temp_img);

    ret = rknn_outputs_release(rknn_ctx, netNumOutput, outputs);

    if (outputs) {
      delete (outputs);
    }


  }



}*/
////////////////////
///cst
////////////////////
/**
* opencl kernel init callback for custom op
* */
//int relu_init_callback_gpu(rknn_custom_op_context* op_ctx,
//                           rknn_custom_op_tensor* inputs, uint32_t n_inputs,
//                           rknn_custom_op_tensor* outputs, uint32_t n_outputs)
//{
//  printf("relu_init_callback_gpu\n");
//  // 获取 opencl context
//  cl_context cl_ctx = (cl_context)op_ctx->gpu_ctx.cl_context;
//  // create tmp cl buffer
//  cl_mem* memObject = (cl_mem*)malloc(sizeof(cl_mem) * 2);
//  memObject[0]
//      = clCreateBuffer(cl_ctx, CL_MEM_READ_WRITE,
//                       inputs[0].attr.size, NULL, NULL);
//  memObject[1]
//      = clCreateBuffer(cl_ctx, CL_MEM_READ_WRITE,
//                       outputs[0].attr.size, NULL, NULL);
//  op_ctx->priv_data = memObject;
//  return 0;
//}

/*bool TestSegCstNetwork(const std::string& model_path,
                    const std::string& model_label_path,
                    const std::string& img_path){
  // init
  uint32_t mod_size = 0;
  uint32_t label_size = 0;

  uint8_t *mod_data = Alg_ReadFile(model_path.c_str(),  &mod_size);
  uint8_t *label_data = Alg_ReadFile(model_label_path.c_str(), &label_size);

  std::vector<std::string> labels;
  NNet_MakeLabelsList(labels, label_data, label_size);

  std::cout<<"71"<<std::endl;
  int ret;
  rknn_context rknn_ctx;
  ret = rknn_init(&rknn_ctx, mod_data, mod_size, 0, NULL);
  if (ret < 0) {
    printf("rknn_init error ret=%d\r\n", ret);
    return -1;
  }
  std::cout<<"79"<<std::endl;
  rknn_input_output_num io_num;
  ret = rknn_query(rknn_ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  if (ret < 0) {
    printf("rknn_query ionum error ret=%d\r\n", ret);
    return -1;
  }
  std::cout<<"86"<<std::endl;
  rknn_tensor_attr input_attrs;
  input_attrs.index = 0;
  ret = rknn_query(rknn_ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs), sizeof(rknn_tensor_attr));
  if (ret < 0) {
    printf("rknn_query attr error ret=%d\r\n", ret);
    return -1;
  }

  int channel = 0;
  int width = 0;
  int height = 0;
  if (input_attrs.fmt == RKNN_TENSOR_NCHW) {
    channel = input_attrs.dims[1];
    height = input_attrs.dims[2];
    width = input_attrs.dims[3];
  } else {
    height = input_attrs.dims[1];
    width = input_attrs.dims[2];
    channel = input_attrs.dims[3];
  }
  std::cout<<"input fmt "<<input_attrs.fmt<<std::endl;
  std::vector<rknn_tensor_attr> output_attrs;
  for (int i = 0; i < io_num.n_output; i++) {
    rknn_tensor_attr attr;
    attr.index = i;
    ret = rknn_query(rknn_ctx, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(rknn_tensor_attr));
    output_attrs.push_back(attr);
  }

  int modelWidth = width;
  int modelHeight = height;
  std::cout<<"117"<<std::endl;
  //infer
  cv::Mat img = cv::imread(img_path);
  cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
  for(int iter=0; iter<10; ++iter){
    auto start = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
    std::cout<<"model h "<<modelWidth<<std::endl;
    cv::Mat ipt;
    ResizeImg(img, ipt, cv::Size(modelWidth, modelHeight),
              ResizeType::NORMAL, cv::Scalar(0, 0, 0), cv::INTER_LINEAR);
    //    cv::resize(img, ipt, cv::Size(modelWidth, modelHeight));
    int ipt_width = ipt.rows;
    int ipt_height = ipt.cols;
    rknn_input inputs[1] = {0};
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = modelWidth * modelHeight * channel;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;
    inputs[0].buf = (void *) ipt.data;

    int netNumInput = io_num.n_input;
    int netNumOutput = io_num.n_output;

    ret = rknn_inputs_set(rknn_ctx, netNumInput, inputs);
    if (ret < 0) {
      printf("rknn_input_set fail! ret=%d\n", ret);
      return -1;
    }

    rknn_output *outputs = new rknn_output[netNumOutput];
    memset(outputs, 0, netNumOutput * sizeof(rknn_output));
    std::cout<<"146"<<std::endl;
    for (int i = 0; i < netNumOutput; i++) {
      outputs[i].want_float = false;
      outputs[i].is_prealloc = false;
    }

    ret = rknn_run(rknn_ctx, NULL);
    if(ret!=0){
      std::cout<<"run err "<<ret<<std::endl;
    }
    std::cout<<"out num "<<netNumOutput<<std::endl;
    ret = rknn_outputs_get(rknn_ctx, netNumOutput, outputs, NULL);
    if(ret!=0){
      std::cout<<"get result err "<<ret<<std::endl;
    }


    auto end =duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
    auto cost_time = end-start;
    std::cout<<"processing time: "<<cost_time<<std::endl;
    //模型已还原出原图大小的rect

    //后处理函数返回mask图
    cv::Mat pred_mask;

    PostProcessSegRbc((int8_t *) outputs[0].buf, 2, modelHeight, modelWidth, pred_mask);
    auto end2 = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
    auto cost_time2 = end2-end;
    std::cout<<"post processing time: "<<cost_time2<<std::endl;
    //view


    cv::Mat pred_merge;
    std::vector<cv::Mat> channels;

    cv::Mat img_b;
    cv::Mat img_g;
    cv::Mat img_r;
    split(img, channels);//分离色彩通道
    img_b = channels.at(0);
    img_g = channels.at(1);
    img_r = channels.at(2);

    cv::Mat ori_size_mask;
    cv::resize(pred_mask, ori_size_mask, cv::Size(img.cols, img.rows));

    img_r = img_r/2 + ori_size_mask*255/2;
    cv::Mat temp_img;
    cv::merge(std::vector<cv::Mat>{img_b, img_g, img_r}, temp_img);

    //    std::cout<<"img type "<<temp_img.type()<<" " <<pred_mask.type()<<" "<<pred_merge.type()<<std::endl;

    cv::cvtColor(temp_img, temp_img, cv::COLOR_RGB2BGR);
    SaveImage("./save_dir/"+std::to_string(start)+"2.bmp", temp_img);

    ret = rknn_outputs_release(rknn_ctx, netNumOutput, outputs);

    if (outputs) {
      delete (outputs);
    }


  }



}*/

/*int TestClusterMatrix(){
  auto start = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();

  cv::Mat a = cv::Mat(10000,10000,CV_32FC2,cv::Scalar(0.7,0.5));
  cv::Mat b(a);
  cv::Mat c = cv::min(a,b);
  std::cout<<"c size "<<c.size()<<std::endl;
  std::cout<<"c dims "<<c.dims<<std::endl;
  cv::Mat d = cv::Mat(4, 10000,CV_32FC2,cv::Scalar(10,44));
  auto func_end =duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
  auto func_cost_time = func_end-start;
  std::cout<<"create time: "<<func_cost_time<<std::endl;
  for(int i=0;i<4;++i){
    d = d*c;
  }

  func_end =duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
  func_cost_time = func_end-start;
  std::cout<<"8 time: "<<func_cost_time<<std::endl;
  return 0;

}*/



//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Dan Ginsburg, Timothy Mattson
// ISBN-10:   ??????????
// ISBN-13:   ?????????????
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/??????????
//            http://www.????????.com
//

// OpenCLInfo.cpp
//
//    This is a simple example that demonstrates use of the clGetInfo* functions,
//    with particular focus on platforms and their associated devices.



///
//	main() for OpenCLInfo example
//
int TestOpenclInfo(){
  cl_context context = 0;

/*  displayInfo();*/

  return 0;
}






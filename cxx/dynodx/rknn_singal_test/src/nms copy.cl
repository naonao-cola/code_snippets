bool devIoU(const __global float  * a, const __global float * b,const int offset, const float threshold) {
  // 计算两个bbox的iou
  float left = max(a[0], b[0]), right = min(a[2], b[2]);
  float top = max(a[1], b[1]), bottom = min(a[3], b[3]);
  float width = max(right - left + offset, 0.f),
        height = max(bottom - top + offset, 0.f);
  float interS = width * height;
  float Sa = (a[2] - a[0] + offset) * (a[3] - a[1] + offset);
  float Sb = (b[2] - b[0] + offset) * (b[3] - b[1] + offset);
  return interS > threshold * (Sa + Sb - interS);
}
constant int nums_per_thread = sizeof(unsigned long )* 8;
__kernel  void nms_cl(__global const float *dev_boxes,
                     __global ulong *mask,
                     __global int * mask_idx,
                     const int box_nums,
                     const int col_num_64,
                     const float iou_thr) {
  int g_x = get_global_id(0);
  int g_y = get_global_id(1);
  if(!(g_x<box_nums)||!(g_y<col_num_64)){//当数据为pad时,跳出,行与列需要是local的整数倍
    //mask_idx[10] = 999;
    return;
  }

  const  __global float * cur_box = dev_boxes+ g_x * 4;
  int g_y_reverse_idx = g_y*nums_per_thread;// 利用当前合并后的索引,计算对应dev_boxes索引

  int tar_idx = 0;
  int g_x_work_size = get_global_size(0);
  int g_y_work_size = get_global_size(1);

  


  unsigned long  t = 0;
  //记录cur box与 g_y_reverse_idx 之后64个box的iou
  for(int i =0; i<nums_per_thread;++i){//每个item计算 iter_num个,
    tar_idx = g_y_reverse_idx + i;
    if(tar_idx<box_nums){// 去除尾部pad的部分
      const __global float * tar_box = dev_boxes+tar_idx*4 ;
      bool iou_threshed = devIoU(cur_box, tar_box, 1, iou_thr);
      if(iou_threshed){
        t |= 1UL << i;
      }
    }else{
       t |= 1UL << i;
    }
  }

  int mask_write_idx = g_x*col_num_64+g_y;
  //mask_idx[mask_write_idx] = box_nums; //调试用,暂留
  mask[mask_write_idx] = t;


}
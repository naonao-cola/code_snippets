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

#define BOX_DIM 4          // 每个 box 的 float4 维度
#define WG_SIZE_X 64       // work-group 中 x 方向的线程数（可调）
#define WG_SIZE_Y 1        // work-group 中 y 方向的线程数（可调）


constant int nums_per_thread = sizeof(unsigned long )* 8;
__kernel  void nms_cl(__global const float *dev_boxes,
                     __global ulong *mask,
                     __global int * mask_idx,
                     const int box_nums,
                     const int col_num_64,
                     const float iou_thr)
{

  // 本地共享内存：缓存当前 tile 内所有 box
    __local float local_boxes[WG_SIZE_X * BOX_DIM];

    int g_x = get_global_id(0);   // 当前处理的 box 行
    int g_y = get_global_id(1);   // 当前处理的 64-box 列
    int l_x = get_local_id(0);    // 当前线程在 work-group 内的 x 坐标

    if (!(g_x < box_nums) || !(g_y < col_num_64)) return;

    // 当前 box 的地址（global）
    __global const float *cur_box = dev_boxes + g_x * BOX_DIM;

    // 将当前 tile 的 box 数据读入共享内存
    // 注意：我们只需缓存当前 tile 内 g_x 所在的行即可，因为每个线程只处理自己的 g_x
    // 但为了避免 bank conflict，采用顺序写 local memory
    if (l_x < WG_SIZE_X) {
        int tile_start_x = (g_x / WG_SIZE_X) * WG_SIZE_X;
        int box_idx = tile_start_x + l_x;
        if (box_idx < box_nums) {
            for (int d = 0; d < BOX_DIM; ++d) {
                local_boxes[l_x * BOX_DIM + d] = dev_boxes[box_idx * BOX_DIM + d];
            }
        } else {
            // pad 处理
            for (int d = 0; d < BOX_DIM; ++d) {
                local_boxes[l_x * BOX_DIM + d] = 0.0f;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);


  // 当前 tile 内 cur_box 的本地副本
    float cur_box_local[4];
    for (int d = 0; d < 4; ++d) {
        cur_box_local[d] = cur_box[d];
    }

    int g_y_reverse_idx = g_y * nums_per_thread;
    unsigned long t = 0;

    for (int i = 0; i < nums_per_thread; ++i) {
        int tar_idx = g_y_reverse_idx + i;
        if (tar_idx < box_nums) {
            // 从共享内存读取目标 box（注意 tile 内偏移）
            int local_tar_idx = tar_idx % WG_SIZE_X;
            __local const float *tar_box = local_boxes + local_tar_idx * BOX_DIM;

            // 计算 IOU
            float left = max(cur_box_local[0], tar_box[0]);
            float right = min(cur_box_local[2], tar_box[2]);
            float top = max(cur_box_local[1], tar_box[1]);
            float bottom = min(cur_box_local[3], tar_box[3]);

            float width = max(right - left + 1.0f, 0.0f);
            float height = max(bottom - top + 1.0f, 0.0f);
            float interS = width * height;

            float Sa = (cur_box_local[2] - cur_box_local[0] + 1.0f) *
                       (cur_box_local[3] - cur_box_local[1] + 1.0f);
            float Sb = (tar_box[2] - tar_box[0] + 1.0f) *
                       (tar_box[3] - tar_box[1] + 1.0f);

            bool iou_threshed = interS > iou_thr * (Sa + Sb - interS);
            if (iou_threshed) {
                t |= 1UL << i;
            }
        } else {
            t |= 1UL << i;
        }
    }

    int mask_write_idx = g_x * col_num_64 + g_y;
    mask[mask_write_idx] = t;
}
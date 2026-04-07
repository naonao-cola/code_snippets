
#include "yolo_pose.hpp"
#include <common/cuda_tools.hpp>

namespace YoloPose
{

__device__ int         NUM_BOX_ELEMENT;   // left, top, right, bottom, confidence, keepflag, (x, y, conf) * 17
static __device__ void affine_project(float* matrix, float x, float y, float* ox, float* oy)
{
    *ox = matrix[0] * x + matrix[1] * y + matrix[2];
    *oy = matrix[3] * x + matrix[4] * y + matrix[5];
}

static __global__ void decode_kernel_v8_Pose(float* predict,
                                             int    num_bboxes,
                                             float  confidence_threshold,
                                             float* invert_affine_matrix,
                                             float* parray,
                                             int    MAX_IMAGE_BOXES,
                                             int    num_key_points,
                                             int    num_cls )
{
    NUM_BOX_ELEMENT = 4 + 1 + 1 + 1 + 3 * num_key_points;   // left, top, right, bottom, cls, confidence, keepflag, (x, y, conf) * 17
    int position    = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= num_bboxes)
        return;

    float* pitem = predict + (4 + num_cls + 3 * num_key_points) * position;

    float cx     = pitem[0];
    float cy     = pitem[1];
    float width  = pitem[2];
    float height = pitem[3];
    // 增加修改
    float max_confidence = 0.f;
    int   best_class     = -1;
    for (int c = 0; c < num_cls; c++) {
        float temp_conf = pitem[4+c];
        if (temp_conf >= max_confidence && temp_conf >= confidence_threshold) {
            max_confidence = temp_conf;
            best_class     = c;
        }
    }
    if (best_class == -1) {
        return;
    }

    int index = atomicAdd(parray, 1);
    if (index >= MAX_IMAGE_BOXES)
        return;

    float left   = cx - width * 0.5f;
    float top    = cy - height * 0.5f;
    float right  = cx + width * 0.5f;
    float bottom = cy + height * 0.5f;
    affine_project(invert_affine_matrix, left, top, &left, &top);
    affine_project(invert_affine_matrix, right, bottom, &right, &bottom);

    float* pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
    *pout_item++     = left;
    *pout_item++     = top;
    *pout_item++     = right;
    *pout_item++     = bottom;
    *pout_item++     = best_class;
    *pout_item++     = max_confidence;
    *pout_item++     = 1;   // 1 = keep, 0 = ignore

    for (int i = 0; i < num_key_points; ++i) {
        float keypoint_x          = pitem[4+num_cls+3*i];
        float keypoint_y          = pitem[4+num_cls+3*i+1];
        float keypoint_confidence = pitem[4+num_cls+3*i+2];

        affine_project(invert_affine_matrix, keypoint_x, keypoint_y, &keypoint_x, &keypoint_y);

        *pout_item++ = keypoint_x;
        *pout_item++ = keypoint_y;
        *pout_item++ = keypoint_confidence;
    }
}

static __device__ float box_iou(float aleft, float atop, float aright, float abottom, float bleft, float btop, float bright, float bbottom)
{

    float cleft   = max(aleft, bleft);
    float ctop    = max(atop, btop);
    float cright  = min(aright, bright);
    float cbottom = min(abottom, bbottom);

    float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
    if (c_area == 0.0f)
        return 0.0f;

    float a_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop);
    float b_area = max(0.0f, bright - bleft) * max(0.0f, bbottom - btop);
    return c_area / (a_area + b_area - c_area);
}

static __global__ void nms_kernel_v8_Pose(float* bboxes, int max_objects, float threshold)
{

    int position = (blockDim.x * blockIdx.x + threadIdx.x);
    int count    = min((int)*bboxes, max_objects);
    if (position >= count)
        return;

    //  left, top, right, bottom, cls, confidence, keepflag, (x, y, conf) * 17
    float* pcurrent = bboxes + 1 + position * NUM_BOX_ELEMENT;
    for (int i = 0; i < count; ++i) {
        float* pitem = bboxes + 1 + i * NUM_BOX_ELEMENT;
        if (i == position)
            continue;

        if (pitem[5] >= pcurrent[5]) {
            if (pitem[5] == pcurrent[5] && i < position)
                continue;

            float iou = box_iou(pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3], pitem[0], pitem[1], pitem[2], pitem[3]);

            if (iou > threshold) {
                pcurrent[6] = 0;   // 1=keep, 0=ignore
                return;
            }
        }
    }
}

void decode_kernel_invoker(float*       predict,
                           int          num_bboxes,
                           float        confidence_threshold,
                           float*       invert_affine_matrix,
                           float*       parray,
                           int          max_objects,
                           int          num_keypoints,
                           int          num_cls,
                           cudaStream_t stream)
{

    auto grid  = CUDATools::grid_dims(num_bboxes);
    auto block = CUDATools::block_dims(num_bboxes);
    checkCudaKernel(decode_kernel_v8_Pose<<<grid, block, 0, stream>>>(
        predict, num_bboxes, confidence_threshold, invert_affine_matrix, parray, max_objects, num_keypoints, num_cls));
}

void nms_kernel_invoker(float* parray, float nms_threshold, int max_objects, cudaStream_t stream)
{

    auto grid  = CUDATools::grid_dims(max_objects);
    auto block = CUDATools::block_dims(max_objects);
    checkCudaKernel(nms_kernel_v8_Pose<<<grid, block, 0, stream>>>(parray, max_objects, nms_threshold));
}
};   // namespace YoloPose
//
// Created by y on 24-9-29.
//


#ifndef TEST_LIBALG_NMSCL_H
#define TEST_LIBALG_NMSCL_H
#ifndef __CL_ENABLE_EXCEPTIONS
#define __CL_ENABLE_EXCEPTIONS
#endif
#include <CL/cl.h>
#include <iostream>
#include <vector>

namespace ALG_CL
{
class NmsCl
{
public:
    NmsCl() = default;
    /*!
     * opencl 初始化接口
     * @param cl_path cl文件路径
     * @return
     */
    int Init(const std::string& cl_path);
    /*!
     *
     * @param box         指向box的指针,box依次存放left,top,right,bottom
     * @param box_nums    box数量
     * @param [out] keep  表明该box是否被iou虑去,-1 for 是, 1 for 否
     * @return
     */
    int  Forward(float* box, const int& box_nums, const float& iou_thr, std::vector<int>& keep);
    void DeInit();
    ~NmsCl();

private:
    unsigned int     box_num        = 0;
    unsigned int     col_num_64     = 0;
    cl_context       context        = 0;
    cl_command_queue commandQueue   = 0;
    cl_program       program        = 0;
    cl_device_id     device         = 0;
    cl_kernel        kernel         = 0;
    cl_mem           memObjects[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::string      cl_kernel_name = "nms_cl";   // 加载的核函数名称
    const cl_uint    ITEM_SIZE      = sizeof(cl_ulong) * 8;
};
}   // namespace ALG_CL



#endif   // TEST_LIBALG_NMSCL_H

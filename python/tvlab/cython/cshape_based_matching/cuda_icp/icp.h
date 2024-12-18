#pragma once

#include "geometry.h"

#ifdef CUDA_ON
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#endif

#include "scene/edge_scene/edge_scene.h"

namespace cuda_icp {

// use custom mat/vec here, otherwise we have to mix eigen with cuda
// then we may face some error due to eigen vesrion
//class defination refer to open3d
struct RegistrationResult
{
    __device__ __host__
    RegistrationResult(const Mat3x3f &transformation =
            Mat3x3f::identity()) : transformation_(transformation),
            inlier_rmse_(0.0), fitness_(0.0) {}

    Mat3x3f transformation_;
    float inlier_rmse_;
    float fitness_;
};

struct ICPConvergenceCriteria
{
public:
    __device__ __host__
    ICPConvergenceCriteria(float relative_fitness = 1e-3f,
            float relative_rmse = 1e-3f, int max_iteration = 30) :
            relative_fitness_(relative_fitness), relative_rmse_(relative_rmse),
            max_iteration_(max_iteration) {}

    float relative_fitness_;
    float relative_rmse_;
    int max_iteration_;
};

// to be used by icp cuda & cpu
// in this way we can avoid eigen mixed with cuda
Mat3x3f eigen_slover_444(float* A, float* b);


template <class Scene>
RegistrationResult ICP2D_Point2Plane_cpu(std::vector<Vec2f>& model_pcd,
        const Scene scene,
        const ICPConvergenceCriteria criteria = ICPConvergenceCriteria());

extern template RegistrationResult ICP2D_Point2Plane_cpu(std::vector<Vec2f> &model_pcd, const Scene_edge scene,
const ICPConvergenceCriteria criteria);

#ifdef CUDA_ON
template<class Scene>
RegistrationResult ICP2D_Point2Plane_cuda(device_vector_holder<Vec2f> &model_pcd, const Scene scene,
                                        const ICPConvergenceCriteria criteria = ICPConvergenceCriteria());

extern template RegistrationResult ICP2D_Point2Plane_cuda(device_vector_holder<Vec2f> &model_pcd, const Scene_edge scene,
const ICPConvergenceCriteria criteria);

#endif


/// !!!!!!!!!!!!!!!!!! low level

typedef vec<16,  float> Vec16f;
// tight: A(symetric 4x4 --> (16-4)/2+4) + ATb 4 + mse(b*b 1) + count 1 = 16

template<class Scene>
struct thrust__pcd2Ab
{
    Scene __scene;

    __host__ __device__
    thrust__pcd2Ab(Scene scene): __scene(scene){

    }

    __host__ __device__ Vec16f operator()(const Vec2f &src_pcd) const {
        Vec16f result;
        Vec2f dst_pcd, dst_normal; bool valid;
        __scene.query(src_pcd, dst_pcd, dst_normal, valid);
        if(!valid) return result;
        else{
            result[15] = 1;  //valid count
            // dot
            float b_temp = (dst_pcd - src_pcd).x * dst_normal.x +
                          (dst_pcd - src_pcd).y * dst_normal.y;
            result[14] = b_temp*b_temp; // mse

            // cross
            float A_temp[4];
            A_temp[0] = dst_normal.y*src_pcd.x - dst_normal.x*src_pcd.y;

            A_temp[1] = dst_normal.x;
            A_temp[2] = dst_normal.y;
            A_temp[3] = src_pcd.x*dst_normal.x + src_pcd.y*dst_normal.y;

            // ATA lower
            // 0  x  x  x
            // 1  4  x  x
            // 2  5  7  x
            // 3  6  8  9
            result[ 0] = A_temp[0] * A_temp[0];
            result[ 1] = A_temp[0] * A_temp[1];
            result[ 2] = A_temp[0] * A_temp[2];
            result[ 3] = A_temp[0] * A_temp[3];

            result[ 4] = A_temp[1] * A_temp[1];
            result[ 5] = A_temp[1] * A_temp[2];
            result[ 6] = A_temp[1] * A_temp[3];

            result[ 7] = A_temp[2] * A_temp[2];
            result[ 8] = A_temp[2] * A_temp[3];

            result[ 9] = A_temp[3] * A_temp[3];

            // ATb
            result[10] = A_temp[0] * b_temp;
            result[11] = A_temp[1] * b_temp;
            result[12] = A_temp[2] * b_temp;
            result[13] = A_temp[3] * b_temp;

            return result;
        }
    }
};

struct thrust__plus{
    __host__ __device__ Vec16f operator()(const Vec16f &in1, const Vec16f &in2) const{
        return in1 + in2;
    }
};

}



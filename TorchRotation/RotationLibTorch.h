// write the rotation operation with libtorch directly.
#pragma once
#ifdef _MSC_VER
#pragma warning(disable:4624)
#pragma warning(disable:4067)
#endif

#include <torch/nn.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#if 0
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cusolver_common.h>
#endif

// #include "../Common/Common.h"
#define BUILD_ROTATION_AS_PYBIND11 1

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS 1
#endif

namespace RotationLibTorch
{
    torch::Tensor quat_multiply(const torch::Tensor q1, const torch::Tensor q2);
    torch::Tensor quat_apply(const torch::Tensor q, const torch::Tensor v);

    torch::Tensor quat_inv(const torch::Tensor q);
    torch::Tensor flip_quat_by_w(const torch::Tensor q);
    torch::Tensor quat_normalize(const torch::Tensor q);
    torch::Tensor quat_integrate(const torch::Tensor q, const torch::Tensor omega, float dt);
    torch::Tensor calc_omega(const torch::Tensor q0, const torch::Tensor q1, float dt);
    
    torch::Tensor quat_to_angle(torch::Tensor q);

    torch::Tensor quat_to_rotvec(torch::Tensor q);
    torch::Tensor quat_from_rotvec(torch::Tensor x);

    torch::Tensor quat_from_matrix(torch::Tensor matrix);
    torch::Tensor quat_to_matrix(torch::Tensor q);

    torch::Tensor quat_from_vec6d(torch::Tensor x);
    torch::Tensor quat_to_vec6d(torch::Tensor x);

    std::vector<torch::Tensor> normalize_vec6d(torch::Tensor x);
    torch::Tensor normalize_vec6d_cat(torch::Tensor x);
    torch::Tensor vec6d_to_matrix(torch::Tensor x);
    
    torch::Tensor rotvec_to_vec6d(torch::Tensor x);
    
    torch::Tensor vector_to_cross_matrix(torch::Tensor x);
    torch::Tensor matrix_to_angle(torch::Tensor x);
    torch::Tensor rotation_matrix_inv(torch::Tensor x);
    torch::Tensor mat22_det(torch::Tensor x);
    torch::Tensor mat33_det(torch::Tensor x);
    torch::Tensor mat33_svd(torch::Tensor x);
    torch::Tensor mat44_det(torch::Tensor x);

    torch::Tensor x_decompose(torch::Tensor q);
    torch::Tensor y_decompose(torch::Tensor q);
    torch::Tensor z_decompose(torch::Tensor q);

    class NormalizeVec6dLayer: public torch::nn::Module
    {
        public:
            torch::Tensor forward(torch::Tensor x);
    };

    #define CHECK_CUDA(x)                                                          \
      do {                                                                         \
        TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor");                     \
      } while (0)

    #define CHECK_IS_FLOAT(x)                                                      \
      do {                                                                         \
        TORCH_CHECK((x.scalar_type() == at::ScalarType::Float) ||                  \
                        (x.scalar_type() == at::ScalarType::Half) ||               \
                        (x.scalar_type() == at::ScalarType::Double),               \
                    #x " must be a double, float or half tensor");                 \
      } while (0)

    #if 0
    template <int success = CUSOLVER_STATUS_SUCCESS, class T,
        class Status> // , class A = Status(*)(P), class D = Status(*)(T)>
    std::unique_ptr<T, Status(*)(T*)> unique_allocate(Status(allocator)(T**),
        Status(deleter)(T*)) {
        T* ptr;
        auto stat = allocator(&ptr);
        TORCH_CHECK(stat == success);
        return { ptr, deleter };
    }

    template <class T>
    std::unique_ptr<T, decltype(&cudaFree)> unique_cuda_ptr(size_t len) {
        T* ptr;
        auto stat = cudaMalloc(&ptr, sizeof(T) * len);
        TORCH_CHECK(stat == cudaSuccess);
        return { ptr, cudaFree };
    }
    #endif
}

#include "RotationLibTorch.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/autograd.h>

#ifndef _DEBUG_AS_MAIN_FILE
#include <pybind11/pybind11.h>
#endif

// #include "Network.h"
#ifdef _WIN32
#pragma warning(disable:4624)
#endif

namespace RotationLibTorch
{
    using namespace torch::indexing;

    // quaternion multiply
    torch::Tensor quat_multiply(const torch::Tensor p, const torch::Tensor q)
    {
        if (q.size(-1) != 4 || p.size(-1) != 4)
        {
            throw std::length_error(std::string(__func__) + ": the shape of input quaternion should be (..., 4)");
        }
        torch::Tensor pw = p.index({Ellipsis, Slice(3, 4)});
        torch::Tensor qw = q.index({Ellipsis, Slice(3, 4)});
        torch::Tensor pxyz = p.index({Ellipsis, Slice(0, 3)});
        torch::Tensor qxyz = q.index({Ellipsis, Slice(0, 3)});
        torch::Tensor w = pw * qw - torch::sum(pxyz * qxyz, -1, true);
        torch::Tensor xyz = pw * qxyz + qw * pxyz + torch::cross(pxyz, qxyz, -1);
        torch::Tensor result = torch::cat({xyz, w}, -1);
        return result;
    }

    // apply quaternion rotation to vectors
    torch::Tensor quat_apply(const torch::Tensor q, const torch::Tensor v)
    {
        if (q.size(-1) != 4 || v.size(-1) != 3)
        {
            throw std::length_error(std::string(__func__) + ": the shape of input quaternion should be (..., 4) ");
        }
        torch::Tensor qxyz = q.index({Ellipsis, Slice(0, 3)});
        torch::Tensor t = 2 * torch::cross(qxyz, v, -1);
        torch::Tensor xyz = v + q.index({Ellipsis, Slice(3, 4)}) * t + torch::cross(qxyz, t, -1);
        return xyz;
    }

    // The inverse of quaternion
    torch::Tensor quat_inv(const torch::Tensor q)
    {
        if (q.size(-1) != 4)
        {
            throw std::length_error(std::string(__func__) + ": the shape of input quaternion should be (..., 4)");
        }
        torch::Tensor w = q.index({Ellipsis, Slice(3, 4)});
        torch::Tensor xyz = q.index({Ellipsis, Slice(0, 3)});
        torch::Tensor result = torch::cat({xyz, -w}, -1);
        return result;
    }

    torch::Tensor flip_quat_by_w(const torch::Tensor q)
    {
        if (q.size(-1) != 4)
        {
            throw std::length_error(std::string(__func__) + ": the shape of input quaternion should be (..., 4)");
        }
        torch::Tensor w = q.index({Ellipsis, torch::indexing::Slice(3, 4)});
        torch::Tensor mask = (w < 0).to(torch::kInt32);
        mask.masked_fill_(mask == 1, -1);
        mask.masked_fill_(mask == 0, 1);
        torch::Tensor result = q * mask;
        return result;
    }

    torch::Tensor flip_quat_by_dot(const torch::Tensor q0, torch::Tensor q1)
    {
        torch::Tensor mask;
        {
            torch::NoGradGuard guard;
            torch::Tensor dot_value = torch::where(torch::sum(q0.view({ -1, 4 }) * q1.view({ -1, 4 }), -1) < 0)[0];
            mask = torch::ones({ q0.numel() / 4, 1 }, q0.options());
            mask.index({ dot_value }) = -1;
        }
        return (q1.view({ -1, 4 }) * mask).view_as(q1);
    }

    torch::Tensor vec_normalize(torch::Tensor q)
    {
        torch::Tensor length = torch::linalg::norm(q, 2, -1, true, c10::nullopt);
        torch::Tensor result = q / length;
        return result;
    }

    // normalize quaternion
    torch::Tensor quat_normalize(const torch::Tensor q)
    {
        if (q.size(-1) != 4)
        {
            throw std::length_error(std::string(__func__) + ": the shape of input quaternion should be (..., 4)");
        }
        torch::Tensor length = torch::linalg::norm(q, 2, -1, true, c10::nullopt);
        torch::Tensor result = q / length;
        return result;
    }

    // quaternion integrate
    torch::Tensor quat_integrate(const torch::Tensor q, const torch::Tensor omega, float dt)
    {
        if (q.size(-1) != 4 || omega.size(-1) !=3)
        {
            throw std::length_error(std::string(__func__) + ": the shape of input quaternion should be (..., 4), omega should be (..., 3)");
        }
        auto option = q.options();
        auto sizes = omega.sizes().vec();
        sizes[sizes.size() - 1] = 1;
        auto dw = torch::cat({ omega, torch::zeros(sizes, option) }, -1);
        torch::Tensor dq = quat_multiply(dw, q);
        torch::Tensor result = quat_normalize(q + (0.5 * dt) * dq);
        return result;
    }

    torch::Tensor calc_omega(const torch::Tensor q0, const torch::Tensor q1_, float dt)
    {
        auto qd = (flip_quat_by_dot(q0, q1_) - q0) / dt;

        auto mask = torch::full_like(q0, -1);
        mask.index({ Ellipsis, 3}) = 1;
        auto q_conj = q0 * mask;

        auto qw = quat_multiply(qd, q_conj);
        auto omega = 2.0f * qw.index({ Ellipsis, Slice(0, 3) });
        return omega;
    }

    // Rotation from vector a to vector b
    torch::Tensor quat_between(torch::Tensor a, torch::Tensor b)
    {
        if (a.size(-1) != 3 || b.size(-1) != 3)
        {
            throw std::length_error(std::string(__func__) + ": the shape of input vector should be (..., 3)");
        }
        torch::Tensor cross_res = torch::cross(a, b, -1);
        torch::Tensor w_ = torch::sqrt((a * a).sum(-1) * (b * b).sum(-1)) + (a * b).sum(-1);
        torch::Tensor res_ = torch::cat({ cross_res, w_.index({Ellipsis, None})}, -1);
        return vec_normalize(res_);
    }

    torch::Tensor decompose_rotation(torch::Tensor q, torch::Tensor vb)
    {
        bool is_flatten = q.ndimension() == 1;
        if (is_flatten)
        {
            q = q.index({ None });
        }
        auto q_shape = q.sizes().vec();
        if (*q_shape.rbegin() != 4)
        {
            throw std::length_error(std::string(__func__) + ": the shape of input quaternion should be (..., 4)");
        }
        *q_shape.rbegin() = 3;
        if (!q.is_contiguous()) q = q.contiguous();
        vb = torch::broadcast_to(vb, q_shape);
        torch::Tensor va = vec_normalize(quat_apply(q, vb));
        torch::Tensor tmp = quat_between(va, vb);
        torch::Tensor result = quat_normalize(quat_multiply(tmp, q));
        if (is_flatten)
        {
            result = result.reshape({ 4 });
        }
        return result;
    }

    torch::Tensor x_decompose(torch::Tensor q)
    {
        torch::Tensor vb = torch::zeros({ 1, 3 }, q.options());
        vb.set_requires_grad(false);
        vb.index({0, 0}) = 1;
        return decompose_rotation(q, vb);
    }

    torch::Tensor y_decompose(torch::Tensor q)
    {
        torch::Tensor vb = torch::zeros({ 1, 3 }, q.options());
        vb.set_requires_grad(false);
        vb.index({ 0, 1 }) = 1;
        return decompose_rotation(q, vb);
    }

    torch::Tensor z_decompose(torch::Tensor q)
    {
        torch::Tensor vb = torch::zeros({ 1, 3 }, q.options());
        vb.set_requires_grad(false);
        vb.index({ 0, 2 }) = 1;
        return decompose_rotation(q, vb);
    }

    // convert quaternion rotation to angle
    torch::Tensor quat_to_angle(torch::Tensor q)
    {
        torch::Tensor result;
        return result;
    }

    // convert quaternion to axis angle format
    torch::Tensor quat_to_rotvec(torch::Tensor q)
    {
        if (q.size(-1) != 4)
        {
            throw std::length_error(std::string(__func__) + "size of q should be 4");
        }
        // auto option = q.options();
        const float eps = 1e-3f;
        q = flip_quat_by_w(q);
        torch::Tensor xyz = q.index({Ellipsis, Slice(0, 3)});
        torch::Tensor w = q.index({Ellipsis, 3});
        torch::Tensor xyz_norm = torch::linalg::norm(xyz, 2, -1, false, c10::nullopt);
        torch::Tensor angle = 2 * torch::atan2(xyz_norm, w);
        torch::Tensor small_angle = angle <= eps;
        torch::Tensor scale_small = 2 + (1.0 / 12) * torch::pow(angle, 2) + (7.0 / 2880) * torch::pow(angle, 4);
        torch::Tensor scale_large = angle / torch::sin(0.5 * angle);
        torch::Tensor scale = torch::where(small_angle, scale_small, scale_large);
        torch::Tensor result = scale.index({Ellipsis, None}) * xyz;
        return result;
    }

    // build quaternion from axis angle format
    torch::Tensor quat_from_rotvec(const torch::Tensor x)
    {
        if (x.size(-1) != 3)
        {
            throw std::length_error(std::string(__func__) + "shape should be (*, 3)");
        }
        torch::Tensor norms = torch::linalg::norm(x, 2, -1, false, c10::nullopt);
        torch::Tensor small_angle = norms <= 1e-3f;
        torch::Tensor scale_small = 0.5 - (1.0 / 48) * torch::square(norms) + torch::pow(norms, 4) / 3840;
        torch::Tensor scale_large = torch::sin(0.5 * norms) / norms;
        torch::Tensor scale = torch::where(small_angle, scale_small, scale_large);
        torch::Tensor quat_xyz = scale.index({Ellipsis, None}) * x;
        torch::Tensor quat_w = torch::cos(0.5 * norms).index({Ellipsis, None});
        torch::Tensor quat = torch::cat({quat_xyz, quat_w}, -1);

        return quat;
    }

    torch::Tensor _sqrt_positive_part(torch::Tensor x)
    {
        auto ret = torch::zeros_like(x);
        auto pos_mask = x > 0;
        ret.index_put_({ pos_mask }, torch::sqrt(x.index({ pos_mask })));
        return ret;
    }

    torch::Tensor quat_from_matrix(torch::Tensor matrix)
    {
        // get the code from pytorch3d\transforms\rotation_conversions.py in PyTorch3D
        if (matrix.size(-1) != 3 || matrix.size(-2) != 3)
        {
            throw std::runtime_error("Invalid rotation matrix shape.");
        }
        auto ori_shape = matrix.sizes().vec();
        ori_shape.pop_back();
        *ori_shape.rbegin() = 9;
        auto mat_ = torch::unbind(matrix.reshape(ori_shape), -1);
        auto m00 = mat_[0], m01 = mat_[1], m02 = mat_[2];
        auto m10 = mat_[3], m11 = mat_[4], m12 = mat_[5];
        auto m20 = mat_[6], m21 = mat_[7], m22 = mat_[8];

        auto q_abs = _sqrt_positive_part(torch::stack({
            1 + m00 + m11 + m22,
            1 + m00 - m11 - m22,
            1 - m00 + m11 - m22,
            1 - m00 - m11 + m22 }, -1));

        auto quat_by_rijk = torch::stack({
            torch::stack({torch::pow(q_abs.index({Ellipsis, 0}), 2), m21 - m12, m02 - m20, m10 - m01}, -1),
            torch::stack({m21 - m12, torch::pow(q_abs.index({Ellipsis, 1}), 2), m10 + m01, m02 + m20}, -1),
            torch::stack({m02 - m20, m10 + m01, torch::pow(q_abs.index({Ellipsis, 2}), 2), m12 + m21}, -1),
            torch::stack({m10 - m01, m20 + m02, m21 + m12, torch::pow(q_abs.index({Ellipsis, 3}), 2)}, -1)
            }, -2);

        auto flr = torch::tensor(0.1).to(q_abs.options());
        auto quat_candidates = quat_by_rijk / (2 * q_abs.index({ Ellipsis, None }).max(flr));
        auto ret = quat_candidates.index({ torch::nn::functional::one_hot(q_abs.argmax(-1), 4) > 0.5 });
        // wxyz => xyzw
        ret = torch::cat({ret.index({Ellipsis, Slice(1, 4)}), ret.index({Ellipsis, Slice(0, 1)})}, -1);
        *ori_shape.rbegin() = 4;
        return ret.reshape(ori_shape);
    }

    // convert quaternion to rotation matrix
    torch::Tensor quat_to_matrix(torch::Tensor q)
    {
        if (q.size(-1) != 4)
        {
            throw std::length_error(std::string(__func__) + ": the shape of input quaternion should be (..., 4)");
        }
        torch::Tensor x = q.index({Ellipsis, Slice(0, 1)});
        torch::Tensor y = q.index({Ellipsis, Slice(1, 2)});
        torch::Tensor z = q.index({Ellipsis, Slice(2, 3)});
        torch::Tensor w = q.index({Ellipsis, Slice(3, 4)});

        torch::Tensor x2 = torch::square(x);
        torch::Tensor y2 = torch::square(y);
        torch::Tensor z2 = torch::square(z);
        torch::Tensor w2 = torch::square(w);

        torch::Tensor xy = x * y;
        torch::Tensor zw = z * w;
        torch::Tensor xz = x * z;
        torch::Tensor yw = y * w;
        torch::Tensor yz = y * z;
        torch::Tensor xw = x * w;

        torch::Tensor res00 = x2 - y2 - z2 + w2;
        torch::Tensor res10 = 2 * (xy + zw);
        torch::Tensor res20 = 2 * (xz - yw);

        torch::Tensor res01 = 2 * (xy - zw);
        torch::Tensor res11 = - x2 + y2 - z2 + w2;
        torch::Tensor res21 = 2 * (yz + xw);

        torch::Tensor res02 = 2 * (xz + yw);
        torch::Tensor res12 = 2 * (yz - xw);
        torch::Tensor res22 = - x2 - y2 + z2 + w2;

        // TODO: check the output and dimension, and reshape
        torch::Tensor result = torch::cat({
            res00, res01, res02,
            res10, res11, res12,
            res20, res21, res22}, -1);

        auto shape = q.sizes().vec();
        shape[shape.size() - 1] = 3;
        shape.push_back(3);
        result = result.reshape(shape);
        return result;
    }

    // build quaternion from 6d vector
    torch::Tensor quat_from_vec6d(const torch::Tensor x)
    {
        torch::Tensor mat = vec6d_to_matrix(x);
        torch::Tensor quat = quat_from_matrix(mat);
        return quat;
    }

    // convert quaternion to 6d representation
    torch::Tensor quat_to_vec6d(torch::Tensor q)
    {
        if (q.size(-1) != 4)
        {
            throw std::length_error(std::string(__func__) + ": the shape of input quaternion should be (..., 4)");
        }
        torch::Tensor x = q.index({ Ellipsis, Slice(0, 1) });
        torch::Tensor y = q.index({ Ellipsis, Slice(1, 2) });
        torch::Tensor z = q.index({ Ellipsis, Slice(2, 3) });
        torch::Tensor w = q.index({ Ellipsis, Slice(3, 4) });

        torch::Tensor x2 = torch::square(x);
        torch::Tensor y2 = torch::square(y);
        torch::Tensor z2 = torch::square(z);
        torch::Tensor w2 = torch::square(w);

        torch::Tensor xy = x * y;
        torch::Tensor zw = z * w;
        torch::Tensor xz = x * z;
        torch::Tensor yw = y * w;
        torch::Tensor yz = y * z;
        torch::Tensor xw = x * w;

        torch::Tensor res00 = x2 - y2 - z2 + w2;
        torch::Tensor res01 = 2 * (xy - zw);

        torch::Tensor res10 = 2 * (xy + zw);
        torch::Tensor res11 = -x2 + y2 - z2 + w2;

        torch::Tensor res20 = 2 * (xz - yw);
        torch::Tensor res21 = 2 * (yz + xw);

        // TODO: check the output and dimension, and reshape
        torch::Tensor result = torch::cat({
            res00, res01,
            res10, res11,
            res20, res21}, -1);

        auto shape = q.sizes().vec();
        shape[shape.size() - 1] = 3;
        shape.push_back(2);
        result = result.reshape(shape);
        return result;
    }

    // normalize 6d rotation representation
    std::vector<torch::Tensor> normalize_vec6d(torch::Tensor x)
    {
        auto ori_shape = x.sizes().vec();
        if (*ori_shape.rbegin() == 6)
        {
            *ori_shape.rbegin() = 3;
            ori_shape.push_back(2);
            x = x.reshape(ori_shape);
        }

        int64_t ndim = ori_shape.size();
        if(ndim < 2 || ori_shape[ndim - 2] != 3 || ori_shape[ndim - 1] != 2)
        {
            throw std::length_error(std::string(__func__) + "size not match.");
        }
        x = x / torch::linalg::norm(x, 2, -2, true, c10::nullopt);
        torch::Tensor first_col = x.index({Ellipsis, 0});
        torch::Tensor second_col = x.index({Ellipsis, 1});
        torch::Tensor last_col = torch::cross(first_col, second_col, -1);
        last_col = last_col / torch::linalg::norm(last_col, 2, -1, true, c10::nullopt);
        second_col = torch::cross(-first_col, last_col, -1);
        second_col = second_col / torch::linalg::norm(second_col, 2, -1, true, c10::nullopt);
        
        std::vector<torch::Tensor> result_list{first_col, second_col, last_col};
        return result_list;
    }

    torch::Tensor normalize_vec6d_cat(torch::Tensor x)
    {
        std::vector<torch::Tensor> res = normalize_vec6d(x);
        auto cat_result = torch::cat({res[0].index({Ellipsis, None}), res[1].index({Ellipsis, None})}, -1);
        return cat_result;
    }

    torch::Tensor vec6d_to_matrix(torch::Tensor x)
    {
        std::vector<torch::Tensor> res = normalize_vec6d(x);
        for (int i = 0; i < 3; i++)
        {
            res[i] = res[i].index({Ellipsis, None});
        }
        auto cat_result = torch::cat(res, -1);
        return cat_result;
    }

    torch::Tensor vector_to_cross_matrix(torch::Tensor x)
    {
        if (x.size(-1) != 3)
        {
            throw std::length_error(std::string(__func__) + ": the shape of input vector should be (..., 3)");
        }
        torch::Tensor x0 = x.index({Ellipsis, 0});
        torch::Tensor x1 = x.index({Ellipsis, 1});
        torch::Tensor x2 = x.index({Ellipsis, 2});
        torch::Tensor zero00 = torch::zeros_like(x0);
        torch::Tensor mat = torch::cat({
            zero00, -x2, x1,
            x2, zero00, -x0,
            -x1, x0, zero00}, -1);
        auto shape = x.sizes().vec();
        shape.push_back(3);
        mat = mat.reshape(shape);
        return mat;
    }

    torch::Tensor matrix_to_angle(torch::Tensor x)
    {
        // acos((tr(R)-1)/2)
        if (x.size(-1) != 3 || x.size(-2) != 3)
        {
            throw std::length_error(std::string(__func__) + ": the shape of input matrix should be (..., 3, 3)");
        }
        float eps = 1e-7f;
        torch::Tensor diag = torch::diagonal(x, 0, -1, -2);
        torch::Tensor trace = torch::sum(diag, -1);
        torch::Tensor trace_inside = torch::clamp(0.5 * (trace - 1), -1.0+eps, 1.0-eps);  // avoid NaN in acos function
        torch::Tensor angle = torch::acos(trace_inside);
        return angle;
    }

    torch::Tensor rotation_matrix_inv(torch::Tensor x)
    {
        if (x.ndimension() < 2 || x.size(-1) != 3 || x.size(-2) != 3)
        {
            throw std::length_error(std::string(__func__) + ": the shape of input matrix should be (..., 3, 3)");
        }
        return torch::transpose(x, -1, -2);
    }

    torch::Tensor matrix_to_rotvec(torch::Tensor matrix)
    {
        return quat_to_rotvec(quat_from_matrix(matrix));
    }

    torch::Tensor rotvec_to_matrix(torch::Tensor x)
    {
        return quat_to_matrix(quat_from_rotvec(x));
    }

    torch::Tensor rotvec_to_vec6d(torch::Tensor x)
    {
        return quat_to_vec6d(quat_from_rotvec(x));
    }

    // compute det of 2x2 matrix. input: (*, 2, 2)
    torch::Tensor mat22_det(torch::Tensor x)
    {
        if (x.ndimension() < 2 || x.size(-1) != 2 || x.size(-2) != 2)
        {
            throw std::length_error(std::string(__func__) + ": the shape of input matrix should be (..., 2, 2)");
        }
        torch::Tensor x00 = x.index({Ellipsis, 0, 0});
        torch::Tensor x01 = x.index({Ellipsis, 0, 1});
        torch::Tensor x10 = x.index({Ellipsis, 1, 0});
        torch::Tensor x11 = x.index({Ellipsis, 1, 1});
        torch::Tensor result = x00 * x11 - x01 * x10;
        return result;
    }

    torch::Tensor mat33_det(torch::Tensor x)
    {
        if (x.ndimension() < 2 || x.size(-1) != 3 || x.size(-2) != 3)
        {
            throw std::length_error(std::string(__func__) + ": the shape of input matrix should be (..., 3, 3)");
        }
        torch::Tensor a1 = x.index({Ellipsis, 0, 0});
        torch::Tensor b1 = x.index({Ellipsis, 0, 1});
        torch::Tensor c1 = x.index({Ellipsis, 0, 2});
        torch::Tensor a2 = x.index({Ellipsis, 1, 0});
        torch::Tensor b2 = x.index({Ellipsis, 1, 1});
        torch::Tensor c2 = x.index({Ellipsis, 1, 2});
        torch::Tensor a3 = x.index({Ellipsis, 2, 0});
        torch::Tensor b3 = x.index({Ellipsis, 2, 1});
        torch::Tensor c3 = x.index({Ellipsis, 2, 2});
        torch::Tensor result =
            a1 * b2 * c3 +
            b1 * c2 * a3 +
            c1 * a2 * b3 -
            a3 * b2 * c1 -
            b3 * c2 * a1 -
            c3 * a2 * b1;

        return result;
    }

    // svd decomposion of 3x3 matrix
    torch::Tensor mat33_svd(torch::Tensor x)
    {
        if (x.ndimension() < 2 || x.size(-1) !=3 || x.size(-2) !=3)
        {
            throw std::length_error(std::string(__func__) + ": the shape of input matrix should be (..., 3, 3)");
        }
        torch::Tensor result;

        return result;
    }

    torch::Tensor mat44_det(torch::Tensor x)
    {
        if (x.ndimension() < 2 || x.size(-1) !=4 || x.size(-2) !=4)
        {
            throw std::length_error(std::string(__func__) + ": the shape of input matrix should be (..., 4, 4)");
        }
        throw std::logic_error("not implemented");
        torch::Tensor result;
        return result;
    }

    torch::Tensor flip_vector(torch::Tensor vt, torch::Tensor normal)
    {
        auto shape = vt.sizes().vec();
        if (*shape.rbegin() != 3)
        {
            throw std::length_error(std::string(__func__) + ": the shape of input vector should be (..., 3)");
        }
        vt = vt.reshape({ -1, 3 });
        normal = normal.reshape({ -1, 3 });
        torch::Tensor res = vt - (2 * torch::sum(vt * normal, -1, true)) * normal;
        *shape.rbegin() = 3;
        return res.reshape(shape);
    }

    torch::Tensor flip_quaternion(torch::Tensor qt, torch::Tensor normal)
    {
        if (qt.size(-1) != 4 || normal.size(-1) != 3)
        {
            throw std::length_error(std::string(__func__) + ": the shape of input quaternion should be (..., 4)");
        }
        torch::Tensor vec_flip = flip_vector(qt.index({ Ellipsis, Slice(0, 3) }), normal);
        torch::Tensor res = torch::cat({ vec_flip, -1 * qt.index({Ellipsis, Slice(3, 4)}) }, -1);
        return res;
    }
        
        
#if 0
    // Get the code from https://github.com/KinglittleQ/torch-batch-svd/blob/master/torch_batch_svd/include/torch_batch_svd.h
    // solve U S V = svd(A)  a.k.a. syevj, where A (b, m, n), U (b, m, m), S (b,
    // min(m, n)), V (b, n, n) see also
    // https://docs.nvidia.com/cuda/cusolver/index.html#batchgesvdj-example1
    void batch_svd_forward(torch::Tensor a, torch::Tensor U, torch::Tensor s, torch::Tensor V,
        bool is_sort, double tol, int max_sweeps,
        bool is_double) {
        CHECK_CUDA(a);
        CHECK_CUDA(U);
        CHECK_CUDA(s);
        CHECK_CUDA(V);
        CHECK_IS_FLOAT(a);

        const at::cuda::OptionalCUDAGuard device_guard(a.device());

        auto handle_ptr = unique_allocate(cusolverDnCreate, cusolverDnDestroy);

        // important!!! Convert from row major to column major
        const auto A =
            a.contiguous().clone().transpose(1, 2).contiguous().transpose(1, 2);

        const auto batch_size = A.size(0);
        const auto m = A.size(1);
        TORCH_CHECK(m <= 32, "matrix row should be <= 32");
        const auto n = A.size(2);
        TORCH_CHECK(n <= 32, "matrix col should be <= 32");
        const auto lda = m;
        const auto ldu = m;
        const auto ldv = n;

        auto params =
            unique_allocate(cusolverDnCreateGesvdjInfo, cusolverDnDestroyGesvdjInfo);
        auto status = cusolverDnXgesvdjSetTolerance(params.get(), tol);
        TORCH_CHECK(CUSOLVER_STATUS_SUCCESS == status,
            "cusolverDnXgesvdjSetTolerance status ", status);
        status = cusolverDnXgesvdjSetMaxSweeps(params.get(), max_sweeps);
        TORCH_CHECK(CUSOLVER_STATUS_SUCCESS == status,
            "cusolverDnXgesvdjSetMaxSweeps status ", status);
        status = cusolverDnXgesvdjSetSortEig(params.get(), is_sort);
        TORCH_CHECK(CUSOLVER_STATUS_SUCCESS == status,
            "cusolverDnXgesvdjSetSortEig status ", status);

        auto jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors
        int lwork;
        auto info_ptr = unique_cuda_ptr<int>(batch_size);

        if (is_double) {
            const auto d_A = A.data_ptr<double>();
            auto d_s = s.data_ptr<double>();
            const auto d_U = U.data_ptr<double>();
            const auto d_V = V.data_ptr<double>();

            status = cusolverDnDgesvdjBatched_bufferSize(
                handle_ptr.get(), jobz, m, n, d_A, lda, d_s, d_U, ldu, d_V, ldv, &lwork,
                params.get(), batch_size);
            TORCH_CHECK(CUSOLVER_STATUS_SUCCESS == status,
                "cusolverDnDgesvdjBatched_bufferSize status ", status);
            auto work_ptr = unique_cuda_ptr<double>(lwork);

            status = cusolverDnDgesvdjBatched(
                handle_ptr.get(), jobz, m, n, d_A, lda, d_s, d_U, ldu, d_V, ldv,
                work_ptr.get(), lwork, info_ptr.get(), params.get(), batch_size);
            TORCH_CHECK(CUSOLVER_STATUS_SUCCESS == status,
                "cusolverDnDgesvdjBatched status ", status);
        }
        else {
            const auto d_A = A.data_ptr<float>();
            auto d_s = s.data_ptr<float>();
            const auto d_U = U.data_ptr<float>();
            const auto d_V = V.data_ptr<float>();

            status = cusolverDnSgesvdjBatched_bufferSize(
                handle_ptr.get(), jobz, m, n, d_A, lda, d_s, d_U, ldu, d_V, ldv, &lwork,
                params.get(), batch_size);
            TORCH_CHECK(CUSOLVER_STATUS_SUCCESS == status,
                "cusolverDnSgesvdjBatched_bufferSize status ", status);
            auto work_ptr = unique_cuda_ptr<float>(lwork);

            status = cusolverDnSgesvdjBatched(
                handle_ptr.get(), jobz, m, n, d_A, lda, d_s, d_U, ldu, d_V, ldv,
                work_ptr.get(), lwork, info_ptr.get(), params.get(), batch_size);
            TORCH_CHECK(CUSOLVER_STATUS_SUCCESS == status,
                "cusolverDnSgesvdjBatched status ", status);
        }

        std::vector<int> hinfo(batch_size);
        auto status_memcpy =
            cudaMemcpy(hinfo.data(), info_ptr.get(), sizeof(int) * batch_size,
                cudaMemcpyDeviceToHost);
        TORCH_CHECK(cudaSuccess == status_memcpy, "cudaMemcpy status ",
            status_memcpy);

        for (int i = 0; i < batch_size; ++i) {
            if (0 == hinfo[i]) {
                continue;
            }
            else if (0 > hinfo[i]) {
                std::cout << "Error: " << -hinfo[i] << "-th parameter is wrong"
                    << std::endl;
                TORCH_CHECK(false);
            }
            else {
                std::cout << "WARNING: matrix " << i << ", info = " << hinfo[i]
                    << ": Jacobi method does not converge" << std::endl;
            }
        }
    }

    // https://j-towns.github.io/papers/svd-derivative.pdf
    //
    // This makes no assumption on the signs of sigma.
    at::Tensor batch_svd_backward(const std::vector<at::Tensor>& grads,
        const at::Tensor& self, bool some,
        bool compute_uv, const at::Tensor& raw_u,
        const at::Tensor& sigma,
        const at::Tensor& raw_v) {
        TORCH_CHECK(compute_uv,
            "svd_backward: Setting compute_uv to false in torch.svd doesn't "
            "compute singular matrices, ",
            "and hence we cannot compute backward. Please use "
            "torch.svd(compute_uv=True)");

        const at::cuda::OptionalCUDAGuard device_guard(self.device());

        // A [b, m, n]
        // auto b = self.size(0);
        auto m = self.size(1);
        auto n = self.size(2);
        auto k = sigma.size(1);
        auto gsigma = grads[1];

        auto u = raw_u;
        auto v = raw_v;
        auto gu = grads[0];
        auto gv = grads[2];

        if (!some) {
            // We ignore the free subspace here because possible base vectors cancel
            // each other, e.g., both -v and +v are valid base for a dimension.
            // Don't assume behavior of any particular implementation of svd.
            u = raw_u.narrow(2, 0, k);
            v = raw_v.narrow(2, 0, k);
            if (gu.defined()) {
                gu = gu.narrow(2, 0, k);
            }
            if (gv.defined()) {
                gv = gv.narrow(2, 0, k);
            }
        }
        auto vt = v.transpose(1, 2);

        at::Tensor sigma_term;
        if (gsigma.defined()) {
            sigma_term = u.bmm(gsigma.diag_embed()).bmm(vt);
        }
        else {
            sigma_term = at::zeros({ 1 }, self.options()).expand_as(self);
        }
        // in case that there are no gu and gv, we can avoid the series of kernel
        // calls below
        if (!gv.defined() && !gu.defined()) {
            return sigma_term;
        }

        auto ut = u.transpose(1, 2);
        auto im = at::eye(m, self.options()); // work if broadcast
        auto in = at::eye(n, self.options());
        auto sigma_mat = sigma.diag_embed();
        auto sigma_mat_inv = sigma.pow(-1).diag_embed();
        auto sigma_expanded_sq = sigma.pow(2).unsqueeze(1).expand_as(sigma_mat);
        auto F = sigma_expanded_sq - sigma_expanded_sq.transpose(1, 2);
        // The following two lines invert values of F, and fills the diagonal with 0s.
        // Notice that F currently has 0s on diagonal. So we fill diagonal with +inf
        // first to prevent nan from appearing in backward of this function.
        F.diagonal(0, -2, -1).fill_(INFINITY);
        F = F.pow(-1);

        at::Tensor u_term, v_term;

        if (gu.defined()) {
            u_term =
                u.bmm(F.mul(ut.bmm(gu) - gu.transpose(1, 2).bmm(u))).bmm(sigma_mat);
            if (m > k) {
                u_term = u_term + (im - u.bmm(ut)).bmm(gu).bmm(sigma_mat_inv);
            }
            u_term = u_term.bmm(vt);
        }
        else {
            u_term = at::zeros({ 1 }, self.options()).expand_as(self);
        }

        if (gv.defined()) {
            auto gvt = gv.transpose(1, 2);
            v_term = sigma_mat.bmm(F.mul(vt.bmm(gv) - gvt.bmm(v))).bmm(vt);
            if (n > k) {
                v_term = v_term + sigma_mat_inv.bmm(gvt.bmm(in - v.bmm(vt)));
            }
            v_term = u.bmm(v_term);
        }
        else {
            v_term = at::zeros({ 1 }, self.options()).expand_as(self);
        }

        return u_term + sigma_term + v_term;
    }

    class BatchSVDFunction: public torch::autograd::Function<BatchSVDFunction>
    {
    public:
        static std::vector<torch::Tensor> forward(torch::autograd::AutogradContext* ctx, torch::Tensor input)
        {
            bool is_double = input.dtype() == torch::kDouble;
            int b = input.size(0), m = input.size(1), n = input.size(2);
            int k = std::min(m, n);
            torch::Tensor U = torch::empty({ b, m, m }, input.options());
            torch::Tensor S = torch::empty({ b, k}, input.options());
            torch::Tensor V = torch::empty({ b, n, n }, input.options());
            batch_svd_forward(input, U, S, V, true, 1e-7, 100, is_double);
            U.transpose_(1, 2);
            V.transpose_(1, 2);
            torch::Tensor U_reduced = U.index({ Slice(), Slice(), Slice(0, k) });
            torch::Tensor V_reduced = V.index({ Slice(), Slice(), Slice(0, k) });
            if (input.requires_grad())
            {
                ctx->save_for_backward({ input, U_reduced, S, V_reduced });
            }
            return {U_reduced, S, V_reduced};
        }

        static std::vector<torch::Tensor> backward(
            torch::autograd::AutogradContext* ctx,
            std::vector<torch::Tensor> grad_in
        )
        {
            auto & saved_vars = ctx->get_saved_variables();
            torch::Tensor A = saved_vars[0], U = saved_vars[1], S = saved_vars[2], V = saved_vars[3];
            torch::Tensor grad_out = batch_svd_backward(
                grad_in, A, true, true, U.to(A.dtype()), S.to(A.dtype()), V.to(A.dtype())
            );
            return { grad_out };
        }
    };

    std::vector<torch::Tensor> fast_svd(torch::Tensor input)
    {
        // szh: 
        // It is much faster when running svd on matrix (batch, large, small)
        // The time usage is close to (batch, small, small)
        // Time usage on (batch, large, large) is similar as torch.svd in Python.
        // performance on float64 is a little better than torch.svd in Python.
        return BatchSVDFunction::apply(input);
    }
#endif


#ifndef _DEBUG_AS_MAIN_FILE
    void add_pybind11_wrapper(pybind11::module & m)
    {
        namespace py = pybind11;
        m.doc() = "Fast implementation of rotation operation with PyTorch. Cuda and cuDNN are required.";
        m.def("quat_multiply", &quat_multiply, py::arg("q1"), py::arg("q2"));
        m.def("quat_apply", &quat_apply, py::arg("q"), py::arg("v"));
        m.def("quat_inv", &quat_inv, py::arg("q"));

        m.def("flip_quat_by_w", &flip_quat_by_w, py::arg("q"));
        m.def("quat_normalize", &quat_normalize, "Normalize the quaternion.", py::arg("q"));
        m.def("quat_integrate", &quat_integrate, py::arg("q"), py::arg("omega"), py::arg("dt"));
        m.def("quat_between", &quat_between, py::arg("a"), py::arg("b"));
        m.def("decompose_rotation", &decompose_rotation, py::arg("q"), py::arg("vb"));
        m.def("x_decompose", &x_decompose, py::arg("q"));
        m.def("y_decompose", &y_decompose, py::arg("q"));
        m.def("z_decompose", &z_decompose, py::arg("q"));

        m.def("quat_to_angle", &quat_to_angle, py::arg("q"));
        m.def("quat_to_rotvec", &quat_to_rotvec, py::arg("q"));
        m.def("quat_from_rotvec", &quat_from_rotvec, py::arg("rotvec"));

        m.def("quat_from_matrix", &quat_from_matrix, py::arg("mat"));
        m.def("quat_to_matrix", &quat_to_matrix, py::arg("q"));

        m.def("quat_from_vec6d", &quat_from_vec6d, py::arg("vec6d"));
        m.def("quat_to_vec6d", &quat_to_vec6d, py::arg("q"));

        m.def("matrix_to_rotvec", &matrix_to_rotvec, py::arg("mat"));
        m.def("rotvec_to_matrix", &rotvec_to_matrix, py::arg("x"));
        m.def("rotvec_to_vec6d", &rotvec_to_vec6d, py::arg("x"));

        m.def("normalize_vec6d", &normalize_vec6d, py::arg("x"));
        m.def("normalize_vec6d_cat", &normalize_vec6d_cat, py::arg("x"));
        m.def("vec6d_to_matrix", &vec6d_to_matrix, py::arg("x"));

        m.def("matrix_to_angle", &matrix_to_angle, py::arg("x"));
        m.def("mat22_det",&mat22_det, py::arg("x"));
        m.def("mat33_det", &mat33_det, py::arg("x"));
        m.def("mat33_svd", &mat33_svd, py::arg("x"));
        m.def("mat44_det", &mat33_det, py::arg("x"));

        m.def("flip_vector", &flip_vector, py::arg("qt"), py::arg("normal"));
        m.def("flip_quaternion", &flip_quaternion, py::arg("qt"), py::arg("normal"));

        // TODO: support euler angle.
#if 0
        m.def("fast_svd", &fast_svd);
#endif
    }
#endif

};

//#ifndef ROTATION_AS_EXTERNAL
//PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
//{
//    RotationLibTorch::add_pybind11_wrapper(m);
//}
//#endif

#include "Config.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include "RotationCppBatch.hpp"

namespace RotationCppBatchWrapper
{
    namespace py = pybind11;
    using namespace RotationCppBatch;
    // TODO: support tile operation here..

    static size_t get_batch_size(const py::ssize_t * shape, py::ssize_t ndim)
    {
        return std::accumulate(shape, shape + ndim, (py::ssize_t) 1, std::multiplies<py::ssize_t>());
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    py::array_t<data_type> quat_multiply_np(
        const py::array_t<data_type, py::array::c_style | py::array::forcecast> q1,
        const py::array_t<data_type, py::array::c_style | py::array::forcecast> q2)
    {
        #if ROTATION_CHECK_INPUT_SIZE
        auto q1_size = q1.size(), q2_size = q2.size();
        if (q1_size != q2_size || q1.shape(q1.ndim() - 1) != 4 || q2.shape(q2.ndim() - 1) != 4)
        {
            throw py::value_error("shape of input quarernions should match.");
        }
        #endif

        auto shape_ptr = q1.shape();
        auto result = py::array_t<data_type>(std::vector<py::ssize_t>(shape_ptr, shape_ptr + q1.ndim()));
        // TODO: make sure gil is not used for openmp..
        quat_multiply<data_type>(
            static_cast<const data_type *>(q1.request().ptr),
            static_cast<const data_type *>(q2.request().ptr),
            static_cast<data_type *>(result.request().ptr),
            get_batch_size(shape_ptr, q1.ndim() - 1)
        );
        return result;
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    py::array_t<data_type> quat_apply_np(
        const py::array_t<data_type, py::array::c_style> q,
        const py::array_t<data_type, py::array::c_style> v)
    {
        #if ROTATION_CHECK_INPUT_SIZE
        if (q.size() < 4 || v.size() < 3)
        {
            throw py::index_error("shape should be larger than 4");
        }
        #endif
        auto shape_ptr = v.shape();
        auto result = py::array_t<data_type>(std::vector<py::ssize_t>(shape_ptr, shape_ptr + v.ndim()));
        quat_apply<data_type>(
            static_cast<const data_type *>(q.request().ptr),
            static_cast<const data_type *>(v.request().ptr),
            static_cast<data_type *>(result.request().ptr),
            get_batch_size(shape_ptr, v.ndim() - 1)
        );
        return result;
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    py::array_t<data_type> quat_to_matrix_np(const py::array_t<data_type, py::array::c_style> q)
    {
        #if ROTATION_CHECK_INPUT_SIZE
        if (q.size() < 4)
        {
            throw py::index_error("shape should be larger than 4");
        }
        #endif
        auto shape_ptr = q.shape();
        auto shape = std::vector<py::ssize_t>(shape_ptr, shape_ptr + q.ndim() - 1);
        shape.emplace_back(3);
        shape.emplace_back(3);

        auto result = py::array_t<data_type>(shape);
        quat_to_matrix(
            static_cast<const data_type *>(q.request().ptr),
            static_cast<data_type *>(result.request().ptr),
            get_batch_size(shape_ptr, q.ndim() - 1)
        );
        return result;
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    py::array_t<data_type> quat_from_matrix_np(const py::array_t<data_type, py::array::c_style> m)
    {
        #if ROTATION_CHECK_INPUT_SIZE
            if (m.size() < 9)
            {
                throw py::index_error("shape should be (3, 3)");
            }
        #endif
        auto shape_ptr = m.shape();
        auto shape = std::vector<py::ssize_t>(shape_ptr, shape_ptr + m.ndim() - 1);
        *shape.rbegin() = 4;
        auto result = py::array_t<data_type>(shape);

        quat_from_matrix(
            static_cast<const data_type *>(m.request().ptr),
            static_cast<data_type *>(result.request().ptr),
            get_batch_size(shape_ptr, m.ndim() - 2)
        );
        return result;
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    py::array_t<data_type> cross_product_np(
        const py::array_t<data_type, py::array::c_style> a,
        const py::array_t<data_type, py::array::c_style> b)
    {
        #if ROTATION_CHECK_INPUT_SIZE
            if (a.size() < 3 || b.size() < 3)
            {
                throw py::index_error("shape should be (3,)");
            }
        #endif

        auto result = py::array_t<data_type>(std::vector<size_t>(a.shape(), a.shape() + a.ndim()));
        cross_product(
            static_cast<const data_type *>(a.request().ptr),
            static_cast<const data_type *>(b.request().ptr),
            static_cast<data_type *>(result.request().ptr),
            get_batch_size(a.shape(), a.ndim() - 1)
        );
        return result;
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    py::array_t<data_type> vector_to_cross_matrix_np(const py::array_t<data_type, py::array::c_style> a)
    {
        #if ROTATION_CHECK_INPUT_SIZE
            if (a.size() < 3)
            {
                throw py::index_error("shape should be (3,)");
            }
        #endif
        auto shape_ptr = a.shape();
        auto shape = std::vector<size_t>(shape_ptr, shape_ptr + a.ndim());
        shape.emplace_back(3);

        auto result = py::array_t<data_type>(shape);
        vector_to_cross_matrix(
            static_cast<const data_type *>(a.request().ptr),
            static_cast<data_type *>(result.request().ptr),
            get_batch_size(shape_ptr, a.ndim() - 1)
        );
        return result;
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    py::array_t<data_type> quat_to_rotvec_np(const py::array_t<data_type, py::array::c_style> q)
    {
        #if ROTATION_CHECK_INPUT_SIZE
            if (q.size() < 4)
            {
                throw py::index_error("shape should be (4,)");
            }
        #endif
        auto shape_ptr = q.shape();
        auto shape = std::vector<size_t>(shape_ptr, shape_ptr + q.ndim());
        *shape.rbegin() = 3;
        auto result = py::array_t<data_type>(shape);
        quat_to_rotvec<data_type>(
            static_cast<const data_type *>(q.request().ptr), nullptr,
            static_cast<data_type *>(result.request().ptr),
            get_batch_size(q.shape(), q.ndim() - 1)
        );
        return result;
    }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    py::array_t<data_type> quat_from_rotvec_np(const py::array_t<data_type, py::array::c_style> x)
    {
        #if ROTATION_CHECK_INPUT_SIZE
        if (x.size() < 3)
        {
            throw py::index_error("shape should be (3,)");
        }
        #endif
        auto shape_ptr = x.shape();
        auto shape = std::vector<size_t>(shape_ptr, shape_ptr + x.ndim());
        *shape.rbegin() = 4;

        auto result = py::array_t<data_type>(shape);
        quat_from_rotvec<data_type>(
            static_cast<const data_type *>(x.request().ptr),
            static_cast<data_type *>(result.request().ptr),
            get_batch_size(shape_ptr, x.ndim() - 1)
        );
        return result;
    }

    /* template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    py::array_t<data_type> vector_normalize_np(py::array_t<data_type, py::array::c_style> x)
    {
        auto size = x.size();
        auto result = py::array_t<data_type>(size);
        vector_normalize(
            static_cast<data_type *>(x.request().ptr), size,
            static_cast<data_type *>(result.request().ptr)
        );
        return result;
    } */

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    py::array_t<data_type> quat_integrate_np(
        const py::array_t<data_type, py::array::c_style> q,
        const py::array_t<data_type, py::array::c_style> omega,
        data_type dt
    )
    {
        #if ROTATION_CHECK_INPUT_SIZE
        if (q.size() < 4 || omega.size() < 3)
        {
            throw py::index_error("shape should be (3,)");
        }
        #endif
        
        auto shape_ptr = q.shape();
        auto shape = std::vector<size_t>(shape_ptr, shape_ptr + q.ndim());
        auto result = py::array_t<data_type>(shape);
        quat_integrate(
            static_cast<const data_type *>(q.request().ptr),
            static_cast<const data_type *>(omega.request().ptr),
            dt,
            static_cast<data_type *>(result.request().ptr),
            get_batch_size(shape_ptr, q.ndim() - 1)
        );
        return result;
    }

    /* void build_pybind11_wrapper(pybind11::module m)
    {
        m.doc() = "";

        m.def("quat_to_vec6d", nullptr);
        m.def("clip_vec", nullptr);
        m.def("clip_vec_by_length", nullptr);
        m.def("mat22_det", nullptr);
        m.def("mat33_det", nullptr);
        m.def("mat33_svd", nullptr);
    } */

    // void build_omp_wrapper(pybind11::module_ m)
    // {
    //     m.def("get_max_threads", &omp_get_max_threads, "Returns max number of threads");
    //     m.def("set_num_threads", &omp_set_num_threads, "Set number of threads");
    // }

    template<class data_type, class = typename std::enable_if<std::is_floating_point<data_type>::value>::type>
    void build_pybind11_wrapper(pybind11::module_ m)
    {
        using namespace RotationCppConfig;
        m.def("get_use_parallel_flag", []() {return use_parallel_flag; });
        m.def("set_use_parallel_flag", [](int value) {use_parallel_flag = value; });
        m.def("quat_multiply", &quat_multiply_np<data_type>);
        m.def("quat_apply", &quat_apply_np<data_type>);
        m.def("quat_to_matrix", &quat_to_matrix_np<data_type>);
        m.def("quat_from_matrix", &quat_from_matrix_np<data_type>);
        m.def("cross_product", &cross_product_np<data_type>);
        m.def("vector_to_cross_matrix", &vector_to_cross_matrix_np<data_type>);
        m.def("quat_to_rotvec", &quat_to_rotvec_np<data_type>);
        m.def("quat_from_rotvec", &quat_from_rotvec_np<data_type>);
        // m.def("vector_normalize", &vector_normalize_np<data_type>);
        m.def("quat_integrate", &quat_integrate_np<data_type>);
    }

    void build_wrapper(pybind11::module_ m)
    {
        auto impl32 = m.def_submodule("impl32", "implementation in float32");
        auto impl64 = m.def_submodule("impl64", "implementation in float64");
        build_pybind11_wrapper<float>(impl32);
        build_pybind11_wrapper<double>(impl64);
        build_pybind11_wrapper<double>(m);
    }
}



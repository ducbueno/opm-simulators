#ifndef ISAIKERNELS_H_
#define ISAIKERNELS_H_

#include <string>
#include <opm/simulators/linalg/bda/opencl.hpp>

namespace bda{
    using isai_L_kernel_type = cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, const unsigned int>;
    using isai_U_kernel_type = cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, const unsigned int>;
    using apply_invL_kernel_type = cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, const unsigned int, cl::LocalSpaceArg>;
    using apply_invU_kernel_type = cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, const unsigned int, cl::LocalSpaceArg>;

    std::string get_isai_L_string();
    std::string get_isai_U_string();
    std::string get_apply_invL_string();
    std::string get_apply_invU_string();
}

#endif // ISAIKERNELS_H_

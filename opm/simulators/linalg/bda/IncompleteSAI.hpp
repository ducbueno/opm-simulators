#ifndef INCOMPLETESAI_H_
#define INCOMPLETESAI_H_

#include <mutex>
#include <vector>
#include <opm/simulators/linalg/bda/opencl.hpp>
#include <opm/simulators/linalg/bda/ISAIKernels.hpp>

namespace bda{
    class IncompleteSAI{
    private:
        unsigned int N, Nb, bs, nnzbs, verbosity;
        std::vector<int> mapping;
        std::once_flag initialize_flag;

        cl_int err;
        cl::Context *context;
        cl::CommandQueue *queue;
        cl::Program program;
        std::vector<cl::Device> devices;
        cl::Buffer d_invLUvals, d_mapping, d_invL_x;

        std::unique_ptr<isai_L_kernel_type> isai_L_k;
        std::unique_ptr<isai_U_kernel_type> isai_U_k;
        std::unique_ptr<apply_invL_kernel_type> apply_invL_k;
        std::unique_ptr<apply_invU_kernel_type> apply_invU_k;

        void isai_L_w(cl::Buffer& d_colPtr, cl::Buffer& d_rowIndex, cl::Buffer& d_diagIndex, cl::Buffer& d_LUvals);
        void isai_U_w(cl::Buffer& d_colPtr, cl::Buffer& d_rowIndex, cl::Buffer& d_diagIndex, cl::Buffer& d_LUvals);
        void apply_invL_w(cl::Buffer& d_colPtr, cl::Buffer& d_rowIndex, cl::Buffer& d_diagIndex, cl::Buffer& x);
        void apply_invU_w(cl::Buffer& d_colPtr, cl::Buffer& d_rowIndex, cl::Buffer& d_diagIndex, cl::Buffer& y);
        void init();

    public:
        void setParams(int Nb, int bs, int nnzbs, int verbosity);
        void setMapping(int *CSRColIndices, int *CSRRowPointers);
        void setOpenCLContext(cl::Context *context);
        void setOpenCLQueue(cl::CommandQueue *queue);
        void create_preconditioner(cl::Buffer& d_colPtr, cl::Buffer& d_rowIndex, cl::Buffer& d_diagIndex, cl::Buffer& d_LUvals);
        void apply(cl::Buffer& d_colPtr, cl::Buffer& d_rowIndex, cl::Buffer& d_diagIndex, cl::Buffer& x, cl::Buffer& y);
    };
}

#endif // INCOMPLETESAI_H_

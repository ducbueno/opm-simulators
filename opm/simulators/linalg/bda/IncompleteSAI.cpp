#include <iostream>

#include <config.h>

#include <opm/common/OpmLog/OpmLog.hpp>
#include <opm/common/ErrorMacros.hpp>
#include <dune/common/timer.hh>

#include <opm/simulators/linalg/bda/IncompleteSAI.hpp>

namespace bda{
    using Opm::OpmLog;
    using Dune::Timer;

    void IncompleteSAI::setParams(int Nb_, int bs_, int nnzbs_, int verbosity_){
        this->Nb = Nb_;
        this->N = bs_ * Nb_;
        this->bs = bs_;
        this->nnzbs = nnzbs_;
        this->verbosity = verbosity_;
    }

    void IncompleteSAI::setMapping(int *CSRColIndices, int *CSRRowPointers){
        Dune::Timer t_mapping;
        std::vector<int> aux(CSRRowPointers, CSRRowPointers + Nb + 1);
        mapping.resize(nnzbs);

        for(int row = 0; row < Nb; row++){
            for(int jj = CSRRowPointers[row]; jj < CSRRowPointers[row+1]; jj++){
                int col = CSRColIndices[jj];
                int dest = aux[col];
                mapping[dest] = jj;
                aux[col]++;
            }
        }

        if(verbosity >= 4){
            std::ostringstream out;
            out << "IncompleteSAI setMapping time: " << t_mapping.stop() << " s";
            OpmLog::info(out.str());
        }
    }

    void IncompleteSAI::setOpenCLContext(cl::Context *context_){
        this->context = context_;
    }

    void IncompleteSAI::setOpenCLQueue(cl::CommandQueue *queue_){
        this->queue = queue_;
    }

    unsigned int ceilDivision(const unsigned int A, const unsigned int B){
        return A / B + (A % B > 0);
    }

    void IncompleteSAI::isai_L_w(cl::Buffer& d_colPtr, cl::Buffer& d_rowIndex, cl::Buffer& d_diagIndex, cl::Buffer& d_LUvals){
        const unsigned int work_group_size = 256;
        const unsigned int num_work_groups = ceilDivision(Nb, work_group_size);
        const unsigned int total_work_items = num_work_groups * work_group_size;


        Dune::Timer t_isai_L;
        cl::Event event = (*isai_L_k)(cl::EnqueueArgs(*queue, cl::NDRange(total_work_items), cl::NDRange(work_group_size)),
                                      d_mapping, d_colPtr, d_rowIndex, d_diagIndex, d_LUvals, d_invLUvals, Nb);

        if(verbosity >= 4){
            event.wait();
            std::ostringstream oss;
            oss << std::scientific << "IncompleteSAI isai_L time: " << t_isai_L.stop() << " s";
            OpmLog::info(oss.str());
        }
    }

    void IncompleteSAI::isai_U_w(cl::Buffer& d_colPtr, cl::Buffer& d_rowIndex, cl::Buffer& d_diagIndex, cl::Buffer& d_LUvals){
        const unsigned int work_group_size = 256;
        const unsigned int num_work_groups = ceilDivision(Nb, work_group_size);
        const unsigned int total_work_items = num_work_groups * work_group_size;

        Dune::Timer t_isai_U;
        cl::Event event = (*isai_U_k)(cl::EnqueueArgs(*queue, cl::NDRange(total_work_items), cl::NDRange(work_group_size)),
                                      d_mapping, d_colPtr, d_rowIndex, d_diagIndex, d_LUvals, d_invLUvals, Nb);

        if(verbosity >= 4){
            event.wait();
            std::ostringstream oss;
            oss << std::scientific << "IncompleteSAI isai_U time: " << t_isai_U.stop() << " s";
            OpmLog::info(oss.str());
        }
    }

    void IncompleteSAI::apply_invL_w(cl::Buffer& d_colPtr, cl::Buffer& d_rowIndex, cl::Buffer& d_diagIndex, cl::Buffer& x){
        const unsigned int work_group_size = 32;
        const unsigned int num_work_groups = ceilDivision(N, work_group_size);
        const unsigned int total_work_items = num_work_groups * work_group_size;
        const unsigned int lmem_per_work_group = sizeof(double) * work_group_size;

        Dune::Timer t_apply_invL;
        cl::Event event = (*apply_invL_k)(cl::EnqueueArgs(*queue, cl::NDRange(total_work_items), cl::NDRange(work_group_size)),
                                          d_colPtr, d_rowIndex, d_diagIndex, d_invLUvals, x, d_invL_x, Nb+1, cl::Local(lmem_per_work_group));

        if (verbosity >= 4) {
            event.wait();
            std::ostringstream oss;
            oss << std::scientific << "IncompleteSAI apply_invL time: " << t_apply_invL.stop() << " s";
            OpmLog::info(oss.str());
        }
    }

    void IncompleteSAI::apply_invU_w(cl::Buffer& d_colPtr, cl::Buffer& d_rowIndex, cl::Buffer& d_diagIndex, cl::Buffer& y){
        const unsigned int work_group_size = 32;
        const unsigned int num_work_groups = ceilDivision(N, work_group_size);
        const unsigned int total_work_items = num_work_groups * work_group_size;
        const unsigned int lmem_per_work_group = sizeof(double) * work_group_size;

        Dune::Timer t_apply_invU;
        cl::Event event = (*apply_invU_k)(cl::EnqueueArgs(*queue, cl::NDRange(total_work_items), cl::NDRange(work_group_size)),
                                          d_colPtr, d_rowIndex, d_diagIndex, d_invLUvals, d_invL_x, y, Nb+1, cl::Local(lmem_per_work_group));

        if (verbosity >= 4) {
            event.wait();
            std::ostringstream oss;
            oss << std::scientific << "IncompleteSAI apply_invU time: " << t_apply_invU.stop() << " s";
            OpmLog::info(oss.str());
        }
    }

    void IncompleteSAI::init(){
        try{
            std::string isai_L_s = get_isai_L_string();
            std::string isai_U_s = get_isai_U_string();
            std::string apply_invL_s = get_apply_invL_string();
            std::string apply_invU_s = get_apply_invU_string();

            cl::Program::Sources sources;
            sources.emplace_back(std::make_pair(isai_L_s.c_str(), isai_L_s.size()));
            sources.emplace_back(std::make_pair(isai_U_s.c_str(), isai_U_s.size()));
            sources.emplace_back(std::make_pair(apply_invL_s.c_str(), apply_invL_s.size()));
            sources.emplace_back(std::make_pair(apply_invU_s.c_str(), apply_invU_s.size()));

            program = cl::Program(*context, sources, &err);
            if(err != CL_SUCCESS){
                OPM_THROW(std::logic_error, "IncompleteSAI: OpenCL could not create program");
            }

            devices = context->getInfo<CL_CONTEXT_DEVICES>();
            program.build(devices);

            isai_L_k.reset(new isai_L_kernel_type(cl::Kernel(program, "isai_L", &err)));
            isai_U_k.reset(new isai_U_kernel_type(cl::Kernel(program, "isai_U", &err)));
            apply_invL_k.reset(new apply_invL_kernel_type(cl::Kernel(program, "apply_invL", &err)));
            apply_invU_k.reset(new apply_invU_kernel_type(cl::Kernel(program, "apply_invU", &err)));

            if(err != CL_SUCCESS){
                OPM_THROW(std::logic_error, "IncompleteSAI: OpenCL could not create kernels");
            }

            d_invLUvals = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(double) * nnzbs * bs * bs);
            d_mapping = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(int) * nnzbs);
            d_invL_x = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(double) * Nb * bs);

            Dune::Timer t_copy_mapping;
            cl::Event event;
            err = queue->enqueueWriteBuffer(d_mapping, CL_TRUE, 0, sizeof(int) * mapping.size(), mapping.data(), nullptr, &event);
            event.wait();;

            if(verbosity >= 4){
                std::ostringstream out;
                out << "IncompleteSAI copy mapping to GPU time: " << t_copy_mapping.stop() << " s";
                OpmLog::info(out.str());
            }

            if(err != CL_SUCCESS){
                OPM_THROW(std::logic_error, "IncompleteSAI: OpenCL error writting data");
            }
        }

        catch(const cl::Error& error){
            std::ostringstream oss;
            oss << "IncompleteSAI OpenCL Error: " << error.what() << "(" << error.err() << ")\n";
            oss << getErrorString(error.err()) << std::endl;

            if(error.err() == CL_BUILD_PROGRAM_FAILURE){
                for(cl::Device dev: devices){
                    cl_build_status status = program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(dev);
                    if(status != CL_BUILD_ERROR) continue;

                    std::string name = dev.getInfo<CL_DEVICE_NAME>();
                    std::string buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev);
                    std::cerr << "Build log for " << name << ":" << std::endl << buildlog << std::endl;
                }
            }

            OPM_THROW(std::logic_error, oss.str());
        }

        catch(const std::logic_error& error){
            throw error;
        }
    }

    void IncompleteSAI::create_preconditioner(cl::Buffer& d_colPtr, cl::Buffer& d_rowIndex, cl::Buffer& d_diagIndex, cl::Buffer& d_LUvals){
        std::call_once(initialize_flag, [&](){
            init();
        });

        cl::Event init_event;
        queue->enqueueFillBuffer(d_invLUvals, 0, 0, sizeof(double) * nnzbs * bs * bs, nullptr, &init_event);
        init_event.wait();

        isai_L_w(d_colPtr, d_rowIndex, d_diagIndex, d_LUvals);
        isai_U_w(d_colPtr, d_rowIndex, d_diagIndex, d_LUvals);
    }

    void IncompleteSAI::apply(cl::Buffer& d_colPtr, cl::Buffer& d_rowIndex, cl::Buffer& d_diagIndex, cl::Buffer& x, cl::Buffer& y){
        cl::Event init_event;
        queue->enqueueFillBuffer(d_invL_x, 0, 0, sizeof(double) * Nb * bs, nullptr, &init_event);
        init_event.wait();

        apply_invL_w(d_colPtr, d_rowIndex, d_diagIndex, x);
        apply_invU_w(d_colPtr, d_rowIndex, d_diagIndex, y);
    }
}

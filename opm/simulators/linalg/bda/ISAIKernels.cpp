#include <opm/simulators/linalg/bda/ISAIKernels.hpp>

namespace bda{
    std::string get_isai_L_string(){
        return R"(
        __kernel void block_mult_sub(__global double *a, __global double *b, __global double *c)
        {
            const unsigned int bs = 3;
            const unsigned int warpsize = 32;
            const unsigned int num_active_threads = (warpsize / bs / bs) * bs * bs;
            const unsigned int idx_t = get_local_id(0);
            const unsigned int lane = idx_t % warpsize;

            if(lane < num_active_threads){
                const unsigned int row = lane % bs;
                const unsigned int col = (lane / bs) % bs;
                double temp = 0.0;

                for (unsigned int k = 0; k < bs; k++) {
                    temp += b[bs * row + k] * c[bs * k + col];
                }

                a[bs * row + col] -= temp;
            }
        }

        __kernel void isai_L(__global const int *mapping,
                            __global const int *colPtr,
                            __global const int *rowIndex,
                            __global const int *diagIndex,
                            __global const double *LU,
                            __global double *invLU,
                            const unsigned int Nb)
        {
            const unsigned int warpsize = 32;
            const unsigned int idx_b = get_group_id(0);
            const unsigned int idx_t = get_local_id(0);
            const unsigned int idx = get_global_id(0);
            const unsigned int bs = 3;
            const unsigned int num_threads = get_global_size(0);
            const unsigned int num_warps_in_grid = num_threads / warpsize;
            const unsigned int num_active_threads = (warpsize / bs / bs) * bs * bs;
            const unsigned int num_blocks_per_warp = warpsize / bs / bs;
            const unsigned int lane = idx_t % warpsize;
            const unsigned int c = (lane / bs) % bs;
            const unsigned int r = lane % bs;
            unsigned int target_block_col = idx / warpsize;

            while(target_block_col < Nb){
                const unsigned int start_cptr = diagIndex[target_block_col];
                const unsigned int end_cptr = colPtr[target_block_col + 1];
                unsigned int curr_cptr = start_cptr;

                if(lane < num_active_threads){
                    for(; curr_cptr < end_cptr - 1; curr_cptr++){
                        unsigned int curr_rowi = rowIndex[curr_cptr];
                        unsigned int aux_cptr = curr_cptr + 1 + lane / bs / bs;

                        while(aux_cptr < end_cptr){
                            unsigned int aux_rowi = rowIndex[aux_cptr];
                            unsigned int aux_rptr = diagIndex[aux_rowi] - 1;

                            if(curr_cptr == start_cptr){
                                invLU[mapping[aux_rowi] * bs * bs + r * bs + c] -= LU[mapping[aux_rowi] * bs * bs + r * bs + c];
                            }
                            else{
                                while(aux_rptr >= colPtr[aux_rowi] && rowIndex[aux_rptr] != curr_rowi){
                                    aux_rptr--;
                                }

                                if(aux_rptr >= colPtr[aux_rowi] && rowIndex[aux_rptr] == curr_rowi){
                                    block_mult_sub(invLU + mapping[aux_rowi] * bs * bs, invLU + mapping[curr_rowi] * bs * bs, LU + rowIndex[aux_rptr] * bs * bs);
                                }
                            }

                            aux_cptr += num_blocks_per_warp;
                        }
                    }
                }

                target_block_col += num_warps_in_grid;
            }
        }
        )";
    }

    std::string get_isai_U_string(){
        return R"(
        __kernel void isai_U(__global const int *mapping,
                            __global const int *colPtr,
                            __global const int *rowIndex,
                            __global const int *diagIndex,
                            __global const double *LU,
                            __global double *invLU,
                            const unsigned int Nb)
        {
            const unsigned int warpsize = 32;
            const unsigned int idx_b = get_group_id(0);
            const unsigned int idx_t = get_local_id(0);
            const unsigned int idx = get_global_id(0);
            const unsigned int bs = 3;
            const unsigned int num_threads = get_global_size(0);
            const unsigned int num_warps_in_grid = num_threads / warpsize;
            const unsigned int num_active_threads = (warpsize / bs / bs) * bs * bs;
            const unsigned int num_blocks_per_warp = warpsize / bs / bs;
            const unsigned int lane = idx_t % warpsize;
            const unsigned int c = (lane / bs) % bs;
            const unsigned int r = lane % bs;
            unsigned int target_block_col = idx / warpsize;

            while(target_block_col < Nb){
                const unsigned int start_cptr = colPtr[target_block_col];
                const unsigned int end_cptr = diagIndex[target_block_col];
                unsigned int curr_cptr = end_cptr;
                unsigned int curr_rowi = rowIndex[curr_cptr];

                if(lane < num_active_threads){
                    for(; curr_cptr >= start_cptr; curr_cptr--){
                        unsigned int aux_cptr = curr_cptr - 1 - lane / bs / bs;

                        if(curr_cptr == end_cptr){
                            invLU[mapping[curr_rowi] * bs * bs + r * bs + c] -= LU[mapping[curr_rowi] * bs * bs + r * bs + c];
                        }

                        while(aux_cptr >= start_cptr){
                            unsigned int aux_rowi = rowIndex[aux_cptr];
                            unsigned int aux_rptr = diagIndex[aux_rowi];

                            while(aux_rptr < colPtr[aux_rowi + 1] && rowIndex[aux_rptr] != curr_rowi){
                                aux_rptr++;
                            }

                            if(aux_rptr < colPtr[aux_rowi + 1] && rowIndex[aux_rptr] == curr_rowi){
                                block_mult_sub(invLU + mapping[aux_rowi] * bs * bs, invLU + mapping[curr_rowi] * bs * bs, LU + rowIndex[aux_rptr] * bs * bs);
                            }

                            aux_cptr += num_blocks_per_warp;
                        }
                    }
                }

                target_block_col += num_warps_in_grid;
            }
        }
        )";
    }

    std::string get_apply_invL_string(){
        return R"(
        __kernel void apply_invL(__global const int *cols,
                                 __global const int *rows,
                                 __global const int *diagIndex,
                                 __global const double *invLU,
                                 __global const double *x,
                                 __global double *b,
                                 const unsigned int Nb,
                                 __local double *tmp)
        {
            const unsigned int warpsize = 32;
            const unsigned int idx_b = get_group_id(0);
            const unsigned int idx_t = get_local_id(0);
            const unsigned int idx = get_global_id(0);
            const unsigned int bs = 3;
            const unsigned int num_threads = get_global_size(0);
            const unsigned int num_warps_in_grid = num_threads / warpsize;
            const unsigned int num_active_threads = (warpsize / bs / bs) * bs * bs;
            const unsigned int num_blocks_per_warp = warpsize / bs / bs;
            const unsigned int lane = idx_t % warpsize;
            const unsigned int c = (lane / bs) % bs;
            const unsigned int r = lane % bs;
            unsigned int target_block_row = 1 + idx / warpsize; // first row of invL is only eye(3)

            while(target_block_row < Nb){
                unsigned int first_block = rows[target_block_row];
                unsigned int last_block = diagIndex[target_block_row];
                unsigned int block = first_block + lane / (bs*bs);
                double local_out = 0.0;

                if(lane < num_active_threads){
                    for(; block < last_block; block += num_blocks_per_warp){
                        double x_elem = x[cols[block]*bs + c];
                        double A_elem = invLU[block*bs*bs + c + r*bs];
                        local_out += x_elem * A_elem;
                    }
                }

                tmp[lane] = local_out;
                barrier(CLK_LOCAL_MEM_FENCE);

                for(unsigned int offset = 3; offset <= 24; offset <<= 1){
                    if (lane + offset < warpsize){
                        tmp[lane] += tmp[lane + offset];
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);
                }

                if(lane < bs){
                    unsigned int row = target_block_row*bs + lane;
                    b[row] = tmp[lane] + x[row]; // the diagonal of invL is eye(3)
                }

                target_block_row += num_warps_in_grid;
            }
        }
        )";
    }

    std::string get_apply_invU_string(){
        return R"(
        __kernel void apply_invU(__global const int *cols,
                                 __global const int *rows,
                                 __global const int *diagIndex,
                                 __global const double *invLU,
                                 __global const double *x,
                                 __global double *b,
                                 const unsigned int Nb,
                                 __local double *tmp)
        {
            const unsigned int warpsize = 32;
            const unsigned int idx_b = get_group_id(0);
            const unsigned int idx_t = get_local_id(0);
            const unsigned int idx = get_global_id(0);
            const unsigned int bs = 3;
            const unsigned int num_threads = get_global_size(0);
            const unsigned int num_warps_in_grid = num_threads / warpsize;
            const unsigned int num_active_threads = (warpsize / bs / bs) * bs * bs;
            const unsigned int num_blocks_per_warp = warpsize / bs / bs;
            const unsigned int lane = idx_t % warpsize;
            const unsigned int c = (lane / bs) % bs;
            const unsigned int r = lane % bs;
            unsigned int target_block_row = 1 + idx / warpsize;

            while(target_block_row < Nb){
                unsigned int first_block = diagIndex[target_block_row];
                unsigned int last_block = rows[target_block_row + 1];
                unsigned int block = first_block + lane / (bs*bs);
                double local_out = 0.0;

                if(lane < num_active_threads){
                    for(; block < last_block; block += num_blocks_per_warp){
                        double x_elem = x[cols[block]*bs + c];
                        double A_elem = invLU[block*bs*bs + c + r*bs];
                        local_out += x_elem * A_elem;
                    }
                }

                tmp[lane] = local_out;
                barrier(CLK_LOCAL_MEM_FENCE);

                for(unsigned int offset = 3; offset <= 24; offset <<= 1){
                    if (lane + offset < warpsize){
                        tmp[lane] += tmp[lane + offset];
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);
                }

                if(lane < bs){
                    unsigned int row = target_block_row*bs + lane;
                    b[row] = tmp[lane];
                }

                target_block_row += num_warps_in_grid;
            }
        }
        )";
    }
}

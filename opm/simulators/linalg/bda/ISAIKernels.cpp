#include <opm/simulators/linalg/bda/ISAIKernels.hpp>

namespace bda{
    std::string get_isai_L_string(){
        return R"(
        __kernel void block_sub(__global double *a, __global double *b)
        {
            const unsigned int bs = 3;
            const unsigned int warpsize = 32;
            const unsigned int num_active_threads = (warpsize / bs / bs) * bs * bs;
            const unsigned int idx_t = get_local_id(0);
            const unsigned int lane = idx_t % warpsize;

            if(lane < num_active_threads){
                const unsigned int row = lane % bs;
                const unsigned int col = (lane / bs) % bs;

                a[bs * row + col] -= b[bs * row + col];
            }
        }

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
            unsigned int tcol = idx / warpsize;

            while(tcol < Nb - 1){
                const unsigned int frow = diagIndex[tcol];
                const unsigned int lrow = colPtr[tcol + 1];
                const unsigned int nx = lrow - frow;

                if(lane < num_active_threads){
                    unsigned int xid = 1 + lane / bs / bs;

                    for(unsigned int sweep = 1; sweep < nx; sweep++){
                        while(xid < nx){
                            unsigned int xpos = mapping[rowIndex[frow + xid]];

                            if(sweep == 1){
                                block_sub(invLU + xpos * bs * bs, LU + xpos * bs * bs);
                            }
                            else if(xid >= sweep){
                                unsigned int dxpos = mapping[rowIndex[frow + sweep - 1]]; // dxpos -> determined (already calculated) x position
                                unsigned int ptr = diagIndex[rowIndex[frow + sweep - 1]];

                                for(; ptr < colPtr[rowIndex[frow + sweep]]; ptr++){
                                    if(rowIndex[ptr] == rowIndex[frow + xid]){
                                        block_mult_sub(invLU + xpos * bs * bs, invLU + dxpos * bs * bs, LU + mapping[rowIndex[ptr]] * bs * bs);
                                    }
                                }
                            }

                            xid += num_blocks_per_warp;
                        }
                    }
                }

                tcol += num_warps_in_grid;
            }
        }
        )";
    }

    std::string get_isai_U_string(){
        return R"(
        __kernel void block_add(__global double *a, __global double *b)
        {
            const unsigned int bs = 3;
            const unsigned int warpsize = 32;
            const unsigned int num_active_threads = (warpsize / bs / bs) * bs * bs;
            const unsigned int idx_t = get_local_id(0);
            const unsigned int lane = idx_t % warpsize;

            if(lane < num_active_threads){
                const unsigned int row = lane % bs;
                const unsigned int col = (lane / bs) % bs;

                a[bs * row + col] += b[bs * row + col];
            }
        }

        __kernel void block_local_copy(__local double *a, __global double *b)
        {
            const unsigned int bs = 3;
            const unsigned int warpsize = 32;
            const unsigned int num_active_threads = (warpsize / bs / bs) * bs * bs;
            const unsigned int idx_t = get_local_id(0);
            const unsigned int lane = idx_t % warpsize;

            if(lane < num_active_threads){
                const unsigned int row = lane % bs;
                const unsigned int col = (lane / bs) % bs;

                a[bs * row + col] = b[bs * row + col];
            }
        }

        __kernel void block_mult(__global double *a, __local double *b, __global double *c)
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

                a[bs * row + col] = temp;
            }
        }

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
            unsigned int tcol = idx / warpsize;
            __local double x_lcopy[3*3];

            while(tcol < Nb){
                const unsigned int frow = colPtr[tcol];
                const unsigned int lrow = diagIndex[tcol] + 1;
                const unsigned int nx = lrow - frow;

                if(lane < num_active_threads){
                    unsigned int xid = lane / bs / bs;

                    for(unsigned int sweep = 0; sweep <= nx; sweep++){
                        while(xid < nx){
                            unsigned int xpos = mapping[rowIndex[lrow - xid - 1]];

                            if(sweep == 0 && xid == 0){
                                block_add(invLU + xpos * bs * bs, LU + xpos * bs * bs);
                            }
                            else if(sweep > 0 && sweep < nx){
                                unsigned int dxpos = mapping[rowIndex[lrow - sweep]];
                                unsigned int ptr = colPtr[rowIndex[lrow - sweep + 1]];

                                for(; ptr < diagIndex[rowIndex[lrow - sweep + 1]]; ptr++){
                                    if(rowIndex[ptr] == rowIndex[lrow - xid - 1]){
                                        block_mult_sub(invLU + xpos * bs * bs, invLU + dxpos * bs * bs, LU + mapping[rowIndex[ptr]] * bs * bs);
                                    }
                                }
                            }
                            else if(xid > 0){ // sweep == nx
                                // add one more sweep that will multiply the X's by inverses of the diagonals!
                                unsigned int diagptr = diagIndex[rowIndex[lrow - xid]];
                                unsigned int diagpos = mapping[rowIndex[diagptr]];

                                block_local_copy(x_lcopy, invLU + xpos * bs * bs);
                                block_mult(invLU + xpos * bs * bs, x_lcopy, LU + diagpos * bs * bs);
                            }

                            xid += num_blocks_per_warp;
                        }
                    }
                }

                tcol += num_warps_in_grid;
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
            unsigned int target_block_row = idx / warpsize;

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

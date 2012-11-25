#include <stdio.h>

#define THREADS_PER_BLOCK 4

__global__ void
expand(int factor,
  int* dst_rows, int* dst_cols, float* dst_vals,
  int* src_rows, int* src_cols, float* src_vals)
{
    int src_row = threadIdx.x + blockIdx.x * blockDim.x;

    int base = src_rows[src_row];
    int per_row = src_rows[src_row+1] - src_rows[src_row];

    for(int j = src_rows[src_row]; j < src_rows[src_row+1]; j++) {
        float val = src_vals[j];
        int src_col = src_cols[j];
        for(int k = 0; k < factor; k++) {
            int dst_idx = j*factor + k*per_row + (j-base);
            dst_rows[ dst_idx ] = src_row*factor+k;
            dst_cols[ dst_idx ] = src_col*factor+k;
            dst_vals[ dst_idx ] = val;
        }
    }
}

extern "C" {

void
OsdCusparseExpand(int src_numrows, int factor,
    int* dst_rows, int* dst_cols, float* dst_vals,
    int* src_rows, int* src_cols, float* src_vals)
{
    expand<<<src_numrows/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(factor,
            dst_rows, dst_cols, dst_vals,
            src_rows, src_cols, src_vals);
}

} /* extern C */

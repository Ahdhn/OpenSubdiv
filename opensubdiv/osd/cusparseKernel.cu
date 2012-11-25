#include <stdio.h>

#define THREADS_PER_BLOCK 4

__global__ void
expand(int factor,
  int* dst_rows, int* dst_cols, float* dst_vals,
  int* src_rows, int* src_cols, float* src_vals)
{
    int src_row = threadIdx.x + blockIdx.x * blockDim.x;
    int v_per_row = src_rows[src_row+1] - src_rows[src_row];
    int base = src_rows[src_row];

    for(int src_idx = src_rows[src_row]; src_idx < src_rows[src_row+1]; src_idx++) {
        for(int k = 0; k < factor; k++) {
            int dst_idx = factor*base + k*v_per_row + src_idx-base;
            dst_rows[dst_idx] = factor * src_row + k;
            dst_cols[dst_idx] = factor * src_cols[src_idx] + k;
            dst_vals[dst_idx] = src_vals[src_idx];
        }
    }
}

extern "C" {

void
OsdCusparseExpand(int src_numrows, int factor,
    int* dst_rows, int* dst_cols, float* dst_vals,
    int* src_rows, int* src_cols, float* src_vals)
{
    expand<<<src_numrows,1>>>(factor,
            dst_rows, dst_cols, dst_vals,
            src_rows, src_cols, src_vals);
}

} /* extern C */

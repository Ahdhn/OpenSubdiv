#include <stdio.h>

#define THREADS_PER_BLOCK 4

__global__ void
expand(int factor, int* dst_rows, int* dst_cols, float* dst_vals, int* src_rows, int* src_cols, float* src_vals) {
    int src_index = threadIdx.x + blockIdx.x * blockDim.x;
    int rowi = ((src_rows[src_index] - 1) * factor) + 1;
    int coli = ((src_cols[src_index] - 1) * factor) + 1;
    int val = src_vals[src_index];

    int dst_index = src_index * factor;
    for(int k = 0; k < factor; k++) {
        dst_rows[dst_index+k] = rowi + k;
        dst_cols[dst_index+k] = coli + k;
        dst_vals[dst_index+k] = val;
    }
}

extern "C" {

void
OsdCusparseExpand(int src_nnz, int factor,
    int* dst_rows, int* dst_cols, float* dst_vals,
    int* src_rows, int* src_cols, float* src_vals)
{
    expand<<<src_nnz/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>
        (factor, dst_rows, dst_cols, dst_vals,
                 src_rows, src_cols, src_vals);
}

} /* extern C */

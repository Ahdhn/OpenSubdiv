#include <stdio.h>

#define THREADS_PER_BLOCK 512

__global__ void
expand(int src_numrows, int factor,
  int* dst_rows, int* dst_cols, float* dst_vals,
  int* src_rows, int* src_cols, float* src_vals)
{
    int src_row = threadIdx.x + blockIdx.x * blockDim.x;
    if (src_row >= src_numrows)
        return;

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

    #if 0
        // how its done on CPU
        int new_i = 0;
        for(int r = 0; r < m; r++) {
            for(int k = 0; k < nve; k++) {
                new_rows[r*nve + k] = new_i+1;
                for(int i = rows[r]; i < rows[r+1]; i++, new_i++) {
                    int col_one = cols[i-1];
                    float val = vals[i-1];
                    new_cols[new_i] = ((col_one-1)*nve + k) + 1;
                    new_vals[new_i] = val;
                }
            }
        }
    #endif
}

__global__ void
spmv(int m, int nnz, const int* M_rows, const int* M_cols, const float* M_vals, const float* V_in, float* V_out)
{
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    if (row >= m)
        return;

    register float answer = 0.0;
    int lb = M_rows[row],
        ub = M_rows[row+1];

    for(int offset = lb; offset < ub; offset++)
	answer += M_vals[offset] * V_in[ M_cols[offset] ];

    V_out[row] = answer;
}

extern "C" {

#include <cusparse_v2.h>

void
OsdCusparseExpand(int src_numrows, int factor,
    int* dst_rows, int* dst_cols, float* dst_vals,
    int* src_rows, int* src_cols, float* src_vals)
{
    int blks = (src_numrows + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    expand<<<blks,THREADS_PER_BLOCK>>>(src_numrows, factor,
            dst_rows, dst_cols, dst_vals,
            src_rows, src_cols, src_vals);
}

cusparseStatus_t
my_cusparseScsrmv(cusparseHandle_t handle, cusparseOperation_t transA,
    int m, int n, int nnz, float* alpha,
    cusparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA, const int *csrColIndA,
    const float *x, float* beta,
    float *y ) {

    const int* M_rows = csrRowPtrA;
    const int* M_cols = csrColIndA;
    const float* M_vals = csrValA;
    const float* V_in = x;
    float* V_out = y;

    int blks = (m + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    spmv<<<blks,THREADS_PER_BLOCK>>>(m, nnz, M_rows, M_cols, M_vals, V_in, V_out);

    return CUSPARSE_STATUS_SUCCESS;
}

} /* extern C */

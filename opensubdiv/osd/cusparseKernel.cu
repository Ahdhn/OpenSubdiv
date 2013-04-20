#include <iostream>

#include <stdio.h>
#include <assert.h>

#define THREADS_PER_BLOCK 1024
#define THREADS_PER_ROW   32

using namespace std;

__global__ void
expand(int src_numthreads, int nve,
  int* dst_rows, int* dst_cols, float* dst_vals,
  int* src_rows, int* src_cols, float* src_vals)
{
    int thread = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread >= src_numthreads)
        return;

    int r = thread / nve; // src_row
    int k = thread % nve; // replica number

    int i = src_rows[r]-1;
    int stride = src_rows[r+1]-src_rows[r];
    int dst_base = i*nve + k*stride;
    int src_base = src_rows[r];
    dst_rows[r*nve + k] = dst_base+1;

    for(i = src_rows[r]; i < src_rows[r+1]; i++) {
	    int offset = i - src_base;
	    int col = src_cols[i-1];
	    float val = src_vals[i-1];
	    dst_cols[dst_base+offset] = ((col-1)*nve + k) +1;
	    dst_vals[dst_base+offset] = val;
    }
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

__global__ void
logical_spmv_coo_kernel(const int m, const int n, const int k,
    const int  * __restrict__ rows, const int * __restrict__ cols, const float * __restrict__ vals,
    const float * __restrict__ v_in, float * __restrict__ v_out)
{
    // TODO
}

__global__ void
logical_spmv_ell_kernel(const int m, const int n, const int k,
    const int  * __restrict__ cols, const float * __restrict__ vals,
    const float * __restrict__ v_in, float * __restrict__ v_out)
{

    int row6   = threadIdx.x + blockIdx.x * blockDim.x,
        row    = row6 / 6,
        elem   = row6 % 6;

    if (row >= m)
        return;

    float sum = 0.0f;

    int lda = m + 256 - m % 256;
    for (int i = 0; i < k; i++) {
        float weight = vals[ row + i*lda ];
        int   col    = cols[ row + i*lda ];

        sum += weight * v_in[col*6 + elem];
    }

    v_out[row*6 + elem] = sum;
}

__global__ void
logical_spmv_csr_kernel(int m, int n,
    const int *rows, const int *cols, const float *vals,
    const float *v_in, float *v_out)
{
    __shared__ float cache[THREADS_PER_ROW*6];
    int offset = threadIdx.x,
        elem   = threadIdx.y;

    for (int row = blockIdx.x; row < m; row += gridDim.x) {

        int base = rows[row];
        int idx = base-1 + offset;
        int nnz = rows[row+1] - base;

        float weight = (offset < nnz) ? vals[idx]   : 0.0f;
        int      col = (offset < nnz) ? cols[idx]-1 : 0   ;

        cache[elem*THREADS_PER_ROW + offset] = weight * v_in[col*6+elem];

        __syncthreads();
        for (int j = blockDim.x/2; j != 0; j /= 2) {
            if (offset < j)
                cache[elem*THREADS_PER_ROW + offset] += cache[elem*THREADS_PER_ROW + offset + j];
            __syncthreads();
        }

        if (offset == 0)
            v_out[row*6+elem] = cache[elem*THREADS_PER_ROW];
    }
}


extern "C" {

#include <cusparse_v2.h>

void
OsdCusparseExpand(int src_numrows, int factor,
    int* dst_rows, int* dst_cols, float* dst_vals,
    int* src_rows, int* src_cols, float* src_vals)
{
    int numthreads = src_numrows * factor;
    int blks = (numthreads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    expand<<<blks,THREADS_PER_BLOCK>>>(numthreads, factor,
            dst_rows, dst_cols, dst_vals,
            src_rows, src_cols, src_vals);

    /* fix up value at end of row index array */
    int dst_sentinel, src_sentinel;
    cudaMemcpy(&src_sentinel, &src_rows[src_numrows], sizeof(int), cudaMemcpyDeviceToHost);
    dst_sentinel = (src_sentinel - 1) * factor + 1;
    cudaMemcpy(&dst_rows[src_numrows*factor], &dst_sentinel, sizeof(int), cudaMemcpyHostToDevice);
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

void
LogicalSpMV_ell(int m, int n, int k, int *ell_cols, float *ell_vals, int *coo_rows, int *coo_cols, float *coo_vals, float *v_in, float *v_out) {

    int nBlocks = (m*6 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    logical_spmv_ell_kernel<<<nBlocks,THREADS_PER_BLOCK>>>
        (m, n, k, ell_cols, ell_vals, v_in, v_out);

#if 0
    logical_spmv_coo_kernel<<<nBlocks,THREADS_PER_BLOCK>>>
        (m, n, k, coo_rows, coo_cols, coo_vals, v_in, v_out);
#endif
}

void
LogicalSpMV_csr(int m, int n, int k, int *rows, int *cols, float *vals, float *v_in, float *v_out) {
    assert( k <= THREADS_PER_ROW);

    int nBlocks = min(m, 32768);
    dim3 nThreads(THREADS_PER_ROW,6);
    logical_spmv_csr_kernel<<<nBlocks,nThreads>>>(m, n, rows, cols, vals, v_in, v_out);
}

} /* extern C */

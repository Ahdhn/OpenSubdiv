#include <iostream>

#include <stdio.h>
#include <assert.h>

#define THREADS_PER_BLOCK 256
#define COO_THREADS_PER_BLOCK_0 1024
#define COO_THREADS_PER_BLOCK_1 1024
#define COO_THREADS_PER_BLOCK_2 1024
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
logical_spmv_coo_kernelB(const int nnz,
    const int  * __restrict__ rows, const int * __restrict__ cols, float * __restrict__ vals,
    float * __restrict__ scratch, const float * __restrict__ v_in, float * __restrict__ v_out)
{
    int nz = threadIdx.x + blockIdx.x * blockDim.x;
    if (nz >= nnz)
        return;

    v_in  += cols[nz]*6;
    v_out += rows[nz]*6;

    float weight = vals[nz];

    float v[6];
    v[0] = weight * v_in[0];
    v[1] = weight * v_in[1];
    v[2] = weight * v_in[2];
    v[3] = weight * v_in[3];
    v[4] = weight * v_in[4];
    v[5] = weight * v_in[5];

    int e = nz % 6;
    atomicAdd( &v_out[e], v[e] ); e = (e+1) % 6;
    atomicAdd( &v_out[e], v[e] ); e = (e+1) % 6;
    atomicAdd( &v_out[e], v[e] ); e = (e+1) % 6;
    atomicAdd( &v_out[e], v[e] ); e = (e+1) % 6;
    atomicAdd( &v_out[e], v[e] ); e = (e+1) % 6;
    atomicAdd( &v_out[e], v[e] ); e = (e+1) % 6;
}

__global__ void
logical_spmv_coo_kernel0A(const int nnz,
    const int  * __restrict__ rows, const int * __restrict__ cols, float * __restrict__ vals,
    float * __restrict__ scratch, const float * __restrict__ v_in, float * __restrict__ v_out)
{
    int nz = threadIdx.x + blockIdx.x * blockDim.x;
    if (nz >= nnz)
        return;

    int col = cols[nz];
    float weight = vals[nz];

    scratch[nz+0*nnz] = weight * v_in[col*6+0];
    scratch[nz+1*nnz] = weight * v_in[col*6+1];
    scratch[nz+2*nnz] = weight * v_in[col*6+2];
    scratch[nz+3*nnz] = weight * v_in[col*6+3];
    scratch[nz+4*nnz] = weight * v_in[col*6+4];
    scratch[nz+5*nnz] = weight * v_in[col*6+5];
}

__global__ void
logical_spmv_coo_kernel1A(const int nnz, const int  * __restrict__ rows,
        const int stride, float * __restrict__ scratch)
{
    int tid = threadIdx.x;
    int nz = stride * (threadIdx.x + blockIdx.x * blockDim.x);
    if (nz >= nnz)
        return;

    int row = rows[nz];
    const int tpb = COO_THREADS_PER_BLOCK_1;
    int effectiveThreads = min(blockDim.x, nnz - blockIdx.x * blockDim.x);
    __shared__ float cache[6*tpb];
    __shared__ int row_cache[tpb];

    row_cache[tid] = row;
    for (int i = 0; i < 6; i++)
        cache[tid + i*tpb] = scratch[nz+i*nnz]; // TODO align to 512 byte boundaries

    __syncthreads();

    register float right0 = 0, right1 = 0, right2 = 0, right3 = 0, right4 = 0, right5 = 0;
    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        if (tid+offset < effectiveThreads && row == row_cache[tid+offset]) {
            right0 = cache[tid+offset + 0*tpb];
            right1 = cache[tid+offset + 1*tpb];
            right2 = cache[tid+offset + 2*tpb];
            right3 = cache[tid+offset + 3*tpb];
            right4 = cache[tid+offset + 4*tpb];
            right5 = cache[tid+offset + 5*tpb];
        }
        __syncthreads();
        cache[tid + 0*tpb] += right0; right0 = 0;
        cache[tid + 1*tpb] += right1; right1 = 0;
        cache[tid + 2*tpb] += right2; right2 = 0;
        cache[tid + 3*tpb] += right3; right3 = 0;
        cache[tid + 4*tpb] += right4; right4 = 0;
        cache[tid + 5*tpb] += right5; right5 = 0;
        __syncthreads();
    }

    for (int i = 0; i < 6; i++)
        scratch[nz+i*nnz] = cache[tid+i*tpb];
}

__global__ void
logical_spmv_coo_kernel2A(const int nnz, const int  * __restrict__ rows,
    float * __restrict__ scratch, float * __restrict__ v_out)
{
    int nz = threadIdx.x + blockIdx.x * blockDim.x;
    if (nz >= nnz)
        return;

    int row = rows[nz], prev = rows[nz-1];
    if (row != prev)
        for (int i = 0; i < 6; i++)
            v_out[row*6+i] += scratch[nz+i*nnz];
}


__global__ void
logical_spmv_ell_kernel(const int m, const int n, const int k,
    const int  * __restrict__ cols, const float * __restrict__ vals,
    const float * __restrict__ v_in, float * __restrict__ v_out)
{
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    if (row >= m)
        return;

    register float
        sum0 = 0.0f,
        sum1 = 0.0f,
        sum2 = 0.0f,
        sum3 = 0.0f,
        sum4 = 0.0f,
        sum5 = 0.0f;

    int lda = m + 512 - m % 512;

    for (int i = 0; i < k; i++) {
        int idx = row + i*lda;
        int col = cols[idx]*6;
        float weight = vals[idx];

        const float *vertex = &v_in[col];
        sum0 += weight * vertex[0];
        sum1 += weight * vertex[1];
        sum2 += weight * vertex[2];
        sum3 += weight * vertex[3];
        sum4 += weight * vertex[4];
        sum5 += weight * vertex[5];
    }

    __shared__ float cache[6*THREADS_PER_BLOCK];
    float *cache_vertex = &cache[6*threadIdx.x];
    cache_vertex[0] = sum0;
    cache_vertex[1] = sum1;
    cache_vertex[2] = sum2;
    cache_vertex[3] = sum3;
    cache_vertex[4] = sum4;
    cache_vertex[5] = sum5;

    int effectiveThreads = min(blockDim.x, m - blockIdx.x * blockDim.x);
    int offset = threadIdx.x, stride = effectiveThreads;
    v_out += 6 * (blockIdx.x * blockDim.x);

    __syncthreads();

    v_out[ offset ] = cache[ offset ]; offset += stride;
    v_out[ offset ] = cache[ offset ]; offset += stride;
    v_out[ offset ] = cache[ offset ]; offset += stride;
    v_out[ offset ] = cache[ offset ]; offset += stride;
    v_out[ offset ] = cache[ offset ]; offset += stride;
    v_out[ offset ] = cache[ offset ]; offset += stride;
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

inline int log2i( int n ) {
  int val = 0 ;
  while( n >>= 1 )
        val++;
  return val;
}

void
LogicalSpMV_ell(int m, int n, int k, int *ell_cols, float *ell_vals, const int coo_nnz, int *coo_rows, int *coo_cols, float *coo_vals, float *coo_scratch, float *v_in, float *v_out) {

    int nBlocks = (m + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    logical_spmv_ell_kernel<<<nBlocks,THREADS_PER_BLOCK>>>
        (m, n, k, ell_cols, ell_vals, v_in, v_out);

#define USE_COO_KERNEL_A 1
#if USE_COO_KERNEL_A
    nBlocks = (coo_nnz + COO_THREADS_PER_BLOCK_0 - 1) / COO_THREADS_PER_BLOCK_0;
    logical_spmv_coo_kernel0A<<<nBlocks,COO_THREADS_PER_BLOCK_0>>>
        (coo_nnz, coo_rows, coo_cols, coo_vals, coo_scratch, v_in, v_out);

    for(int nLeft = coo_nnz, stride = 1; nLeft > 0;
            nLeft /= COO_THREADS_PER_BLOCK_1,
            stride *= COO_THREADS_PER_BLOCK_1) {

        nBlocks = (nLeft + COO_THREADS_PER_BLOCK_1 - 1) / COO_THREADS_PER_BLOCK_1;
        printf("\ntree reduce: %d blocks, %d left, %d stride\n", nBlocks, nLeft, stride);
        logical_spmv_coo_kernel1A<<<nBlocks,COO_THREADS_PER_BLOCK_1>>>
            (coo_nnz, coo_rows, stride, coo_scratch);
    }

    nBlocks = (coo_nnz + COO_THREADS_PER_BLOCK_2 - 1) / COO_THREADS_PER_BLOCK_2;
    logical_spmv_coo_kernel2A<<<nBlocks,COO_THREADS_PER_BLOCK_2>>>
        (coo_nnz, coo_rows, coo_scratch, v_out);

#else
    nBlocks = (coo_nnz + COO_THREADS_PER_BLOCK_2 - 1) / COO_THREADS_PER_BLOCK_2;
    logical_spmv_coo_kernelB<<<nBlocks,COO_THREADS_PER_BLOCK_2>>>
        (coo_nnz, coo_rows, coo_cols, coo_vals, coo_scratch, v_in, v_out);
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

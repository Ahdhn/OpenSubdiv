#include <iostream>

#include <stdio.h>
#include <assert.h>

#define THREADS_PER_BLOCK 256
#define COO_THREADS_PER_BLOCK_0 1024
#define COO_THREADS_PER_BLOCK_1 1024
#define COO_THREADS_PER_BLOCK_2 1024
#define VVADD_THREADS_PER_BLOCK 1024
#define THREADS_PER_ROW   32

using namespace std;

__global__ void
vvadd_kernel(float *d, float *e, int n)
{
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    if (tid >= n)
        return;

    d[tid] += e[tid];
}

__global__ void
transpose(float *odata, float *idata, int m, int n)
{
    int src_tid = threadIdx.x + blockIdx.x*blockDim.x;
    if (src_tid >= m*n)
        return;

    int src_row = src_tid / n,
        src_col = src_tid % n,
        dst_row = src_col,
        dst_col = src_row,
        dst_tid = dst_row * m + dst_col;

    odata[dst_tid] = idata[src_tid];
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

    int lda = nnz + ((512/sizeof(float)) - (nnz % (512/sizeof(float))));

    scratch[nz+0*lda] = weight * v_in[col*6+0];
    scratch[nz+1*lda] = weight * v_in[col*6+1];
    scratch[nz+2*lda] = weight * v_in[col*6+2];
    scratch[nz+3*lda] = weight * v_in[col*6+3];
    scratch[nz+4*lda] = weight * v_in[col*6+4];
    scratch[nz+5*lda] = weight * v_in[col*6+5];
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
    __shared__ float _cache[6*tpb];
    __shared__ int row_cache[tpb];
    float *cache = &_cache[tid];

    int lda = nnz + ((512/sizeof(float)) - (nnz % (512/sizeof(float))));

    row_cache[tid] = row;
    cache[0*tpb] = scratch[nz+0*lda];
    cache[1*tpb] = scratch[nz+1*lda];
    cache[2*tpb] = scratch[nz+2*lda];
    cache[3*tpb] = scratch[nz+3*lda];
    cache[4*tpb] = scratch[nz+4*lda];
    cache[5*tpb] = scratch[nz+5*lda];

    __syncthreads();

    register float right0 = 0, right1 = 0, right2 = 0,
                   right3 = 0, right4 = 0, right5 = 0;

    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        if (tid+offset < effectiveThreads && row == row_cache[tid+offset]) {
            right0 = cache[offset + 0*tpb];
            right1 = cache[offset + 1*tpb];
            right2 = cache[offset + 2*tpb];
            right3 = cache[offset + 3*tpb];
            right4 = cache[offset + 4*tpb];
            right5 = cache[offset + 5*tpb];
        }
        __syncthreads();
        cache[0*tpb] += right0; right0 = 0;
        cache[1*tpb] += right1; right1 = 0;
        cache[2*tpb] += right2; right2 = 0;
        cache[3*tpb] += right3; right3 = 0;
        cache[4*tpb] += right4; right4 = 0;
        cache[5*tpb] += right5; right5 = 0;
        __syncthreads();
    }

    scratch[nz+0*lda] = cache[0*tpb];
    scratch[nz+1*lda] = cache[1*tpb];
    scratch[nz+2*lda] = cache[2*tpb];
    scratch[nz+3*lda] = cache[3*tpb];
    scratch[nz+4*lda] = cache[4*tpb];
    scratch[nz+5*lda] = cache[5*tpb];
}

__global__ void
logical_spmv_coo_kernel2A(const int nnz, const int  * __restrict__ rows,
    float * __restrict__ scratch, float * __restrict__ v_out)
{
    int nz = threadIdx.x + blockIdx.x * blockDim.x;
    if (nz >= nnz)
        return;

    int lda = nnz + ((512/sizeof(float)) - (nnz % (512/sizeof(float))));

    int row  = rows[nz],
        prev = rows[nz-1];

    if (row != prev)
        for (int i = 0; i < 6; i++)
            v_out[row*6+i] += scratch[nz+i*lda];
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

    int lda = m + ((512/sizeof(float)) - (m % (512/sizeof(float))));

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
OsdTranspose(float *odata, float *idata, int m, int n, cudaStream_t& stream) {
    int nBlocks = (m*n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    transpose<<<nBlocks,THREADS_PER_BLOCK,0,stream>>>(odata, idata, m, n);
}

void
LogicalSpMV_ell0_gpu(int m, int n, int k, int *ell_cols, float *ell_vals, float *v_in, float *v_out, cudaStream_t& stream) {

    int nBlocks = (m + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    logical_spmv_ell_kernel<<<nBlocks,THREADS_PER_BLOCK,0,stream>>>
        (m, n, k, ell_cols, ell_vals, v_in, v_out);
}

void
LogicalSpMV_coo0_gpu(int m, int n, const int coo_nnz, int *coo_rows, int *coo_cols, float *coo_vals, float *coo_scratch, float *v_in, float *v_out) {

    int nBlocks = (coo_nnz + COO_THREADS_PER_BLOCK_0 - 1) / COO_THREADS_PER_BLOCK_0;
    logical_spmv_coo_kernel0A<<<nBlocks,COO_THREADS_PER_BLOCK_0>>>
        (coo_nnz, coo_rows, coo_cols, coo_vals, coo_scratch, v_in, v_out);

    const int tpb1 = COO_THREADS_PER_BLOCK_1;
    for(int nLeft = coo_nnz, stride = 1; nLeft > 0; nLeft /= tpb1, stride *= tpb1) {
        nBlocks = (nLeft + COO_THREADS_PER_BLOCK_1 - 1) / COO_THREADS_PER_BLOCK_1;
        logical_spmv_coo_kernel1A<<<nBlocks,COO_THREADS_PER_BLOCK_1>>>
            (coo_nnz, coo_rows, stride, coo_scratch);
    }

    nBlocks = (coo_nnz + COO_THREADS_PER_BLOCK_2 - 1) / COO_THREADS_PER_BLOCK_2;
    logical_spmv_coo_kernel2A<<<nBlocks,COO_THREADS_PER_BLOCK_2>>>
        (coo_nnz, coo_rows, coo_scratch, v_out);
}

void
LogicalSpMV_csr(int m, int n, int k, int *rows, int *cols, float *vals, float *v_in, float *v_out) {
    assert( k <= THREADS_PER_ROW);

    int nBlocks = min(m, 32768);
    dim3 nThreads(THREADS_PER_ROW,6);
    logical_spmv_csr_kernel<<<nBlocks,nThreads>>>(m, n, rows, cols, vals, v_in, v_out);
}

void
vvadd(float *d, float *e, int n, cudaStream_t& stream) {
    int nBlocks = (n + VVADD_THREADS_PER_BLOCK - 1) / VVADD_THREADS_PER_BLOCK;
    vvadd_kernel<<<nBlocks,VVADD_THREADS_PER_BLOCK,0,stream>>>(d, e, n);
}

} /* extern C */

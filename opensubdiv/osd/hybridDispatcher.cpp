#include "../version.h"
#include "../osd/mutex.h"
#include "../osd/hybridDispatcher.h"
#include "../osd/cusparseDispatcher.h"
#include "../osd/mklKernel.h"
#include "../osd/spmvKernel.h"

#include <stdio.h>

#include <iostream>
#include <vector>
extern "C" {
#include <mkl_spblas.h>
}

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <nvToolsExt.h>

#include <omp.h>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

static cusparseHandle_t handle;

HybridCsrMatrix*
HybridCooMatrix::gemm(HybridCsrMatrix* rhs) {
    HybridCsrMatrix* lhs = new HybridCsrMatrix(this, rhs->nve);
    HybridCsrMatrix* answer = lhs->gemm(rhs);
    delete lhs;
    return answer;
}

HybridCsrMatrix::HybridCsrMatrix(int m, int n, int nnz, int nve) :
    CsrMatrix(m, n, nnz, nve), rows(NULL), cols(NULL), vals(NULL), ell_k(0)
{
    /* make cusparse matrix descriptor */
    cusparseCreateMatDescr(&desc);
    cusparseSetMatType(desc,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(desc,CUSPARSE_INDEX_BASE_ONE);

    cudaMalloc( &d_in_scratch,  n*nve*sizeof(float) );
    cudaMalloc( &d_out_scratch, m*nve*sizeof(float) );
}

HybridCsrMatrix::HybridCsrMatrix(const HybridCooMatrix* StagedOp, int nve) :
    CsrMatrix(StagedOp, nve), rows(NULL), cols(NULL), vals(NULL)
{
    /* make cusparse matrix descriptor */
    cusparseCreateMatDescr(&desc);
    cusparseSetMatType(desc,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(desc,CUSPARSE_INDEX_BASE_ONE);

    m = StagedOp->m;
    n = StagedOp->n;
    nnz = StagedOp->nnz;
    int *h_rows = (int*) malloc((m+1) * sizeof(int));
    int *h_cols = (int*) malloc(nnz * sizeof(int));
    float *h_vals = (float*) malloc(nnz * sizeof(float));

    int job[] = {
        2, // job(1)=2 (coo->csr with sorting)
        1, // job(2)=1 (zero-based indexing for csr matrix)
        1, // job(3)=1 (one-based indexing for coo matrix)
        0, // empty
        nnz, // job(5)=nnz (sets nnz for csr matrix)
        0  // job(6)=0 (all output arrays filled)
    };

    float* acoo = (float*) &StagedOp->vals[0];
    int* rowind = (int*) &StagedOp->rows[0];
    int* colind = (int*) &StagedOp->cols[0];
    int info;

    /* use mkl because cusparse doesn't offer sorting */
    g_matrixTimer.Start();
    {
        mkl_scsrcoo(job, &m, h_vals, h_cols, h_rows, &nnz, acoo, rowind, colind, &info);
    }
    g_matrixTimer.Stop();

    assert(info == 0);

    /* allocate device memory */
    cudaMalloc(&rows, (m+1) * sizeof(int));
    cudaMalloc(&cols, nnz * sizeof(int));
    cudaMalloc(&vals, nnz * sizeof(float));
    cudaMalloc( &d_in_scratch,  n*nve*sizeof(float) );
    cudaMalloc( &d_out_scratch, m*nve*sizeof(float) );

    /* copy data to device */
    cudaMemcpy(rows, &h_rows[0], (m+1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cols, &h_cols[0], nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(vals, &h_vals[0], nnz * sizeof(float), cudaMemcpyHostToDevice);

    /* cleanup */
    free(h_rows);
    free(h_cols);
    free(h_vals);
}

int
HybridCsrMatrix::NumBytes() {
    if (ell_k != 0)
        return m*ell_k*(sizeof(float)+sizeof(int)) +
               h_csr_rowPtrs.size()*sizeof(int) +
               h_csr_vals.size()*(sizeof(int) + sizeof(float));
    else
        return this->CsrMatrix::NumBytes();
}

void
HybridCsrMatrix::logical_spmv(float *d_out, float* d_in, float *h_in) {

    // compute ELL portion - asynchronous
    LogicalSpMV_ell0_gpu(m, n, ell_k, ell_cols, ell_vals, d_in, d_out, computeStream);

    // compute CSR portion - synchronous
    nvtxRangePushA("logical_spmv_csr_cpu");
    {
        LogicalSpMV_csr0_cpu(m, &h_csr_rowPtrs[0], &h_csr_colInds[0], &h_csr_vals[0], &h_in[0], &h_out[0]);
    }
    nvtxRangePop();

    // copy CSR results to GPU - asynchronous
    cudaMemcpyAsync(d_csr_out, h_out, m*nve*sizeof(float), cudaMemcpyHostToDevice, memStream);

    // wait for CSR/ELL updates to be in place
    cudaStreamSynchronize(memStream);
    cudaStreamSynchronize(computeStream);

    // combine CSR/ELL results
    vvadd(d_out, d_csr_out, m*nve, computeStream);
}

void
HybridCsrMatrix::spmv(float *d_out, float* d_in) {
    assert(!"Not implemented.");
}

HybridCsrMatrix*
HybridCsrMatrix::gemm(HybridCsrMatrix* B) {
    HybridCsrMatrix* A = this;
    int mm = A->m,
        nn = A->n,
        kk = B->n;
    assert(A->n == B->m);

    cusparseOperation_t transA = CUSPARSE_OPERATION_NON_TRANSPOSE,
                        transB = CUSPARSE_OPERATION_NON_TRANSPOSE;

    HybridCsrMatrix* C = new HybridCsrMatrix(mm, kk, 0, nve);

    /* check that we're in host pointer mode to get C->nnz */
    cusparsePointerMode_t pmode;
    cusparseGetPointerMode(handle, &pmode);
    assert(pmode == CUSPARSE_POINTER_MODE_HOST);

    cusparseStatus_t status;
    cudaMalloc(&C->rows, (mm+1) * sizeof(int));
    g_matrixTimer.Start();
    {
        status = cusparseXcsrgemmNnz(handle, transA, transB,
                mm, nn, kk,
                A->desc, A->nnz, A->rows, A->cols,
                B->desc, B->nnz, B->rows, B->cols,
                C->desc, C->rows, &C->nnz);
    }
    g_matrixTimer.Stop();
    cusparseCheckStatus(status);

    cudaMalloc(&C->cols, C->nnz * sizeof(int));
    cudaMalloc(&C->vals, C->nnz * sizeof(float));
    g_matrixTimer.Start();
    {
        status = cusparseScsrgemm(handle, transA, transB,
                mm, nn, kk,
                A->desc, A->nnz, A->vals, A->rows, A->cols,
                B->desc, B->nnz, B->vals, B->rows, B->cols,
                C->desc, C->vals, C->rows, C->cols);
    }
    g_matrixTimer.Stop();
    cusparseCheckStatus(status);

    return C;
}

HybridCsrMatrix::~HybridCsrMatrix() {
    /* clean up device memory */
    cusparseDestroyMatDescr(desc);

    cudaFree(d_in_scratch);
    cudaFree(d_out_scratch);
}

void
HybridCsrMatrix::dump(std::string ofilename) {
    assert(!"No support for dumping matrices to file on GPUs. Use MKL kernel.");
}

void
HybridCsrMatrix::ellize() {

    omp_set_num_threads( omp_get_num_procs() );

    std::vector<float> h_full_vals(nnz);
    std::vector<int> h_full_rows(m+1);
    std::vector<int> h_full_cols(nnz);

    cudaMemcpy(&h_full_rows[0], rows, (m+1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_full_cols[0], cols, (nnz) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_full_vals[0], vals, (nnz) * sizeof(float), cudaMemcpyDeviceToHost);

    // determine width of ELL table using Bell and Garland's approach
    std::vector<int> histogram(40, 0);
    for (int i = 0; i < m; i++)
        histogram[ h_full_rows[i+1] - h_full_rows[i] ] += 1;

    std::vector<int> cdf(40, 0);
    for (int i = 38; i >= 0; i--)
        cdf[i] = histogram[i] + cdf[i+1];

    int k = 16;
    while ( cdf[k] > std::max(4096, m/3) && k < 39)
        k++;

    int lda = m + ((512/sizeof(float)) - (m % (512/sizeof(float))));
    std::vector<float> h_ell_vals(lda*k, 0.0f);
    std::vector<int>   h_ell_cols(lda*k, 0);

    h_csr_rowPtrs.clear(); h_csr_rowPtrs.reserve(m+1);
    h_csr_colInds.clear(); h_csr_colInds.reserve(nnz/4);
    h_csr_vals.clear();    h_csr_vals.reserve(nnz/4);

    cudaMallocHost(&h_out, m*nve*sizeof(float));
    memset(h_out, 0, m*nve*sizeof(float));

    for (int i = 0; i < m; i++) {
        int j, z;
        // regular part
        // convert to zero-based indices while we're at it...
        for (j = h_full_rows[i]-1, z = 0; j < h_full_rows[i+1]-1 && z < k; j++, z++) {
            h_ell_cols[ i + z*lda ] = h_full_cols[j]-1;
            h_ell_vals[ i + z*lda ] = h_full_vals[j];
        }
        // irregular part
        int row = h_csr_vals.size();
        h_csr_rowPtrs.push_back(row);
        for ( ; j < h_full_rows[i+1]-1; j++) {
            int col = h_full_cols[j]-1;
            float val = h_full_vals[j];

            h_csr_colInds.push_back(col);
            h_csr_vals.push_back(val);

            assert( 0 <= row && row < nnz);
            assert( 0 <= col && col < n );
        }
    }
    h_csr_rowPtrs.push_back(h_csr_vals.size());

    assert(h_csr_rowPtrs.size() == m+1);

#if BENCHMARKING
    printf(" irreg=%d m=%d k=%d", (int) h_csr_vals.size(), m, k);
#endif

    ell_k = k;
    cudaMalloc(&ell_vals, h_ell_vals.size() * sizeof(float));
    cudaMalloc(&ell_cols, h_ell_cols.size() * sizeof(int));
    cudaMemcpy(ell_vals, &h_ell_vals[0], h_ell_vals.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(ell_cols, &h_ell_cols[0], h_ell_cols.size() * sizeof(int),   cudaMemcpyHostToDevice);

    cudaMalloc(&d_csr_out, m*nve*sizeof(float));

    cudaStreamCreate(&memStream);
    cudaStreamCreate(&computeStream);
}

OsdHybridKernelDispatcher::OsdHybridKernelDispatcher(int levels) :
    super(levels, true) {
    /* make cusparse handle if null */
    assert (handle == NULL);
    cusparseCreate(&handle);
}

OsdHybridKernelDispatcher::~OsdHybridKernelDispatcher() {
    /* clean up cusparse handle */
    cusparseDestroy(handle);
    handle = NULL;
}

void
OsdHybridKernelDispatcher::FinalizeMatrix() {
    SubdivOp->ellize();
    this->super::FinalizeMatrix();
}

static OsdHybridKernelDispatcher::OsdKernelDispatcher *
Create(int levels) {
    return new OsdHybridKernelDispatcher(levels);
}

void
OsdHybridKernelDispatcher::Register() {
    Factory::GetInstance().Register(Create, kHYB);
}

void
OsdHybridKernelDispatcher::Synchronize()
{
    cudaDeviceSynchronize();
}

} // end namespace OPENSUBDIV_VERSION

} // end namespace OpenSubdiv

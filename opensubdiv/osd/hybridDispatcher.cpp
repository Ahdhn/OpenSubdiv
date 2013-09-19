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

int g_HybridSplitParam = -1;

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
               h_coo_vals.size()*(2*sizeof(int) + sizeof(float));
    else
        return this->CsrMatrix::NumBytes();
}

void
HybridCsrMatrix::logical_spmv(float *d_out, float* d_in, float *h_in) {

    // compute ELL portion - asynchronous
    LogicalSpMV_ell0_gpu(m, n, ell_k, ell_cols, ell_vals, d_in, d_out, computeStream);

    // compute CSR portion - synchronous
    if (h_coo_vals.size() > 0) {
        nvtxRangePushA("logical_spmv_cpu");
        {
            LogicalSpMV_coo0_cpu(
                &h_coo_schedule[0], &h_coo_offsets[0],
                &h_coo_rowInds[0], &h_coo_colInds[0], &h_coo_vals[0],
                &h_in[0], &h_coo_out_inds[0], &h_coo_out_vals[0]);
        }
        nvtxRangePop();

        // copy CSR results to GPU - asynchronous
        int nOutVals = h_coo_offsets[omp_get_max_threads()];
        cudaMemcpyAsync(d_coo_out_inds, h_coo_out_inds, nOutVals*sizeof(int),       cudaMemcpyHostToDevice, memStream);
        cudaMemcpyAsync(d_coo_out_vals, h_coo_out_vals, nOutVals*nve*sizeof(float), cudaMemcpyHostToDevice, memStream);

        // wait for CSR/ELL updates to be in place
        cudaStreamSynchronize(memStream);
        cudaStreamSynchronize(computeStream);

        // combine CSR/ELL results
        spvvadd(d_out, d_coo_out_inds, d_coo_out_vals, nOutVals*nve, computeStream);
    }
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
    if (g_HybridSplitParam != -1)
        k = g_HybridSplitParam;
    else
        while ( cdf[k] > std::max(4096, m/3) && k < 39)
            k++;

    int lda = m + ((512/sizeof(float)) - (m % (512/sizeof(float))));
    std::vector<float> h_ell_vals(lda*k, 0.0f);
    std::vector<int>   h_ell_cols(lda*k, 0);

    h_coo_rowInds.clear(); h_coo_rowInds.reserve(nnz/4);
    h_coo_colInds.clear(); h_coo_colInds.reserve(nnz/4);
    h_coo_vals.clear();    h_coo_vals.reserve(nnz/4);

    int nIrregRows = 0;

    for (int i = 0; i < m; i++) {
        int j, z;
        // regular part
        // convert to zero-based indices while we're at it...
        for (j = h_full_rows[i]-1, z = 0; j < h_full_rows[i+1]-1 && z < k; j++, z++) {
            h_ell_cols[ i + z*lda ] = h_full_cols[j]-1;
            h_ell_vals[ i + z*lda ] = h_full_vals[j];
        }

        // irregular part
        if (j < h_full_rows[i+1]-1) nIrregRows += 1;

        for ( ; j < h_full_rows[i+1]-1; j++) {
            int col = h_full_cols[j]-1;
            float val = h_full_vals[j];

            h_coo_rowInds.push_back(i);
            h_coo_colInds.push_back(col);
            h_coo_vals.push_back(val);

            assert( 0 <= i   &&   i < m );
            assert( 0 <= col && col < n );
        }
    }

#if BENCHMARKING
    printf(" irreg=%d n_irreg_rows=%d m=%d k=%d", (int) h_coo_vals.size(), nIrregRows, m, k);
#endif

    // build schedule for cpu coo evaluation
    int nThreads = omp_get_max_threads();
    int cooNnzPerThread = h_coo_vals.size() / nThreads;
    h_coo_schedule.resize(nThreads+1);
    h_coo_schedule[0] = 0;
    for (int i = 1; i < nThreads; i++) {
        int cooIndex = i*cooNnzPerThread;
        // advance the index until we have our own row
        // XXX doesn't handle every possible matrix
        while (h_coo_rowInds[cooIndex-1] == h_coo_rowInds[cooIndex])
            cooIndex++;
        h_coo_schedule[i] = cooIndex;

    }
    h_coo_schedule[nThreads] = h_coo_vals.size();

    // compute offsets
    h_coo_offsets.resize(nThreads+1);
    h_coo_offsets[0] = 0;
    for (int t = 0; t < nThreads; t++) {
        int nRows = 0;
        int prevRow = -1;
        for (int idx = h_coo_schedule[t]; idx < h_coo_schedule[t+1]; idx++) {
            int row = h_coo_rowInds[idx];
            if (row != prevRow)
                nRows += 1;
            prevRow = row;
        }
        h_coo_offsets[t+1] = h_coo_offsets[t] + nRows;
    }

    // allocate space for sparse output vector on host
    int nOutVals = h_coo_offsets[nThreads];
    cudaMallocHost(&h_coo_out_inds, nOutVals*sizeof(int));
    cudaMalloc    (&d_coo_out_inds, nOutVals*sizeof(int));
    cudaMalloc    (&d_coo_out_vals, nOutVals*nve*sizeof(float));
    cudaMallocHost(&h_coo_out_vals, nOutVals*nve*sizeof(float));

    ell_k = k;
    cudaMalloc(&ell_vals, h_ell_vals.size() * sizeof(float));
    cudaMalloc(&ell_cols, h_ell_cols.size() * sizeof(int));
    cudaMemcpy(ell_vals, &h_ell_vals[0], h_ell_vals.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(ell_cols, &h_ell_cols[0], h_ell_cols.size() * sizeof(int),   cudaMemcpyHostToDevice);

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

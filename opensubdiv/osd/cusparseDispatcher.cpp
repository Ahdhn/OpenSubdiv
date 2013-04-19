#include "../version.h"
#include "../osd/mutex.h"
#include "../osd/cusparseDispatcher.h"
#include "../osd/spmvKernel.h"

#include <stdio.h>

#include <iostream>
#include <vector>
extern "C" {
#include <mkl_spblas.h>
}


#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

extern "C" {
void
OsdCusparseExpand(int src_m, int factor, int* dst_rows, int* dst_cols, float* dst_vals, int* src_rows, int* src_cols, float* src_vals);

cusparseStatus_t
my_cusparseScsrmv(cusparseHandle_t handle, cusparseOperation_t transA,
    int m, int n, int nnz, float* alpha,
    cusparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA, const int *csrColIndA,
    const float *x, float* beta,
    float *y );

void
LogicalSpMV_ell(int m, int n, int k, int *cols, float *vals, float *v_in, float *v_out);
void LogicalSpMV_csr(int m, int n, int k, int *rows, int *cols, float *vals, float *v_in, float *v_out);

}

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

static cusparseHandle_t handle = NULL;

inline void
cusparseCheckStatus(cusparseStatus_t status) {
    if (status != CUSPARSE_STATUS_SUCCESS)
        switch (status) {
            case CUSPARSE_STATUS_NOT_INITIALIZED:  printf("bad status 1: CUSPARSE_STATUS_NOT_INITIALIZED\n"); break;
            case CUSPARSE_STATUS_ALLOC_FAILED:     printf("bad status 1: CUSPARSE_STATUS_ALLOC_FAILED\n"); break;
            case CUSPARSE_STATUS_INVALID_VALUE:    printf("bad status 1: CUSPARSE_STATUS_INVALID_VALUE\n"); break;
            case CUSPARSE_STATUS_ARCH_MISMATCH:    printf("bad status 1: CUSPARSE_STATUS_ARCH_MISMATCH\n"); break;
            case CUSPARSE_STATUS_MAPPING_ERROR:    printf("bad status 1: CUSPARSE_STATUS_MAPPING_ERROR\n"); break;
            case CUSPARSE_STATUS_EXECUTION_FAILED: printf("bad status 1: CUSPARSE_STATUS_EXECUTION_FAILED\n"); break;
            case CUSPARSE_STATUS_INTERNAL_ERROR:   printf("bad status 1: CUSPARSE_STATUS_INTERNAL_ERROR\n"); break;
            case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED: printf("bad status 1: CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED\n"); break;
            default: printf("bad status 2: unknown (%d)\n", status); break;
        }
    assert(status == CUSPARSE_STATUS_SUCCESS);
}

CudaCsrMatrix*
CudaCooMatrix::gemm(CudaCsrMatrix* rhs) {
    CudaCsrMatrix* lhs = new CudaCsrMatrix(this, rhs->nve);
    CudaCsrMatrix* answer = lhs->gemm(rhs);
    delete lhs;
    return answer;
}

CudaCsrMatrix::CudaCsrMatrix(int m, int n, int nnz, int nve, mode_t mode) :
    CsrMatrix(m, n, nnz, nve, mode), rows(NULL), cols(NULL), vals(NULL), hyb(NULL)
{
    /* make cusparse matrix descriptor */
    cusparseCreateMatDescr(&desc);
    cusparseSetMatType(desc,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(desc,CUSPARSE_INDEX_BASE_ONE);
}

CudaCsrMatrix::CudaCsrMatrix(const CudaCooMatrix* StagedOp, int nve, mode_t mode) :
    CsrMatrix(StagedOp, nve, mode), rows(NULL), cols(NULL), vals(NULL), hyb(NULL)
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
    mkl_scsrcoo(job, &m, h_vals, h_cols, h_rows, &nnz, acoo, rowind, colind, &info);
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

void
CudaCsrMatrix::logical_spmv(float *d_out, float* d_in) {
    LogicalSpMV_ell(m, n, ell_k, ell_cols, ell_vals, d_in, d_out);
    //LogicalSpMV_csr(m, n, ell_k, rows, cols, vals, d_in, d_out);
}

void
CudaCsrMatrix::spmv(float *d_out, float* d_in) {
    cusparseStatus_t status;
    cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
    float alpha = 1.0,
          beta = 0.0;
    status = cusparseShybmv(handle, op, &alpha, desc, hyb, d_in, &beta, d_out);
    cusparseCheckStatus(status);
}

CudaCsrMatrix*
CudaCsrMatrix::gemm(CudaCsrMatrix* B) {
    CudaCsrMatrix* A = this;
    int mm = A->m,
        nn = A->n,
        kk = B->n;
    assert(A->n == B->m);

    cusparseOperation_t transA = CUSPARSE_OPERATION_NON_TRANSPOSE,
                        transB = CUSPARSE_OPERATION_NON_TRANSPOSE;

    CudaCsrMatrix* C = new CudaCsrMatrix(mm, kk, 0, nve);

    /* check that we're in host pointer mode to get C->nnz */
    cusparsePointerMode_t pmode;
    cusparseGetPointerMode(handle, &pmode);
    assert(pmode == CUSPARSE_POINTER_MODE_HOST);

    cusparseStatus_t status;
    cudaMalloc(&C->rows, (mm+1) * sizeof(int));
    status = cusparseXcsrgemmNnz(handle, transA, transB,
            mm, nn, kk,
            A->desc, A->nnz, A->rows, A->cols,
            B->desc, B->nnz, B->rows, B->cols,
            C->desc, C->rows, &C->nnz);
    cusparseCheckStatus(status);

    cudaMalloc(&C->cols, C->nnz * sizeof(int));
    cudaMalloc(&C->vals, C->nnz * sizeof(float));
    status = cusparseScsrgemm(handle, transA, transB,
            mm, nn, kk,
            A->desc, A->nnz, A->vals, A->rows, A->cols,
            B->desc, B->nnz, B->vals, B->rows, B->cols,
            C->desc, C->vals, C->rows, C->cols);
    cusparseCheckStatus(status);

    return C;
}

CudaCsrMatrix::~CudaCsrMatrix() {
    /* clean up device memory */
    cusparseDestroyMatDescr(desc);

    if (hyb != NULL)
        cusparseDestroyHybMat(hyb);
}

void
CudaCsrMatrix::expand() {
    if (mode == CsrMatrix::VERTEX) {
        int *new_rows, *new_cols;
        float *new_vals;
        cudaMalloc(&new_rows, (nve*m+1) * sizeof(int));
        cudaMalloc(&new_cols, nve*nnz * sizeof(int));
        cudaMalloc(&new_vals, nve*nnz * sizeof(float));

        OsdCusparseExpand(m, nve, new_rows, new_cols, new_vals, rows, cols, vals);

        cudaFree(rows);
        cudaFree(cols);
        cudaFree(vals);

        m *= nve;
        n *= nve;
        nnz *= nve;
        rows = new_rows;
        cols = new_cols;
        vals = new_vals;
        mode = CsrMatrix::ELEMENT;
    }

    assert(hyb == NULL);
    cusparseCreateHybMat(&hyb);
    cusparseScsr2hyb(handle, m, n, desc, vals, rows, cols, hyb, 0, CUSPARSE_HYB_PARTITION_MAX);

    cudaFree(rows);
    cudaFree(cols);
    cudaFree(vals);
}

void
CudaCsrMatrix::dump(std::string ofilename) {
    assert(!"No support for dumping matrices to file on GPUs. Use MKL kernel.");
}

void
CudaCsrMatrix::ellize() {
    std::vector<float> h_vals(nnz);
    std::vector<int> h_rows(m+1);
    std::vector<int> h_cols(nnz);

    cudaMemcpy(&h_rows[0], rows, (m+1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_cols[0], cols, (nnz) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_vals[0], vals, (nnz) * sizeof(float), cudaMemcpyDeviceToHost);

    int k = 16;

    std::vector<float> h_ell_vals(m*k, 0.0f);
    std::vector<int>   h_ell_cols(m*k, 0);

    std::vector<float> h_coo_vals;
    std::vector<int>   h_coo_rows,
                       h_coo_cols;

    // convert to zero-based indices while we're at it...
    for (int i = 0; i < m; i++) {
        int j, z;
        // regular part
        for (j = h_rows[i]-1, z = 0; j < h_rows[i+1]-1 && z < k; j++, z++) {
            h_ell_cols[ i*k + z ] = h_cols[j]-1;
            h_ell_vals[ i*k + z ] = h_vals[j];
        }
        // irregular part
        for ( ; j < h_rows[i+1]-1; j++) {
            h_coo_rows.push_back( i           );
            h_coo_cols.push_back( h_cols[j]-1 );
            h_coo_vals.push_back( h_vals[j]   );
        }
    }

    ell_k = k;
    cudaMalloc(&ell_vals, (m*k) * sizeof(float));
    cudaMalloc(&ell_cols, (m*k) * sizeof(int));
    cudaMemcpy(ell_vals, &h_ell_vals[0], (m*k) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(ell_cols, &h_ell_cols[0], (m*k) * sizeof(int),   cudaMemcpyHostToDevice);
}

OsdCusparseKernelDispatcher::OsdCusparseKernelDispatcher(int levels, bool logical) :
    OsdSpMVKernelDispatcher<CudaCooMatrix,
                            CudaCsrMatrix,
                            OsdCudaVertexBuffer>(levels, logical) {
    /* make cusparse handle if null */
    assert (handle == NULL);
    cusparseCreate(&handle);
}

OsdCusparseKernelDispatcher::~OsdCusparseKernelDispatcher() {
    /* clean up cusparse handle */
    cusparseDestroy(handle);
    handle = NULL;
}

void
OsdCusparseKernelDispatcher::FinalizeMatrix() {
    this->OsdSpMVKernelDispatcher<CudaCooMatrix, CudaCsrMatrix, OsdCudaVertexBuffer>::FinalizeMatrix();

    if (logical)
        SubdivOp->ellize();
}

static OsdCusparseKernelDispatcher::OsdKernelDispatcher *
Create(int levels) {
    return new OsdCusparseKernelDispatcher(levels, false);
}

static OsdCusparseKernelDispatcher::OsdKernelDispatcher *
CreateLogical(int levels) {
    return new OsdCusparseKernelDispatcher(levels, true);
}

void
OsdCusparseKernelDispatcher::Register() {
    Factory::GetInstance().Register(Create, kCUSPARSE);
    Factory::GetInstance().Register(CreateLogical, kCGPU);
}

void
OsdCusparseKernelDispatcher::Synchronize()
{
    cudaThreadSynchronize();
}

} // end namespace OPENSUBDIV_VERSION

} // end namespace OpenSubdiv

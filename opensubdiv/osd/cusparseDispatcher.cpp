#include "../version.h"
#include "../osd/mutex.h"
#include "../osd/cusparseDispatcher.h"
#include "../osd/spmvKernel.h"

#include <stdio.h>

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
}

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

static cusparseHandle_t handle = NULL;

CudaCsrMatrix*
CudaCooMatrix::gemm(CudaCsrMatrix* rhs) {
    CudaCsrMatrix* lhs = new CudaCsrMatrix(this);
    CudaCsrMatrix* answer = lhs->gemm(rhs);
    delete lhs;
    return answer;
}

CudaCsrMatrix::CudaCsrMatrix(int m, int n, int nnz, int nve, mode_t mode) :
    CsrMatrix(m, n, nnz, nve, mode), rows(NULL), cols(NULL), vals(NULL)
{
    /* make cusparse matrix descriptor */
    cusparseCreateMatDescr(&desc);
    cusparseSetMatType(desc,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(desc,CUSPARSE_INDEX_BASE_ZERO);

    /* make cusparse handle if null */
    if (handle == NULL)
        cusparseCreate(&handle);
}

CudaCsrMatrix::CudaCsrMatrix(const CudaCooMatrix* StagedOp, int nve, mode_t mode) :
    CsrMatrix(StagedOp, nve, mode), rows(NULL), cols(NULL), vals(NULL)
{
    /* make cusparse matrix descriptor */
    cusparseCreateMatDescr(&desc);
    cusparseSetMatType(desc,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(desc,CUSPARSE_INDEX_BASE_ZERO);

    /* make cusparse handle if null */
    if (handle == NULL)
        cusparseCreate(&handle);

    /* allocate device memory */
    cudaMalloc(&rows, (StagedOp->m+1) * sizeof(int));
    cudaMalloc(&cols, StagedOp->nnz * sizeof(int));
    cudaMalloc(&vals, StagedOp->nnz * sizeof(float));

    /* copy data to device */
    cudaMemcpy(rows, &StagedOp->rows[0], StagedOp->rows.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cols, &StagedOp->cols[0], StagedOp->cols.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(vals, &StagedOp->vals[0], StagedOp->vals.size() * sizeof(float), cudaMemcpyHostToDevice);
}

void
CudaCsrMatrix::spmv(float *d_out, float* d_in) {
    cusparseStatus_t status;
    cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
    float alpha = 1.0,
          beta = 0.0;
    status = cusparseScsrmv(handle, op, m, n, nnz, &alpha, desc,
            vals, rows, cols, d_in, &beta, d_out);
    assert(status == CUSPARSE_STATUS_SUCCESS);
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

    CudaCsrMatrix* C = new CudaCsrMatrix(mm, kk);

    cusparseStatus_t status;
    cudaMalloc(&C->rows, (mm+1) * sizeof(int));
    status = cusparseXcsrgemmNnz(handle, transA, transB,
            mm, nn, kk,
            A->desc, A->nnz, A->rows, A->cols,
            B->desc, B->nnz, B->rows, B->cols,
            C->desc, C->rows, &C->nnz);
    assert(status == CUSPARSE_STATUS_SUCCESS);

    cudaMalloc(&C->cols, C->nnz * sizeof(int));
    cudaMalloc(&C->vals, C->nnz * sizeof(float));
    status = cusparseScsrgemm(handle, transA, transB,
            mm, nn, kk,
            A->desc, A->nnz, A->vals, A->rows, A->cols,
            B->desc, B->nnz, B->vals, B->rows, B->cols,
            C->desc, C->vals, C->rows, C->cols);
    assert(status == CUSPARSE_STATUS_SUCCESS);

    return C;
}

CudaCsrMatrix::~CudaCsrMatrix() {
    /* clean up device memory */
    cusparseDestroyMatDescr(desc);
    cudaFree(rows);
    cudaFree(cols);
    cudaFree(vals);
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
        cudaMemcpy(rows, &nnz, sizeof(int), cudaMemcpyHostToDevice);
    }
}

void
CudaCsrMatrix::dump(std::string ofilename) {
    assert(!"No support for dumping matrices on GPUs. Use MKL kernel.");
}

OsdCusparseKernelDispatcher::OsdCusparseKernelDispatcher(int levels) :
    OsdSpMVKernelDispatcher<CudaCooMatrix,CudaCsrMatrix,OsdCudaVertexBuffer>(levels)
{ }

static OsdCusparseKernelDispatcher::OsdKernelDispatcher *
Create(int levels) {
    return new OsdCusparseKernelDispatcher(levels);
}

void
OsdCusparseKernelDispatcher::Register() {
    Factory::GetInstance().Register(Create, kCUSPARSE);
}

void
OsdCusparseKernelDispatcher::Synchronize()
{
    cudaThreadSynchronize();
}

} // end namespace OPENSUBDIV_VERSION

} // end namespace OpenSubdiv

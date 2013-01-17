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

#if 0
device_csr_matrix_view::device_csr_matrix_view(csr_matrix* M) :
    m(M->size1()), n(M->size2()), nnz(M->nnz()) {

    /* make cusparse matrix descriptor */
    cusparseCreateMatDescr(&desc);
    cusparseSetMatType(desc,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(desc,CUSPARSE_INDEX_BASE_ZERO);

    /* make cusparse handle if null */
    if (handle == NULL)
        cusparseCreate(&handle);

    /* alias csr vectors */
    std::vector<int> &r = M->index1_data();
    std::vector<int> &c = M->index2_data();
    std::vector<float> &v = M->value_data();

    /* allocate device memory */
    cudaMalloc(&rows, r.size() * sizeof(r[0]));
    cudaMalloc(&cols, c.size() * sizeof(c[0]));
    cudaMalloc(&vals, v.size() * sizeof(v[0]));

    /* copy data to device */
    cudaMemcpy(rows, &r[0], r.size() * sizeof(r[0]), cudaMemcpyHostToDevice);
    cudaMemcpy(cols, &c[0], c.size() * sizeof(c[0]), cudaMemcpyHostToDevice);
    cudaMemcpy(vals, &v[0], v.size() * sizeof(v[0]), cudaMemcpyHostToDevice);
}

device_csr_matrix_view::~device_csr_matrix_view() {
    /* clean up device memory */
    cusparseDestroyMatDescr(desc);
    cudaFree(rows);
    cudaFree(cols);
    cudaFree(vals);
}

void
device_csr_matrix_view::report(std::string name) {
    printf("%s: %d-%d, %d nnz, r/c/v: 0x%p 0x%p 0x%p\n",
            name.c_str(), m, n, nnz, rows, cols, vals);
}

void
device_csr_matrix_view::spmv(float* d_out, const float* d_in) {
    cusparseStatus_t status;
    cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
    float alpha = 1.0,
          beta = 0.0;
    status = cusparseScsrmv(handle, op, m, n, nnz, &alpha, desc,
            vals, rows, cols, d_in, &beta, d_out);
    assert(status == CUSPARSE_STATUS_SUCCESS);
}

device_csr_matrix_view*
device_csr_matrix_view::times(device_csr_matrix_view* B) {
    device_csr_matrix_view* A = this;
    int mm = A->m,
        nn = A->n,
        kk = B->n;
    assert(A->n == B->m);

    cusparseOperation_t transA = CUSPARSE_OPERATION_NON_TRANSPOSE,
                        transB = CUSPARSE_OPERATION_NON_TRANSPOSE;

    device_csr_matrix_view* C = new device_csr_matrix_view(mm, kk);

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

device_csr_matrix_view::device_csr_matrix_view(int m, int n, int nnz) :
    m(m), n(n), nnz(nnz), rows(NULL), cols(NULL), vals(NULL)
{
    /* make cusparse matrix descriptor */
    cusparseCreateMatDescr(&desc);
    cusparseSetMatType(desc,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(desc,CUSPARSE_INDEX_BASE_ZERO);

    /* make cusparse handle if null */
    if (handle == NULL)
        cusparseCreate(&handle);

    /* allocate space if nnz > 0 */
    if (nnz > 0) {
        cudaMalloc(&rows, (m+1) * sizeof(int));
        cudaMalloc(&cols, nnz * sizeof(int));
        cudaMalloc(&vals, nnz * sizeof(float));
    }
}

void
device_csr_matrix_view::expand(int factor) {
}
#endif

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

#include "../version.h"
#include "../osd/mutex.h"
#include "../osd/cusparseDispatcher.h"
#include "../osd/spmvKernel.h"

#include <stdio.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

device_csr_matrix_view::device_csr_matrix_view(csr_matrix1* M) :
    m(M->size1()), n(M->size2()), nnz(M->nnz()) {

    /* make cusparse matrix descriptor */
    cusparseCreateMatDescr(&desc);
    cusparseSetMatType(desc,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(desc,CUSPARSE_INDEX_BASE_ONE);

    /* make cusparse handle */
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
    cusparseDestroy(handle);
    cudaFree(rows);
    cudaFree(cols);
    cudaFree(vals);
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
device_csr_matrix_view::spgemm(csr_matrix1* h_leftMatrix) {
    device_csr_matrix_view* A = this;
    device_csr_matrix_view* B = new device_csr_matrix_view(h_leftMatrix);
    int mm = A->m,
        nn = A->n,
        kk = B->n;
    assert(A->n == B->m);

    cusparseOperation_t transA = CUSPARSE_OPERATION_NON_TRANSPOSE,
                        transB = CUSPARSE_OPERATION_NON_TRANSPOSE;

    device_csr_matrix_view* C = new device_csr_matrix_view();

    int baseC;
    cudaMalloc(&C->rows, sizeof(int)*(mm+1));
    cusparseXcsrgemmNnz(handle, transA, transB,
            mm, nn, kk,
            A->desc, A->nnz, A->rows, A->cols,
            B->desc, B->nnz, B->rows, B->cols,
            C->desc, C->rows, C->cols);
    cudaMemcpy(&C->nnz, C->rows+mm, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&baseC, C->rows, sizeof(int), cudaMemcpyDeviceToHost);
    C->nnz -= baseC;
    cudaMalloc(&C->cols, sizeof(int)*C->nnz);
    cudaMalloc(&C->vals, sizeof(float)*C->nnz);
    cusparseScsrgemm(handle, transA, transB,
            mm, nn, kk,
            A->desc, A->nnz, A->vals, A->rows, A->cols,
            B->desc, B->nnz, B->vals, B->rows, B->cols,
            C->desc, C->vals, C->rows, C->cols);

    //delete A; // or by caller
    delete B;
    return C;
}

device_csr_matrix_view::device_csr_matrix_view() :
    m(0), n(0), nnz(0), rows(NULL), cols(NULL), vals(NULL), desc(NULL), handle(NULL) { }

void
OsdCusparseKernelDispatcher::BindVertexBuffer(OsdVertexBuffer *vertex, OsdVertexBuffer *varying)
{
    _currentVertexBuffer = (vertex) ?
        dynamic_cast<OsdCusparseVertexBuffer *>(vertex) : NULL;

    _currentVaryingBuffer = (varying) ?
        dynamic_cast<OsdCusparseVertexBuffer *>(varying) : NULL;

    _vdesc = new SpMVVertexDescriptor(this,
            _currentVertexBuffer  ? _currentVertexBuffer->GetNumElements()  : 0,
            _currentVaryingBuffer ? _currentVaryingBuffer->GetNumElements() : 0);
}

OsdVertexBuffer *
OsdCusparseKernelDispatcher::InitializeVertexBuffer(int numElements, int numVertices)
{
    return new OsdCusparseVertexBuffer(numElements, numVertices);
}

OsdCusparseKernelDispatcher::OsdCusparseKernelDispatcher( int levels )
    : OsdMklKernelDispatcher(levels), _deviceMatrix(NULL) { }

OsdCusparseKernelDispatcher::~OsdCusparseKernelDispatcher()
{
    delete _deviceMatrix;
}

static OsdCusparseKernelDispatcher::OsdKernelDispatcher *
Create(int levels) {
    return new OsdCusparseKernelDispatcher(levels);
}

void
OsdCusparseKernelDispatcher::Register() {
    Factory::GetInstance().Register(Create, kCUSPARSE);
}

void
OsdCusparseKernelDispatcher::FinalizeMatrix()
{
    /* use mkl to build M_big */
    this->OsdMklKernelDispatcher::FinalizeMatrix();
    _deviceMatrix = new device_csr_matrix_view(M_big);
}

void
OsdCusparseKernelDispatcher::ApplyMatrix(int offset)
{
    float* V_in = (float*) _currentVertexBuffer->Map();
    float* V_out = V_in + offset * _currentVertexBuffer->GetNumElements();

    _deviceMatrix->spmv(V_out, V_in);

    _currentVertexBuffer->Unmap();
}

} // end namespace OPENSUBDIV_VERSION

} // end namespace OpenSubdiv

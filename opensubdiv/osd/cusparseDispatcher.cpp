#include "../version.h"
#include "../osd/mutex.h"
#include "../osd/cusparseDispatcher.h"
#include "../osd/spmvKernel.h"

#include <stdio.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

static cusparseHandle_t handle = NULL;

device_csr_matrix_view::device_csr_matrix_view(csr_matrix1* M) :
    m(M->size1()), n(M->size2()), nnz(M->nnz()) {

    /* make cusparse matrix descriptor */
    cusparseCreateMatDescr(&desc);
    cusparseSetMatType(desc,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(desc,CUSPARSE_INDEX_BASE_ONE);

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
    printf("%s: %d-%d, %d nnz, r/c/v: 0%x 0%x 0%x\n",
            name.c_str(), m, n, nnz, (void*) rows, (void*) cols, (void*) vals);
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

device_csr_matrix_view::device_csr_matrix_view(int m, int n) :
    m(m), n(n), nnz(0), rows(NULL), cols(NULL), vals(NULL)
{
    /* make cusparse matrix descriptor */
    cusparseCreateMatDescr(&desc);
    cusparseSetMatType(desc,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(desc,CUSPARSE_INDEX_BASE_ONE);

    /* make cusparse handle if null */
    if (handle == NULL)
        cusparseCreate(&handle);
}

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
OsdCusparseKernelDispatcher::PushMatrix()
{
    if (_deviceMatrix == NULL) {
        printf("PushMatrix set %d-%d\n", S->size1(), S->size2());
        csr_matrix1 S_csr(*S);
        _deviceMatrix = new device_csr_matrix_view(&S_csr);
    } else {
        printf("PushMatrix mul %d-%d = %d-%d * %d-%d\n",
                (int) S->size1(), (int) _deviceMatrix->n,
                (int) S->size1(),  (int) S->size2(),
                _deviceMatrix->m, _deviceMatrix->n);
        csr_matrix1 S_csr(*S);
        device_csr_matrix_view A (&S_csr);
        device_csr_matrix_view *C = A.times(_deviceMatrix);
        delete _deviceMatrix;
        _deviceMatrix = C;
    }
}

bool
OsdCusparseKernelDispatcher::MatrixReady()
{
    return (_deviceMatrix != NULL);
}

void
OsdCusparseKernelDispatcher::FinalizeMatrix()
{
}

void
OsdCusparseKernelDispatcher::ApplyMatrix(int offset)
{
    float* d_in = (float*) _currentVertexBuffer->Map();
    float* d_out = d_in + offset * _currentVertexBuffer->GetNumElements();
    _deviceMatrix->spmv(d_out, d_in);
    _currentVertexBuffer->Unmap();
}

} // end namespace OPENSUBDIV_VERSION

} // end namespace OpenSubdiv

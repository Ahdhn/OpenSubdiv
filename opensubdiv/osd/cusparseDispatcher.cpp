#include "../version.h"
#include "../osd/mutex.h"
#include "../osd/cusparseDispatcher.h"
#include "../osd/spmvKernel.h"

#include <stdio.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

extern "C" {
void OsdCusparseExpand(int src_m, int factor, int* dst_rows, int* dst_cols, float* dst_vals, int* src_rows, int* src_cols, float* src_vals);
}

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

static cusparseHandle_t handle = NULL;

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

    /* allocate space if nnz != 0 */
    if (nnz > 0) {
        cudaMalloc(&rows, (m+1) * sizeof(int));
        cudaMalloc(&cols, nnz * sizeof(int));
        cudaMalloc(&vals, nnz * sizeof(float));
    }
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
    : OsdSpMVKernelDispatcher(levels),
      _deviceMatrix(NULL), _deviceMatrixBig(NULL), S(NULL) { }

OsdCusparseKernelDispatcher::~OsdCusparseKernelDispatcher()
{
    if (S)                  delete S;
    if (_deviceMatrix)      delete _deviceMatrix;
    if (_deviceMatrixBig)   delete _deviceMatrixBig;
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
OsdCusparseKernelDispatcher::StageMatrix(int i, int j)
{
    if (S) delete S;
    S = new coo_matrix(i,j);
}

inline void
OsdCusparseKernelDispatcher::StageElem(int i, int j, float value)
{
#ifdef DEBUG
    assert(0 <= i);
    assert(i < S->size1());
    assert(0 <= j);
    assert(j < S->size2());
#endif
    S->append_element(i, j, value);
}

void
OsdCusparseKernelDispatcher::PushMatrix()
{
    if (_deviceMatrix == NULL) {
        printf("PushMatrix set %d-%d\n", S->size1(), S->size2());
        csr_matrix S_csr(*S);
        _deviceMatrix = new device_csr_matrix_view(&S_csr);
    } else {
        printf("PushMatrix mul %d-%d = %d-%d * %d-%d\n",
                (int) S->size1(), (int) _deviceMatrix->n,
                (int) S->size1(),  (int) S->size2(),
                _deviceMatrix->m, _deviceMatrix->n);
        csr_matrix S_csr(*S);
        device_csr_matrix_view A (&S_csr);
        device_csr_matrix_view *C = A.times(_deviceMatrix);
        delete _deviceMatrix;
        _deviceMatrix = C;
    }
}

bool
OsdCusparseKernelDispatcher::MatrixReady()
{
    return (_deviceMatrixBig != NULL);
}

template<typename T> void
printmat(T* d_mat, std::string name, size_t len) {
    T *h_mat= new T[len];
    cudaMemcpy(h_mat, d_mat, len * sizeof(T), cudaMemcpyDeviceToHost);
    std::cout << name << ": ";
    for(int i = 0; i < len; i++) {
        T elem = h_mat[i];
        if (elem != elem || elem < 0 || elem > 100) {
            std::cout << elem << " X ...";
            break;
        }
        std::cout << elem << " ";
    }
    std::cout << std::endl;
    delete[] h_mat;
}

void
OsdCusparseKernelDispatcher::FinalizeMatrix()
{
    int nve = _currentVertexBuffer->GetNumElements();
    device_csr_matrix_view* M_src = _deviceMatrix;
    device_csr_matrix_view* M_dst =
        new device_csr_matrix_view(nve*M_src->m, nve*M_src->n, nve*M_src->nnz);

    int *coo_dst_rows;
    cusparseStatus_t status;
    cudaMalloc(&coo_dst_rows, M_dst->nnz * sizeof(int));
    {
        /* expand by a factor of nve */
        OsdCusparseExpand(M_src->m, nve,
                coo_dst_rows, M_dst->cols, M_dst->vals,
                M_src->rows, M_src->cols, M_src->vals);

        /* convert to csr format */
        status = cusparseXcoo2csr(handle, coo_dst_rows, M_dst->nnz, M_dst->m,
                M_dst->rows, CUSPARSE_INDEX_BASE_ZERO);
        assert(status == CUSPARSE_STATUS_SUCCESS);
    }
    cudaFree(coo_dst_rows);
    _deviceMatrixBig = M_dst;

    PrintReport();
}

void
OsdCusparseKernelDispatcher::ApplyMatrix(int offset)
{
    float* d_in = (float*) _currentVertexBuffer->Map();
    float* d_out = d_in + offset * _currentVertexBuffer->GetNumElements();
    _deviceMatrixBig->spmv(d_out, d_in);
    _currentVertexBuffer->Unmap();
}

void
OsdCusparseKernelDispatcher::PrintReport()
{
    int size_in_bytes = (_deviceMatrixBig->nnz * 2 +
            _deviceMatrixBig->m + 1) * sizeof(float);
    printf("Subdiv matrix is %d-by-%d with %f%% nonzeroes, takes %d MB.\n",
        _deviceMatrixBig->m, _deviceMatrixBig->n,
        100.0 * _deviceMatrixBig->nnz / _deviceMatrixBig->m / _deviceMatrixBig->n,
        size_in_bytes / 1024 / 1024);
}

} // end namespace OPENSUBDIV_VERSION

} // end namespace OpenSubdiv

#include "../version.h"
#include "../osd/mutex.h"
#include "../osd/cusparseDispatcher.h"
#include "../osd/spmvKernel.h"

#include <stdio.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

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
    : OsdMklKernelDispatcher(levels)
{
    /* make cusparse matrix descriptor */
    cusparseCreateMatDescr(&desc);
    cusparseSetMatType(desc,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(desc,CUSPARSE_INDEX_BASE_ONE);

    /* make cusparse handle */
    cusparseCreate(&handle);
}

OsdCusparseKernelDispatcher::~OsdCusparseKernelDispatcher()
{
    /* clean up device memory */
    cusparseDestroy(handle);
    cudaFree(d_rows);
    cudaFree(d_cols);
    cudaFree(d_vals);
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

    /* allocate device memory */
    cudaMalloc(&d_rows, M_big->index1_data().size()*sizeof(int));
    cudaMalloc(&d_cols, M_big->index2_data().size()*sizeof(int));
    cudaMalloc(&d_vals, M_big->value_data().size()*sizeof(float));

    /* copy data to device */
    cudaMemcpy(d_rows, &M_big->index1_data()[0], M_big->index1_data().size()*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cols, &M_big->index2_data()[0], M_big->index2_data().size()*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vals, &M_big->value_data()[0], M_big->value_data().size()*sizeof(float), cudaMemcpyHostToDevice);
}

void
OsdCusparseKernelDispatcher::ApplyMatrix(int offset)
{
    int numElems = _currentVertexBuffer->GetNumElements();
    float* V_in = (float*) _currentVertexBuffer->Map();
    float* V_out = V_in + offset * numElems;

    /* do spmv */
    cusparseStatus_t status;
    cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
    int m = M_big->size1();
    int n = M_big->size2();
    int nnz = M_big->nnz();
    float alpha = 1.0,
          beta = 0.0;
    status = cusparseScsrmv(handle, op, m, n, nnz, &alpha, desc, d_vals, d_rows, d_cols, V_in, &beta, V_out);
    assert(status == CUSPARSE_STATUS_SUCCESS);
}

} // end namespace OPENSUBDIV_VERSION

} // end namespace OpenSubdiv

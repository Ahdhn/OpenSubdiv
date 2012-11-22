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
    if (vertex)
        _currentVertexBuffer = dynamic_cast<OsdCpuVertexBuffer *>(vertex);
    else
        _currentVertexBuffer = NULL;

    if (varying)
        _currentVaryingBuffer = dynamic_cast<OsdCpuVertexBuffer *>(varying);
    else
        _currentVaryingBuffer = NULL;

    _vdesc = new SpMVVertexDescriptor(this,
            _currentVertexBuffer  ? _currentVertexBuffer->GetNumElements()  : 0,
            _currentVaryingBuffer ? _currentVaryingBuffer->GetNumElements() : 0);
}

void
OsdCusparseKernelDispatcher::UnbindVertexBuffer()
{
    delete _vdesc;
    _vdesc = NULL;

    _currentVertexBuffer = NULL;
    _currentVaryingBuffer = NULL;
}


OsdCusparseKernelDispatcher::OsdCusparseKernelDispatcher( int levels )
    : OsdMklKernelDispatcher(levels)
{
    /* make cusparse matrix descriptor */
    cusparseCreateMatDescr(&desc);
    cusparseSetMatType(desc,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(desc,CUSPARSE_INDEX_BASE_ONE);

    /* make cusparse handle */
    cusparseStatus_t status = cusparseCreate(&handle);
    assert(status == CUSPARSE_STATUS_SUCCESS);
}

OsdCusparseKernelDispatcher::~OsdCusparseKernelDispatcher()
{
    /* clean up device memory */
    cusparseDestroy(handle);
    cudaFree(d_in);
    cudaFree(d_out);
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
    cudaMalloc(&d_in, M_big->size2()*sizeof(float));
    cudaMalloc(&d_out, M_big->size1()*sizeof(float));
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
    int n_in = M_big->size2();
    int n_out = M_big->size1();
    int numElems = _currentVertexBuffer->GetNumElements();
    float* V_in = _currentVertexBuffer->GetCpuBuffer();
    float* V_out = _currentVertexBuffer->GetCpuBuffer()
                   + offset * numElems;

    /* copy coarse vertices to device */
    cudaMemcpy(d_in, V_in, n_in*sizeof(float), cudaMemcpyHostToDevice);

    /* do spmv */
    cusparseStatus_t status;
    cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
    int m = M_big->size1();
    int n = M_big->size2();
    int nnz = M_big->nnz();
    float alpha = 1.0,
          beta = 0.0;
    status = cusparseScsrmv(handle, op, m, n, nnz, &alpha, desc, d_vals, d_rows, d_cols, d_in, &beta, d_out);
    assert(status == CUSPARSE_STATUS_SUCCESS);

    /* copy refined vertices back */
    cudaMemcpy(V_out, d_out, n_out*sizeof(float), cudaMemcpyDeviceToHost);
}

} // end namespace OPENSUBDIV_VERSION

} // end namespace OpenSubdiv

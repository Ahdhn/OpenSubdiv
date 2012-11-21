#include "../version.h"
#include "../osd/mutex.h"
#include "../osd/cusparseDispatcher.h"

#include <stdio.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cusparse_v2.h>

using namespace std;
using namespace boost::numeric::ublas;

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

OsdCusparseKernelDispatcher::OsdCusparseKernelDispatcher( int levels )
    : OsdMklKernelDispatcher(levels)
{
}

OsdCusparseKernelDispatcher::~OsdCusparseKernelDispatcher()
{
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
    this->OsdMklKernelDispatcher::FinalizeMatrix();
}

void
OsdCusparseKernelDispatcher::ApplyMatrix(int offset)
{
    int *d_rows, *d_cols;
    float *d_in, *d_out, *d_vals;

    int n_in = M_big->size2();
    int n_out = M_big->size1();
    int numElems = _currentVertexBuffer->GetNumElements();
    float* V_in = _currentVertexBuffer->GetCpuBuffer();
    float* V_out = _currentVertexBuffer->GetCpuBuffer()
                   + offset * numElems;

    /* allocate device memory */
    cudaMalloc(&d_in, n_in*sizeof(float));
    cudaMalloc(&d_out, n_out*sizeof(float));
    cudaMalloc(&d_rows, M_big->index1_data().size()*sizeof(int));
    cudaMalloc(&d_cols, M_big->index2_data().size()*sizeof(int));
    cudaMalloc(&d_vals, M_big->value_data().size()*sizeof(float));

    /* copy data to device */
    cudaMemcpy(d_in, V_in, n_in*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rows, &M_big->index1_data()[0], M_big->index1_data().size()*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cols, &M_big->index2_data()[0], M_big->index2_data().size()*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vals, &M_big->value_data()[0], M_big->value_data().size()*sizeof(float), cudaMemcpyHostToDevice);

    /* make cusparse matrix descriptor */
    cusparseMatDescr_t desc = 0;
    cusparseCreateMatDescr(&desc);
    cusparseSetMatType(desc,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(desc,CUSPARSE_INDEX_BASE_ONE);

    /* make cusparse handle */
    cusparseHandle_t handle;
    cusparseStatus_t status = cusparseCreate(&handle);
    assert(status == CUSPARSE_STATUS_SUCCESS);

    /* do spmv */
    cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
    int m = M_big->size1();
    int n = M_big->size2();
    int nnz = M_big->nnz();
    float alpha = 1.0,
          beta = 0.0;
    status = cusparseScsrmv(handle, op, m, n, nnz, &alpha, desc, d_vals, d_rows, d_cols, d_in, &beta, d_out);
    assert(status == CUSPARSE_STATUS_SUCCESS);

    /* copy vertices back */
    cudaMemcpy(V_out, d_out, n_out*sizeof(float), cudaMemcpyDeviceToHost);

    /* clean up */
    cusparseDestroy(handle);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_rows);
    cudaFree(d_cols);
    cudaFree(d_vals);
}

} // end namespace OPENSUBDIV_VERSION

} // end namespace OpenSubdiv

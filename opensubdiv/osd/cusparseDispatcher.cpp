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
}

void
OsdCusparseKernelDispatcher::ApplyMatrix(int offset)
{
    int *d_rows, *d_cols;
    float *d_in, *d_out, *d_vals;

    int n_in = M_big->size1();
    int n_out = M_big->size2();
    int numElems = _currentVertexBuffer->GetNumElements();
    float* V_in = _currentVertexBuffer->GetCpuBuffer();
    float* V_out = _currentVertexBuffer->GetCpuBuffer()
                   + offset * numElems;

    /* allocate device memory for vertices */
    cudaMalloc((void**) &d_in, n_in*sizeof(float));
    cudaMalloc((void**) &d_out, n_out*sizeof(float));

    /* allocate device memory for M */
    cudaMalloc((void**) &d_rows, M_big->size1()*sizeof(int));
    cudaMalloc((void**) &d_cols, M_big->nnz()*sizeof(int));
    cudaMalloc((void**) &d_vals, M_big->nnz()*sizeof(float));

    /* cusparse matrix descriptor */
    cusparseMatDescr_t desc = 0;
    cusparseCreateMatDescr(&desc);
    cusparseSetMatType(desc,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(desc,CUSPARSE_INDEX_BASE_ZERO);

    cusparseHandle_t handle = 0;
    cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
    int m = M_big->size1();
    int n = M_big->size2();
    int nnz = M_big->nnz();
    const float alpha = 1.0;
    const cusparseMatDescr_t descrA = desc;
    const float *csrValA = d_vals;
    const int *csrRowPtrA = d_rows;
    const int *csrColIndA = d_cols;
    const float *x = d_in;
    const float beta = 0.0;
    float *y = d_out;

    cusparseScsrmv(handle, op, m, n, nnz, &alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, &beta, y);
}

bool
OsdCusparseKernelDispatcher::MatrixReady()
{
    return true;
}

} // end namespace OPENSUBDIV_VERSION

} // end namespace OpenSubdiv

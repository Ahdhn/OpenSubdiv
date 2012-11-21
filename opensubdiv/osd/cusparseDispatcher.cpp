#include "../version.h"
#include "../osd/mutex.h"
#include "../osd/cusparseDispatcher.h"

#include <stdio.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

using namespace std;
using namespace boost::numeric::ublas;

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

OsdCusparseKernelDispatcher::OsdCusparseKernelDispatcher( int levels )
    : OsdSpMVKernelDispatcher(levels), S(NULL), _currentVertexBuffer(NULL)
{
}

OsdCusparseKernelDispatcher::~OsdCusparseKernelDispatcher()
{
    if (S) delete S;
}

static OsdCusparseKernelDispatcher::OsdKernelDispatcher *
Create(int levels) {
    return new OsdCusparseKernelDispatcher(levels);
}

void
OsdCusparseKernelDispatcher::Register() {
    Factory::GetInstance().Register(Create, kCUSPARSE);
}

OsdVertexBuffer*
OsdCusparseKernelDispatcher::InitializeVertexBuffer(int numElements, int numVertices) {
    return new OsdCudaVertexBuffer(numElements, numVertices);
}

void
OsdCusparseKernelDispatcher::BindVertexBuffer(OsdVertexBuffer *vertex, OsdVertexBuffer *varying) {

    if (vertex) {
        _currentVertexBuffer = dynamic_cast<OsdCudaVertexBuffer *>(vertex);
        _numVertexElements = _currentVertexBuffer->GetNumElements();
    } else {
        _currentVertexBuffer = NULL;
    }

    if (_currentVertexBuffer)
        _deviceVertices = (float*)_currentVertexBuffer->Map();
}

void
OsdCusparseKernelDispatcher::UnbindVertexBuffer()
{
    if (_currentVertexBuffer){
        _currentVertexBuffer->Unmap();
    }
}


void
OsdCusparseKernelDispatcher::StageMatrix(int i, int j)
{
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
    (*S)(i,j) = value;
}

void
OsdCusparseKernelDispatcher::PushMatrix()
{
    // csr_matrix A(*S);

    // cusparseScsrgemm

    delete S;
}

void
OsdCusparseKernelDispatcher::FinalizeMatrix()
{
}

void
OsdCusparseKernelDispatcher::ApplyMatrix(int offset)
{
    float* V_in = _deviceVertices;
    float* V_out = _deviceVertices + offset * _numVertexElements;

    // cusparseScsrmv
}

void
OsdCusparseKernelDispatcher::WriteMatrix()
{
}

bool
OsdCusparseKernelDispatcher::MatrixReady()
{
    return true;
}

void
OsdCusparseKernelDispatcher::PrintReport()
{
}

// -------------------------------------------------------------------------------
OsdCusparseKernelDispatcher::DeviceTable::~DeviceTable() {

    if (devicePtr) cudaFree(devicePtr);
}

void
OsdCusparseKernelDispatcher::DeviceTable::Copy(int size, const void *ptr) {

    if (devicePtr)
        cudaFree(devicePtr);
    cudaMalloc(&devicePtr, size);
    cudaMemcpy(devicePtr, ptr, size, cudaMemcpyHostToDevice);
}

} // end namespace OPENSUBDIV_VERSION

} // end namespace OpenSubdiv

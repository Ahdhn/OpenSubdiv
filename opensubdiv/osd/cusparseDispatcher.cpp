#include "../version.h"
#include "../osd/mutex.h"
#include "../osd/cusparseDispatcher.h"

#include <stdio.h>

using namespace std;
using namespace boost::numeric::ublas;

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

OsdCusparseKernelDispatcher::OsdCusparseKernelDispatcher( int levels )
    : OsdSpMVKernelDispatcher(levels), S(NULL)
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
    csr_matrix A(*S);

    // cusparseScsrgemm

    delete S;
}

void
OsdCusparseKernelDispatcher::ApplyMatrix(int offset)
{
    int numElems = _currentVertexBuffer->GetNumElements();
    float* V_in = _currentVertexBuffer->GetCpuBuffer();
    float* V_out = _currentVertexBuffer->GetCpuBuffer()
                   + offset * numElems;

    // cusparseScsrmv
}

void
OsdCusparseKernelDispatcher::WriteMatrix()
{
}

bool
OsdCusparseKernelDispatcher::MatrixReady()
{
}

void
OsdCusparseKernelDispatcher::PrintReport()
{
}

} // end namespace OPENSUBDIV_VERSION

} // end namespace OpenSubdiv

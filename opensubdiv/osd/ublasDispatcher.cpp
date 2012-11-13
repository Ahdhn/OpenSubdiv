#include "../version.h"
#include "../osd/mutex.h"
#include "../osd/ublasDispatcher.h"

#include <stdio.h>

using namespace std;
using namespace boost::numeric::ublas;

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

OsdUBlasKernelDispatcher::OsdUBlasKernelDispatcher( int levels )
    : OsdSpMVKernelDispatcher(levels), S(NULL)
{
}

OsdUBlasKernelDispatcher::~OsdUBlasKernelDispatcher()
{
    if (S) delete S;
}

static OsdUBlasKernelDispatcher::OsdKernelDispatcher *
Create(int levels) {
    return new OsdUBlasKernelDispatcher(levels);
}

void
OsdUBlasKernelDispatcher::Register() {
    Factory::GetInstance().Register(Create, kUBLAS);
}

void
OsdUBlasKernelDispatcher::StageMatrix(int i, int j)
{
    S = new coo_matrix(i,j);
}

inline void
OsdUBlasKernelDispatcher::StageElem(int i, int j, float value)
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
OsdUBlasKernelDispatcher::PushMatrix()
{
    csr_matrix A(*S);
    delete S;
}

void
OsdUBlasKernelDispatcher::ApplyMatrix(int offset)
{
    int numElems = _currentVertexBuffer->GetNumElements();
    float* V_in = _currentVertexBuffer->GetCpuBuffer();
    float* V_out = _currentVertexBuffer->GetCpuBuffer()
                   + offset * numElems;
}

void
OsdUBlasKernelDispatcher::WriteMatrix()
{
}

bool
OsdUBlasKernelDispatcher::MatrixReady()
{
}

void
OsdUBlasKernelDispatcher::PrintReport()
{
}

} // end namespace OPENSUBDIV_VERSION

} // end namespace OpenSubdiv

#include "../version.h"
#include "../osd/mutex.h"
#include "../osd/clspmvDispatcher.h"

using namespace std;

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

OsdClSpMVKernelDispatcher::OsdClSpMVKernelDispatcher( int levels )
    : OsdSpMVKernelDispatcher(levels), S(NULL)
{
    // add code here optionally
}

OsdClSpMVKernelDispatcher::~OsdClSpMVKernelDispatcher()
{
    // add code here optionally
}

static OsdClSpMVKernelDispatcher::OsdKernelDispatcher *
Create(int levels) {
    return new OsdClSpMVKernelDispatcher(levels);
}

void
OsdClSpMVKernelDispatcher::Register() {
    Factory::GetInstance().Register(Create, kCLSPMV);
}

void
OsdClSpMVKernelDispatcher::StageMatrix(int i, int j)
{
    if (S != NULL) delete S;
    int numElems = _currentVertexBuffer->GetNumElements();
    S = new coo_matrix(i*numElems,j*numElems);
}

inline void
OsdClSpMVKernelDispatcher::StageElem(int i, int j, float value)
{
#ifdef DEBUG
    assert(0 <= i);
    assert(i < S->size1());
    assert(0 <= j);
    assert(j < S->size2());
#endif
    int numElems = _currentVertexBuffer->GetNumElements();
    for(int k = 0; k < numElems; k++)
        (*S)(i*numElems+k,j*numElems+k) = value;
}

void
OsdClSpMVKernelDispatcher::PushMatrix()
{
    csr_matrix A(*S);
}

void
OsdClSpMVKernelDispatcher::FinalizeMatrix()
{
    this->PrintReport();
    // add code here
}

void
OsdClSpMVKernelDispatcher::ApplyMatrix(int offset)
{
    /*
    int numElems = _currentVertexBuffer->GetNumElements();
    float* V_in = _currentVertexBuffer->GetCpuBuffer();
    float* V_out = _currentVertexBuffer->GetCpuBuffer()
                   + offset * numElems;
    */

    // add code here
}

bool
OsdClSpMVKernelDispatcher::MatrixReady()
{
    // add code here
    return true;
}

} // end namespace OPENSUBDIV_VERSION

} // end namespace OpenSubdiv

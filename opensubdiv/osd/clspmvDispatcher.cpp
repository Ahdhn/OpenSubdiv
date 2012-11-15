#include "../version.h"
#include "../osd/mutex.h"
#include "../osd/clspmvDispatcher.h"

using namespace std;

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

OsdClSpMVKernelDispatcher::OsdClSpMVKernelDispatcher( int levels )
    : OsdSpMVKernelDispatcher(levels)
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
    // add code here
}

inline void
OsdClSpMVKernelDispatcher::StageElem(int i, int j, float value)
{
    // add code here
}

void
OsdClSpMVKernelDispatcher::PushMatrix()
{
    // add code here
}

void
OsdClSpMVKernelDispatcher::ApplyMatrix(int offset)
{
    int numElems = _currentVertexBuffer->GetNumElements();
    float* V_in = _currentVertexBuffer->GetCpuBuffer();
    float* V_out = _currentVertexBuffer->GetCpuBuffer()
                   + offset * numElems;

    // add code here
}

void
OsdClSpMVKernelDispatcher::WriteMatrix()
{
    // add code here optionally
}

bool
OsdClSpMVKernelDispatcher::MatrixReady()
{
    // add code here
    return true;
}

void
OsdClSpMVKernelDispatcher::PrintReport()
{
    // add code here optionally
}

} // end namespace OPENSUBDIV_VERSION

} // end namespace OpenSubdiv

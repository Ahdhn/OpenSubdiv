#include "../version.h"
#include "../osd/mutex.h"
#include "../osd/clspmvDispatcher.h"

using namespace std;

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

OsdClSpMVKernelDispatcher::OsdClSpMVKernelDispatcher( int levels )
    : OsdSpMVKernelDispatcher(levels)
{
}

OsdClSpMVKernelDispatcher::~OsdClSpMVKernelDispatcher()
{
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
}

inline void
OsdClSpMVKernelDispatcher::StageElem(int i, int j, float value)
{
}

void
OsdClSpMVKernelDispatcher::PushMatrix()
{
}

void
OsdClSpMVKernelDispatcher::ApplyMatrix(int offset)
{
    int numElems = _currentVertexBuffer->GetNumElements();
    float* V_in = _currentVertexBuffer->GetCpuBuffer();
    float* V_out = _currentVertexBuffer->GetCpuBuffer()
                   + offset * numElems;
}

void
OsdClSpMVKernelDispatcher::WriteMatrix()
{
}

bool
OsdClSpMVKernelDispatcher::MatrixReady()
{
}

void
OsdClSpMVKernelDispatcher::PrintReport()
{
}

} // end namespace OPENSUBDIV_VERSION

} // end namespace OpenSubdiv

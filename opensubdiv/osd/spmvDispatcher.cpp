#include "../version.h"
#include "../osd/mutex.h"
#include "../osd/spmvDispatcher.h"
#include "../osd/spmvKernel.h"

#include <stdio.h>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

OsdSpMVKernelDispatcher::OsdSpMVKernelDispatcher( int levels )
    : OsdCpuKernelDispatcher(levels), matrix_id(0)
{
#if BENCHMARKING
    printf("\n");
#endif
}

OsdSpMVKernelDispatcher::~OsdSpMVKernelDispatcher() {
    if (_vdesc)
        delete _vdesc;
}

void
OsdSpMVKernelDispatcher::BindVertexBuffer(OsdVertexBuffer *vertex, OsdVertexBuffer *varying) {

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

int
OsdSpMVKernelDispatcher::CopyNVerts(int nVerts, int dstIndex, int srcIndex) {
    for (int i = 0; i < nVerts; i++)
        _vdesc->AddWithWeight(NULL, dstIndex+i, srcIndex+i, 1.0);
    return nVerts;
}

} // end namespace OPENSUBDIV_VERSION

} // end namespace OpenSubdiv

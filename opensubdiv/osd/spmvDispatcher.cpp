#include "../version.h"
#include "../osd/mutex.h"
#include "../osd/spmvDispatcher.h"
#include "../osd/spmvKernel.h"

#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <sys/time.h>

using namespace std;
using namespace boost::numeric::ublas;

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

OsdSpMVKernelDispatcher::OsdSpMVKernelDispatcher( int levels )
    : OsdCpuKernelDispatcher(levels) { }

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

static OsdSpMVKernelDispatcher::OsdKernelDispatcher *
Create(int levels) {
    return new OsdSpMVKernelDispatcher(levels);
}

void
OsdSpMVKernelDispatcher::Register() {
    assert(false && "SpMV Kernel Dispatchers shouldn't be instantiated. Use subclass instead.");
}

int
OsdSpMVKernelDispatcher::CopyNVerts(int nVerts, int dstIndex, int srcIndex) {
    for (int i = 0; i < nVerts; i++)
        _vdesc->AddWithWeight(NULL, dstIndex+i, srcIndex+i, 1.0);
    return nVerts;
}

void
OsdSpMVKernelDispatcher::StageMatrix(int i, int j) {
    assert(false && "OsdSpMVKernelDispatcher::StageMatrix must be overridden.");
}

inline void
OsdSpMVKernelDispatcher::StageElem(int i, int j, float value) {
    assert(false && "OsdSpMVKernelDispatcher::StageElem must be overridden.");
}

void
OsdSpMVKernelDispatcher::PushMatrix() {
    assert(false && "OsdSpMVKernelDispatcher::PushMatrix must be overridden.");
}

void
OsdSpMVKernelDispatcher::ApplyM(int offset) {
    assert(false && "OsdSpMVKernelDispatcher::ApplyM must be overridden.");
}

void
OsdSpMVKernelDispatcher::WriteM() {
    assert(false && "OsdSpMVKernelDispatcher::WriteM must be overridden.");
}

} // end namespace OPENSUBDIV_VERSION

} // end namespace OpenSubdiv

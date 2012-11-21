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
    //float* V_in = _deviceVertices;
    //float* V_out = _deviceVertices + offset * _numVertexElements;

    // cusparseScsrmv
}

bool
OsdCusparseKernelDispatcher::MatrixReady()
{
    return true;
}

} // end namespace OPENSUBDIV_VERSION

} // end namespace OpenSubdiv

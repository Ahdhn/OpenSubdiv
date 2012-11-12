#include "../version.h"
#include "../osd/mutex.h"
#include "../osd/mklDispatcher.h"

#include <stdio.h>

using namespace std;
using namespace boost::numeric::ublas;

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

extern void mkl_scsrmultcsr(char *trans, int *request, int *sort, int *m, int *n, int *k,
        float *a, int *ja, int *ia, float *b, int *jb, int *ib, float *c, 
        int *jc, int *ic, int *nzmax, int *info);

OsdMklKernelDispatcher::OsdMklKernelDispatcher( int levels )
    : OsdSpMVKernelDispatcher(levels), S(NULL)
{
}

OsdMklKernelDispatcher::~OsdMklKernelDispatcher()
{
    if (S) delete S;
}

static OsdMklKernelDispatcher::OsdKernelDispatcher *
Create(int levels) {
    return new OsdMklKernelDispatcher(levels);
}

void
OsdMklKernelDispatcher::Register() {
    Factory::GetInstance().Register(Create, kMKL);
}

void
OsdMklKernelDispatcher::StageMatrix(int i, int j)
{
    S = new coo_matrix(i,j);
}

inline void
OsdMklKernelDispatcher::StageElem(int i, int j, float value)
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
OsdMklKernelDispatcher::PushMatrix()
{

    delete S;
}

void
OsdMklKernelDispatcher::ApplyMatrix(int offset)
{
    int numElems = _currentVertexBuffer->GetNumElements();
    float* V_in = _currentVertexBuffer->GetCpuBuffer();
    float* V_out = _currentVertexBuffer->GetCpuBuffer()
                   + offset * numElems;
}

void
OsdMklKernelDispatcher::WriteMatrix()
{
}

bool
OsdMklKernelDispatcher::MatrixReady()
{
}

void
OsdMklKernelDispatcher::PrintReport()
{
}

} // end namespace OPENSUBDIV_VERSION

} // end namespace OpenSubdiv

#include "../version.h"
#include "../osd/mutex.h"
#include "../osd/mklDispatcher.h"

#include <stdio.h>

using namespace std;
using namespace boost::numeric::ublas;

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

OsdMklKernelDispatcher::OsdMklKernelDispatcher( int levels )
    : OsdSpMVKernelDispatcher(levels)
{
}

OsdMklKernelDispatcher::~OsdMklKernelDispatcher()
{
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
}

inline void
OsdMklKernelDispatcher::StageElem(int i, int j, float value)
{
}

void
OsdMklKernelDispatcher::PushMatrix()
{
}

void
OsdMklKernelDispatcher::ApplyMatrix(int offset)
{
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

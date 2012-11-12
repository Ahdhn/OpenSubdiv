#include "../version.h"
#include "../osd/mutex.h"
#include "../osd/mklDispatcher.h"

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
OsdMklKernelDispatcher::ApplyM(int offset)
{
}

void
OsdMklKernelDispatcher::WriteM()
{
}

bool
OsdMklKernelDispatcher::MReady()
{
}

void
OsdMklKernelDispatcher::PrintReport()
{
}

} // end namespace OPENSUBDIV_VERSION

} // end namespace OpenSubdiv

#ifndef OSD_CLSPMV_DISPATCHER_H
#define OSD_CLSPMV_DISPATCHER_H

#include "../version.h"
#include "../osd/spmvDispatcher.h"

#if defined(__APPLE__)
    #include <OpenCL/opencl.h>
#else
    #include <CL/opencl.h>
#endif

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

class OsdClSpMVKernelDispatcher : public OsdSpMVKernelDispatcher
{
public:
    OsdClSpMVKernelDispatcher(int levels);
    virtual ~OsdClSpMVKernelDispatcher();

    static void Register();

    virtual void StageMatrix(int i, int j);
    virtual void StageElem(int i, int j, float value);
    virtual void PushMatrix();
    virtual void ApplyMatrix(int offset);
    virtual void WriteMatrix();
    virtual bool MatrixReady();
    virtual void PrintReport();
};

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* OSD_CLSPMV_DISPATCHER_H */

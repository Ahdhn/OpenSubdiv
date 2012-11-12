#ifndef OSD_SPMV_DISPATCHER_H
#define OSD_SPMV_DISPATCHER_H

#include "../version.h"
#include "../osd/cpuDispatcher.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

struct SpMVVertexDescriptor;

class OsdSpMVKernelDispatcher : public OsdCpuKernelDispatcher
{
public:
    OsdSpMVKernelDispatcher(int levels);
    virtual ~OsdSpMVKernelDispatcher();

    virtual FarMesh<OsdVertex>::Strategy GetStrategy() {
        return FarMesh<OsdVertex>::SpMV;
    }

    static void Register();
    virtual void BindVertexBuffer(OsdVertexBuffer *vertex, OsdVertexBuffer *varying);

    virtual int CopyNVerts(int nVerts, int dstIndex, int srcIndex);
    virtual void StageMatrix(int i, int j);
    virtual void StageElem(int i, int j, float value);
    virtual void PushMatrix();
    virtual void ApplyM(int offset);
    virtual void WriteM();
    virtual bool MReady();
    virtual void PrintReport();

    virtual int GetElemsPerVertex() const { return _currentVertexBuffer ? _currentVertexBuffer->GetNumElements() : NULL; }
    virtual int GetElemsPerVarying() const { return _currentVaryingBuffer ? _currentVaryingBuffer->GetNumElements() : NULL; }
};

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* OSD_SPMV_DISPATCHER_H */

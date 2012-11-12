#ifndef OSD_SPMV_DISPATCHER_H
#define OSD_SPMV_DISPATCHER_H

#include "../version.h"
#include "../osd/cpuDispatcher.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

class OsdSpMVKernelDispatcher : public OsdCpuKernelDispatcher
{
public:
    OsdSpMVKernelDispatcher(int levels);
    virtual ~OsdSpMVKernelDispatcher();

    virtual FarMesh<OsdVertex>::Strategy GetStrategy() {
        return FarMesh<OsdVertex>::SpMV;
    }

    void BindVertexBuffer(OsdVertexBuffer *vertex, OsdVertexBuffer *varying);
    int GetElemsPerVertex() const { return _currentVertexBuffer ? _currentVertexBuffer->GetNumElements() : NULL; }
    int GetElemsPerVarying() const { return _currentVaryingBuffer ? _currentVaryingBuffer->GetNumElements() : NULL; }
    virtual int CopyNVerts(int nVerts, int dstIndex, int srcIndex);

    // static OsdSpMVKernelDispatcher* Create(int levels) = 0;
    // static void Register() = 0;

    virtual void StageMatrix(int i, int j) = 0;
    virtual void StageElem(int i, int j, float value) = 0;
    virtual void PushMatrix() = 0;
    virtual void ApplyMatrix(int offset) = 0;
    virtual void WriteMatrix() = 0;
    virtual bool MatrixReady() = 0;
    virtual void PrintReport() = 0;
};

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* OSD_SPMV_DISPATCHER_H */

#ifndef OSD_POSKI_DISPATCHER_H
#define OSD_POSKI_DISPATCHER_H

#include "../version.h"
#include "../osd/kernelDispatcher.h"

#include <oski/oski.h>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

struct PoskiVertexDescriptor;

class OsdPoskiKernelDispatcher : public OsdKernelDispatcher
{
public:
    OsdPoskiKernelDispatcher(int levels, int numOmpThreads=1);

    virtual ~OsdPoskiKernelDispatcher();


    virtual void ApplyBilinearFaceVerticesKernel(FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const;

    virtual void ApplyBilinearEdgeVerticesKernel(FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const;

    virtual void ApplyBilinearVertexVerticesKernel(FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const;


    virtual void ApplyCatmarkFaceVerticesKernel(FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const;

    virtual void ApplyCatmarkEdgeVerticesKernel(FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const;

    virtual void ApplyCatmarkVertexVerticesKernelB(FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const;

    virtual void ApplyCatmarkVertexVerticesKernelA(FarMesh<OsdVertex> * mesh, int offset, bool pass, int level, int start, int end, void * data) const;


    virtual void ApplyLoopEdgeVerticesKernel(FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const;

    virtual void ApplyLoopVertexVerticesKernelB(FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const;

    virtual void ApplyLoopVertexVerticesKernelA(FarMesh<OsdVertex> * mesh, int offset, bool pass, int level, int start, int end, void * data) const;

    virtual void ApplyVertexEdits(FarMesh<OsdVertex> *mesh, int offset, int level, void * clientdata) const;

    virtual void CopyTable(int tableIndex, size_t size, const void *ptr);

    virtual void AllocateEditTables(int n);

    virtual void UpdateEditTable(int tableIndex, const FarTable<unsigned int> &offsets, const FarTable<float> &values,
                                 int operation, int primVarOffset, int primVarWidth);

    virtual void OnKernelLaunch();

    virtual void OnKernelFinish() {}

    virtual OsdVertexBuffer *InitializeVertexBuffer(int numElements, int numVertices);

    virtual void BindVertexBuffer(OsdVertexBuffer *vertex, OsdVertexBuffer *varying);

    virtual void UnbindVertexBuffer();

    virtual void Synchronize();

    static void Register();

protected:

    // XXX: until far refactoring finishes, use this.
    struct Table {
        Table() : ptr(NULL) { }

       ~Table();

        void Copy(int size, const void *ptr);

        void *ptr;
    };

    float *GetVertexBuffer() const { return _currentVertexBuffer ? _currentVertexBuffer->GetCpuBuffer() : NULL; }

    float *GetVaryingBuffer() const { return _currentVaryingBuffer ? _currentVaryingBuffer->GetCpuBuffer() : NULL; }

    OsdCpuVertexBuffer *_currentVertexBuffer,
                       *_currentVaryingBuffer;

    PoskiVertexDescriptor *_vdesc;

    int _numOmpThreads;
    std::vector<Table> _tables;
    std::vector<Table> _editTables;
};

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* OSD_POSKI_DISPATCHER_H */

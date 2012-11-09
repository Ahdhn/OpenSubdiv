#ifndef OSD_OSKI_DISPATCHER_H
#define OSD_OSKI_DISPATCHER_H

#include "../version.h"
#include "../osd/kernelDispatcher.h"

extern "C" {
    #include <oski/oski_Tis.h>
}

#include <boost/numeric/ublas/operation.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

using namespace boost::numeric::ublas;

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

struct OskiVertexDescriptor;

class OsdOskiKernelDispatcher : public OsdKernelDispatcher
{
public:
    OsdOskiKernelDispatcher(int levels);

    virtual ~OsdOskiKernelDispatcher();


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

    virtual void OnKernelFinish();

    virtual OsdVertexBuffer *InitializeVertexBuffer(int numElements, int numVertices);

    virtual void BindVertexBuffer(OsdVertexBuffer *vertex, OsdVertexBuffer *varying);

    virtual void UnbindVertexBuffer();

    virtual void Synchronize();

    virtual FarMesh<OsdVertex>::Strategy GetStrategy() {
        return FarMesh<OsdVertex>::SpMV;
    }

    static void Register();

    virtual void StageMatrix(int i, int j);
    virtual void StageElem(int i, int j, float value);
    virtual void PushMatrix();
    virtual void ApplyM(int offset);

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

    virtual int GetElemsPerVertex() const { return _currentVertexBuffer ? _currentVertexBuffer->GetNumElements() : NULL; }

    virtual int GetElemsPerVarying() const { return _currentVaryingBuffer ? _currentVaryingBuffer->GetNumElements() : NULL; }



    OsdCpuVertexBuffer *_currentVertexBuffer,
                       *_currentVaryingBuffer;

    OskiVertexDescriptor *_vdesc;

    std::vector<Table> _tables;
    std::vector<Table> _editTables;

    oski_matrix_t A_tunable;
    oski_vecview_t x_view, y_view;
};

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* OSD_OSKI_DISPATCHER_H */

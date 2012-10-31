#include "../version.h"
#include "../osd/mutex.h"
#include "../osd/oskiDispatcher.h"
#include "../osd/oskiKernel.h"

#include <stdlib.h>
#include <string.h>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

OsdOskiKernelDispatcher::Table::~Table() {

    if (ptr)
        free(ptr);
}

void
OsdOskiKernelDispatcher::Table::Copy( int size, const void *table ) {

    if (size > 0) {
        if (ptr)
            free(ptr);
        ptr = malloc(size);
        memcpy(ptr, table, size);
    }
}

OsdOskiKernelDispatcher::OsdOskiKernelDispatcher( int levels )
    : OsdKernelDispatcher(levels), _currentVertexBuffer(NULL), _currentVaryingBuffer(NULL), _vdesc(NULL) {
    _tables.resize(TABLE_MAX);
}

OsdOskiKernelDispatcher::~OsdOskiKernelDispatcher() {

    if (_vdesc)
        delete _vdesc;
}

static OsdOskiKernelDispatcher::OsdKernelDispatcher *
Create(int levels) {
    return new OsdOskiKernelDispatcher(levels);
}

void
OsdOskiKernelDispatcher::Register() {
    Factory::GetInstance().Register(Create, kOSKI);
}

void
OsdOskiKernelDispatcher::OnKernelLaunch() {
    /* allocate sparse matrix */
}

void
OsdOskiKernelDispatcher::OnKernelFinish() {
    /* apply sparse matrix to point vector */
}

void
OsdOskiKernelDispatcher::CopyTable(int tableIndex, size_t size, const void *ptr) {

    _tables[tableIndex].Copy((int)size, ptr);
}

void
OsdOskiKernelDispatcher::AllocateEditTables(int n) {

    _editTables.resize(n*2);
    _edits.resize(n);
}

void
OsdOskiKernelDispatcher::UpdateEditTable(int tableIndex, const FarTable<unsigned int> &offsets, const FarTable<float> &values,
                                        int operation, int primVarOffset, int primVarWidth) {

    _editTables[tableIndex*2+0].Copy(offsets.GetMemoryUsed(), offsets[0]);
    _editTables[tableIndex*2+1].Copy(values.GetMemoryUsed(), values[0]);

    _edits[tableIndex].offsetOffsets.resize(_maxLevel);
    _edits[tableIndex].valueOffsets.resize(_maxLevel);
    _edits[tableIndex].numEdits.resize(_maxLevel);
    for (int i = 0; i < _maxLevel; ++i) {
        _edits[tableIndex].offsetOffsets[i] = (int)(offsets[i] - offsets[0]);
        _edits[tableIndex].valueOffsets[i] = (int)(values[i] - values[0]);
        _edits[tableIndex].numEdits[i] = offsets.GetNumElements(i);
    }
    _edits[tableIndex].operation = operation;
    _edits[tableIndex].primVarOffset = primVarOffset;
    _edits[tableIndex].primVarWidth = primVarWidth;
}

OsdVertexBuffer *
OsdOskiKernelDispatcher::InitializeVertexBuffer(int numElements, int numVertices)
{
    return new OsdCpuVertexBuffer(numElements, numVertices);
}

void
OsdOskiKernelDispatcher::BindVertexBuffer(OsdVertexBuffer *vertex, OsdVertexBuffer *varying) {

    if (vertex)
        _currentVertexBuffer = dynamic_cast<OsdCpuVertexBuffer *>(vertex);
    else
        _currentVertexBuffer = NULL;

    if (varying)
        _currentVaryingBuffer = dynamic_cast<OsdCpuVertexBuffer *>(varying);
    else
        _currentVaryingBuffer = NULL;

    _vdesc = new OskiVertexDescriptor(_currentVertexBuffer ? _currentVertexBuffer->GetNumElements() : 0,
                                  _currentVaryingBuffer ? _currentVaryingBuffer->GetNumElements() : 0);
}

void
OsdOskiKernelDispatcher::UnbindVertexBuffer()
{
    delete _vdesc;
    _vdesc = NULL;

    _currentVertexBuffer = NULL;
    _currentVaryingBuffer = NULL;
}

void
OsdOskiKernelDispatcher::Synchronize() { }


void
OsdOskiKernelDispatcher::ApplyBilinearFaceVerticesKernel( FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const {

    oskiComputeFace(_vdesc, GetVertexBuffer(), GetVaryingBuffer(),
                (int*)_tables[F_IT].ptr + _tableOffsets[F_IT][level-1],
                (int*)_tables[F_ITa].ptr + _tableOffsets[F_ITa][level-1],
                offset, start, end);
}

void
OsdOskiKernelDispatcher::ApplyBilinearEdgeVerticesKernel( FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const {

    oskiComputeBilinearEdge(_vdesc, GetVertexBuffer(), GetVaryingBuffer(),
                        (int*)_tables[E_IT].ptr + _tableOffsets[E_IT][level-1],
                        offset,
                        start, end);
}

void
OsdOskiKernelDispatcher::ApplyBilinearVertexVerticesKernel( FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const {

    oskiComputeBilinearVertex(_vdesc, GetVertexBuffer(), GetVaryingBuffer(),
                          (int*)_tables[V_ITa].ptr + _tableOffsets[V_ITa][level-1],
                          offset, start, end);
}

void
OsdOskiKernelDispatcher::ApplyCatmarkFaceVerticesKernel( FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const {

    oskiComputeFace(_vdesc, GetVertexBuffer(), GetVaryingBuffer(),
                (int*)_tables[F_IT].ptr + _tableOffsets[F_IT][level-1],
                (int*)_tables[F_ITa].ptr + _tableOffsets[F_ITa][level-1],
                offset, start, end);
}

void
OsdOskiKernelDispatcher::ApplyCatmarkEdgeVerticesKernel( FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const {

    oskiComputeEdge(_vdesc, GetVertexBuffer(), GetVaryingBuffer(),
                (int*)_tables[E_IT].ptr + _tableOffsets[E_IT][level-1],
                (float*)_tables[E_W].ptr + _tableOffsets[E_W][level-1],
                offset,
                start, end);
}

void
OsdOskiKernelDispatcher::ApplyCatmarkVertexVerticesKernelB( FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const {

    oskiComputeVertexB(_vdesc, GetVertexBuffer(), GetVaryingBuffer(),
                   (int*)_tables[V_ITa].ptr + _tableOffsets[V_ITa][level-1],
                   (int*)_tables[V_IT].ptr + _tableOffsets[V_IT][level-1],
                   (float*)_tables[V_W].ptr + _tableOffsets[V_W][level-1],
                   offset, start, end);
}

void
OsdOskiKernelDispatcher::ApplyCatmarkVertexVerticesKernelA( FarMesh<OsdVertex> * mesh, int offset, bool pass, int level, int start, int end, void * data) const {

    oskiComputeVertexA(_vdesc, GetVertexBuffer(), GetVaryingBuffer(),
                   (int*)_tables[V_ITa].ptr + _tableOffsets[V_ITa][level-1],
                   (float*)_tables[V_W].ptr + _tableOffsets[V_W][level-1],
                   offset, start, end, pass);
}

void
OsdOskiKernelDispatcher::ApplyLoopEdgeVerticesKernel( FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const {

    oskiComputeEdge(_vdesc, GetVertexBuffer(), GetVaryingBuffer(),
                (int*)_tables[E_IT].ptr + _tableOffsets[E_IT][level-1],
                (float*)_tables[E_W].ptr + _tableOffsets[E_W][level-1],
                offset,
                start, end);
}

void
OsdOskiKernelDispatcher::ApplyLoopVertexVerticesKernelB( FarMesh<OsdVertex> * mesh, int offset, int level, int start, int end, void * data) const {

    oskiComputeLoopVertexB(_vdesc, GetVertexBuffer(), GetVaryingBuffer(),
                       (int*)_tables[V_ITa].ptr + _tableOffsets[V_ITa][level-1],
                       (int*)_tables[V_IT].ptr + _tableOffsets[V_IT][level-1],
                       (float*)_tables[V_W].ptr + _tableOffsets[V_W][level-1],
                       offset, start, end);
}

void
OsdOskiKernelDispatcher::ApplyLoopVertexVerticesKernelA( FarMesh<OsdVertex> * mesh, int offset, bool pass, int level, int start, int end, void * data) const {

    oskiComputeVertexA(_vdesc, GetVertexBuffer(), GetVaryingBuffer(),
                   (int*)_tables[V_ITa].ptr + _tableOffsets[V_ITa][level-1],
                   (float*)_tables[V_W].ptr + _tableOffsets[V_W][level-1],
                   offset, start, end, pass);
}

void
OsdOskiKernelDispatcher::ApplyVertexEdits(FarMesh<OsdVertex> *mesh, int offset, int level, void * clientdata) const {
    for (int i=0; i<(int)_edits.size(); ++i) {
        const VertexEditArrayInfo &info = _edits[i];

        if (info.operation == FarVertexEdit::Add) {
            oskiEditVertexAdd(_vdesc, GetVertexBuffer(), info.primVarOffset, info.primVarWidth, info.numEdits[level-1],
                          (int*)_editTables[i*2+0].ptr + info.offsetOffsets[level-1],
                          (float*)_editTables[i*2+1].ptr + info.valueOffsets[level-1]);
        } else if (info.operation == FarVertexEdit::Set) {
            oskiEditVertexSet(_vdesc, GetVertexBuffer(), info.primVarOffset, info.primVarWidth, info.numEdits[level],
                          (int*)_editTables[i*2+0].ptr + info.offsetOffsets[level],
                          (float*)_editTables[i*2+1].ptr + info.valueOffsets[level]);
        }
    }
}

} // end namespace OPENSUBDIV_VERSION

} // end namespace OpenSubdiv

#include "../version.h"
#include "../osd/mutex.h"
#include "../osd/oskiDispatcher.h"
#include "../osd/oskiKernel.h"

#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <sys/time.h>

#include <boost/foreach.hpp>
#define foreach BOOST_FOREACH

#define TIMEBLOCK(name, body)                                          \
        timeval name ## start, name ## end;                            \
        gettimeofday(&name ## start, NULL);                            \
        { body }                                                       \
        gettimeofday(&name ## end, NULL);                              \
        double name ## duration =                                      \
            (name ## end.tv_sec - name ## start.tv_sec) * 1000.0 +     \
            (name ## end.tv_usec - name ## start.tv_usec) / 1000.0;    \
        printf("%s took %f milliseconds.\n", #name, name ## duration);

using namespace std;
using namespace boost::numeric::ublas;

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
    : OsdKernelDispatcher(levels), A_tunable(NULL), _currentVertexBuffer(NULL), _currentVaryingBuffer(NULL), _vdesc(NULL) {
    _tables.resize(TABLE_MAX);
    M = NULL;
    S = NULL;

    oski_Init();
}

OsdOskiKernelDispatcher::~OsdOskiKernelDispatcher() {
    oski_Close();

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
}

void
OsdOskiKernelDispatcher::OnKernelFinish() {
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

    _vdesc = new OskiVertexDescriptor(this,
            _currentVertexBuffer ? _currentVertexBuffer->GetNumElements() : 0,
            _currentVaryingBuffer ? _currentVaryingBuffer->GetNumElements() : 0);
}

void
OsdOskiKernelDispatcher::StageMatrix(int i, int j)
{
    if (S != NULL) delete S;
    S = new coordinate_matrix<float>(i,j);
}

void
OsdOskiKernelDispatcher::StageElem(int i, int j, float value)
{
    assert(0 <= i);
    assert(i < S->size1());
    assert(0 <= j);
    assert(j < S->size2());

    (*S)(i,j) = value;
}

void
OsdOskiKernelDispatcher::PushMatrix()
{
    if (M != NULL) {
        compressed_matrix<float> A(*S);
        compressed_matrix<float> *B = M;
        compressed_matrix<float> *C = new compressed_matrix<float>(A.size1(), B->size2());
        printf("PushMatrix mul %d-%d = %d-%d * %d-%d\n",
                (int) C->size1(), (int) C->size2(),
                (int) A.size1(), (int) A.size2(),
                (int) B->size1(), (int) B->size2());
        axpy_prod(A, *B, *C, true);
        M = C;
        delete B;
    } else {
        M = new compressed_matrix<float>(*S);
        printf("PushMatrix set %d-%d\n", (int) M->size1(), (int) M->size2());
    }

    assert(M);
    delete S;
    S = NULL;
}

void
OsdOskiKernelDispatcher::ApplyM(int offset)
{
    if (A_tunable == NULL) {
        int numElems = _currentVertexBuffer->GetNumElements();
        float* V_in = _currentVertexBuffer->GetCpuBuffer();
        float* V_out = _currentVertexBuffer->GetCpuBuffer() + offset * numElems;

        x_view = oski_CreateVecView( V_in, M->size2(), STRIDE_UNIT );
        y_view = oski_CreateVecView( V_out, M->size1(), STRIDE_UNIT );

        std::vector<int> rowIndx(M->index1_data().begin(), M->index1_data().end());
        std::vector<int> colIndx(M->index2_data().begin(), M->index2_data().end());
        std::vector<float>  vals(M->value_data().begin(),  M->value_data().end());

        A_tunable = oski_CreateMatCSR(
                &rowIndx[0],        // row ptrs
                &colIndx[0],        // idx ptrs
                &vals[0],           // values
                M->size1(),         // num rows
                M->size2(),         // num cols
                COPY_INPUTMAT,      // both use and oski share array
                1,                  // number of args to follow
                INDEX_ZERO_BASED    // zero based indexing
                );

        oski_SetHintMatMult( A_tunable, OP_NORMAL,
                1.0, x_view, 0.0, y_view, ALWAYS_TUNE_AGGRESSIVELY );
        oski_SetHint( A_tunable, HINT_NO_BLOCKS, ARGS_NONE );
        oski_TuneMat( A_tunable );

        WriteM();
    }

    oski_MatMult( A_tunable, OP_NORMAL, 1.0, x_view, 0.0, y_view );
}

void
OsdOskiKernelDispatcher::WriteM()
{
    MM_typecode matcode;

    std::vector<int> rows(M->index1_data().begin(), M->index1_data().end());
    std::vector<int> cols(M->index2_data().begin(), M->index2_data().end());
    std::vector<float>  vals(M->value_data().begin(),  M->value_data().end());

    int *I = &rows[0];
    int *J = &cols[0];
    float *val = &vals[0];
    int Mlen = (int) M->size1();
    int Nlen = (int) M->size2();
    int nz = vals.size();

    FILE* ofile = fopen("subdiv_matrix.mm", "w");
    assert(ofile != NULL);

    mm_initialize_typecode(&matcode);
    mm_set_matrix(&matcode);
    mm_set_coordinate(&matcode);
    mm_set_real(&matcode);

    mm_write_banner(ofile, matcode);
    mm_write_mtx_crd_size(ofile, Mlen, Nlen, nz);

    for (int i=0; i<nz; i++)
        fprintf(ofile, "%d %d %10.3g\n", I[i]+1, J[i]+1, val[i]);

    fclose(ofile);
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

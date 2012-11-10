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

OsdOskiKernelDispatcher::OsdOskiKernelDispatcher( int levels )
    : OsdCpuKernelDispatcher(levels), A_tunable(NULL) {
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
            _currentVertexBuffer  ? _currentVertexBuffer->GetNumElements()  : 0,
            _currentVaryingBuffer ? _currentVaryingBuffer->GetNumElements() : 0);
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
        csr_matrix A(*S);
        csr_matrix *B = M;
        csr_matrix *C = new csr_matrix(A.size1(), B->size2());
        printf("PushMatrix mul %d-%d = %d-%d * %d-%d\n",
                (int) C->size1(), (int) C->size2(),
                (int) A.size1(), (int) A.size2(),
                (int) B->size1(), (int) B->size2());
        axpy_prod(A, *B, *C, true);
        M = C;
        delete B;
    } else {
        M = new csr_matrix(*S);
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
        assert(M != NULL);
        assert(_currentVertexBuffer != NULL);

        int numElems = _currentVertexBuffer->GetNumElements();
        float* V_in = _currentVertexBuffer->GetCpuBuffer();
        float* V_out = _currentVertexBuffer->GetCpuBuffer() + offset * numElems;

        x_view = oski_CreateVecView( V_in, M->size2(), STRIDE_UNIT );
        y_view = oski_CreateVecView( V_out, M->size1(), STRIDE_UNIT );

        A_tunable = oski_CreateMatCSR(
                &M->index1_data()[0], // row ptrs
                &M->index2_data()[0], // idx ptrs
                &M->value_data()[0],  // values
                M->size1(),           // num rows
                M->size2(),           // num cols
                COPY_INPUTMAT,        // both use and oski share array
                1,                    // number of args to follow
                INDEX_ZERO_BASED      // zero based indexing
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

    int *I = &M->index1_data()[0];
    int *J = &M->index2_data()[0];
    float *val = &M->value_data()[0];
    int Mlen = (int) M->size1() / 6;
    int Nlen = (int) M->size2() / 6;
    int nz = M->value_data().size();

    FILE* ofile = fopen("subdiv_matrix.mm", "w");
    assert(ofile != NULL);

    mm_initialize_typecode(&matcode);
    mm_set_matrix(&matcode);
    mm_set_coordinate(&matcode);
    mm_set_real(&matcode);

    mm_write_banner(ofile, matcode);
    mm_write_mtx_crd_size(ofile, Mlen, Nlen, nz);

    for(int i = 0; i < M->size1(); i++)
        for(int j = 0; j < M->size2(); j++)
            if ((*M)(i,j) != 0.0 && i%6==0 && j%6==0)
                fprintf(ofile, "%d %d %10.3g\n", i/6+1, j/6+1, (float) (*M)(i,j));

    fclose(ofile);
}

} // end namespace OPENSUBDIV_VERSION

} // end namespace OpenSubdiv

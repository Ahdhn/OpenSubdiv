#include "../version.h"
#include "../osd/mutex.h"
#include "../osd/mklDispatcher.h"

#include <stdio.h>

using namespace boost::numeric::ublas;

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

OsdMklKernelDispatcher::OsdMklKernelDispatcher( int levels )
    : OsdSpMVKernelDispatcher(levels),
    S(NULL), M(NULL), mj(NULL), mi(NULL),
    Mlen(0), milen(0), mjlen(0), m(0), n(0)
{ }

OsdMklKernelDispatcher::~OsdMklKernelDispatcher()
{
    if (S) delete S;
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
    S = new coo_matrix1(i,j);
}

inline void
OsdMklKernelDispatcher::StageElem(int i, int j, float value)
{
#ifdef DEBUG
    assert(0 <= i);
    assert(i < S->size1());
    assert(0 <= j);
    assert(j < S->size2());
#endif
    (*S)(i,j) = value;
}

void
OsdMklKernelDispatcher::PushMatrix()
{
    /* convert coo matrix to csr */
    csr_matrix1 A(*S);
    delete S;

    /* TODO set sizes */
    Mlen = 0;
    milen = 0;
    mjlen = 0;

    /* if no M exists, create one from A */
    if (M == NULL) {
        M = new float[Mlen];
        mj = new int[mjlen];
        mi = new int[milen];
        return;
    }

    /* otherwise, compute the new M */
    char trans = 'N'; // no transpose A
    int request = 0; // output arrays pre allocated
    int sort = 7; // reordering of zeroes
    int m = A.size1(); // rows of A
    int n = A.size2(); // cols of A
    int k = 0; // rows of B
    float* a = &A.value_data()[0]; // A values
    int* ja = &A.index2_data()[0]; // A col indices
    int* ia = &A.index1_data()[0]; // A row ptrs
    float* b = M; // B values
    int* jb = mj; // B col indices
    int* ib = mi; // B row ptrs
    int nzmax = Mlen; // max number of nonzeroes
    float* c = new float[Mlen];
    int* jc = new int[mjlen];
    int* ic = new int[milen];
    int info; // output info flag

    /* perform SpM*SpM */
    mkl_scsrmultcsr(&trans, &request, &sort,
            &m, &n, &k, a, ja, ia, b, jb, ib,
            c, jc, ic, &nzmax, &info);
    assert(info == 0);

    /* clean up and replace old M */
    delete[] M;
    delete[] mi;
    delete[] mj;
    M = c;
    mi = ic;
    mj = jc;
}

void
OsdMklKernelDispatcher::ApplyMatrix(int offset)
{
    int numElems = _currentVertexBuffer->GetNumElements();
    float* V_in = _currentVertexBuffer->GetCpuBuffer();
    float* V_out = _currentVertexBuffer->GetCpuBuffer()
                   + offset * numElems;

    char transa = 'N';
    int m = milen;
    float* a = M;
    int* ia = mi;
    int* ja = mj;
    float* x = V_in;
    float* y = V_out;

    mkl_scsrgemv(&transa, &m, a, ia, ja, x, y);
}

void
OsdMklKernelDispatcher::WriteMatrix()
{
    assert(!"WriteMatrix not implemented for MKL dispatcher.");
}

bool
OsdMklKernelDispatcher::MatrixReady()
{
    return (M == NULL);
}

void
OsdMklKernelDispatcher::PrintReport()
{
    assert(!"PrintReport not implemented for MKL dispatcher.");
}

} // end namespace OPENSUBDIV_VERSION

} // end namespace OpenSubdiv

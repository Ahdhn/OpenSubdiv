#include "../version.h"
#include "../osd/mutex.h"
#include "../osd/ublasDispatcher.h"

#include <stdio.h>
#include <boost/numeric/ublas/operation.hpp>

using namespace boost::numeric::ublas;

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

OsdUBlasKernelDispatcher::OsdUBlasKernelDispatcher( int levels )
    : OsdSpMVKernelDispatcher(levels), S(NULL), M(NULL)
{ }

OsdUBlasKernelDispatcher::~OsdUBlasKernelDispatcher()
{
    if (S) delete S;
}

static OsdUBlasKernelDispatcher::OsdKernelDispatcher *
Create(int levels) {
    return new OsdUBlasKernelDispatcher(levels);
}

void
OsdUBlasKernelDispatcher::Register() {
    Factory::GetInstance().Register(Create, kUBLAS);
}

void
OsdUBlasKernelDispatcher::StageMatrix(int i, int j)
{
    if (S != NULL) delete S;

    int numElems = _currentVertexBuffer->GetNumElements();
    S = new coo_matrix(i*numElems,j*numElems);
}

inline void
OsdUBlasKernelDispatcher::StageElem(int i, int j, float value)
{
#ifdef DEBUG
    assert(0 <= i);
    assert(i < S->size1());
    assert(0 <= j);
    assert(j < S->size2());
#endif
    int numElems = _currentVertexBuffer->GetNumElements();
    for(int k = 0; k < numElems; k++)
        (*S)(i*numElems+k,j*numElems+k) = value;
}

void
OsdUBlasKernelDispatcher::FinalizeMatrix()
{
    this->PrintReport();
}

void
OsdUBlasKernelDispatcher::PushMatrix()
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

    WriteMatrix(M, "subdiv_matrix");
}

void
OsdUBlasKernelDispatcher::ApplyMatrix(int offset)
{
    int numElems = _currentVertexBuffer->GetNumElements();
    float* V_in = _currentVertexBuffer->GetCpuBuffer();
    float* V_out = _currentVertexBuffer->GetCpuBuffer()
                   + offset * numElems;

    vector<float> vin(M->size2());
    vector<float> vout(M->size1());

    for(int i = 0; i < vin.size(); i++)
        vin(i) = V_in[i];

    axpy_prod(*M, vin, vout, true);

    for(int i = 0; i < vout.size(); i++)
        V_out[i] = vout(i);
}

bool
OsdUBlasKernelDispatcher::MatrixReady()
{
    return (M != NULL);
}

void
OsdUBlasKernelDispatcher::PrintReport()
{
    assert(M != NULL);
    int size_in_bytes =
            (M->index1_data().size() +
             M->index2_data().size() +
             M->value_data().size()) * sizeof(float);
    printf("Subdivision matrix is %d-by-%d with %d nonzeroes (%f%%) %dMB\n",
            (int) M->size1(),
            (int) M->size2(),
            (int) M->value_data().size(),
            100.0 * M->value_data().size() / (M->size1() * M->size2()),
            size_in_bytes / 1024 / 1024
          );
}

} // end namespace OPENSUBDIV_VERSION

} // end namespace OpenSubdiv

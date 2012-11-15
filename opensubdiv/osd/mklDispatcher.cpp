#include "../version.h"
#include "../osd/mutex.h"
#include "../osd/mklDispatcher.h"

#include <stdio.h>

using namespace boost::numeric::ublas;

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

OsdMklKernelDispatcher::OsdMklKernelDispatcher( int levels )
    : OsdSpMVKernelDispatcher(levels), S(NULL), M(NULL)
{ }

OsdMklKernelDispatcher::~OsdMklKernelDispatcher()
{
    if (S) delete S;
    if (M) delete M;
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
    S->append_element(i, j, value);
}

void
OsdMklKernelDispatcher::PushMatrix()
{
    /* if no M exists, create one from A */
    if (M == NULL) {

        printf("PushMatrix set %d-%d\n", S->size1(), S->size2());
        M = new csr_matrix1(*S);

    } else {

#if INTEL_DIDNT_PUT_A_DUMMY_ELEMENT_AT_END_OF_INDEX1_DATA_IN_THEIR_CSR_FORMAT
        /* convert S from COO to CSR format efficiently */
        csr_matrix1 A(S->size1(), S->size2(), S->value_data().size());
        {
            int nnz = S->value_data().size();
            int job[] = {
                2, // job(1)=2 (coo->csr with sorting)
                1, // job(2)=1 (one-based indexing for csr matrix)
                1, // job(3)=1 (one-based indexing for coo matrix)
                0, // empty
                nnz, // job(5)=nnz (sets nnz for csr matrix)
                0  // job(6)=0 (all output arrays filled)
            };
            int n = A.size1();
            float* acsr = &A.value_data()[0];
            int* ja = &A.index2_data()[0];
            int* ia = &A.index1_data()[0];
            float* acoo = &S->value_data()[0];
            int* rowind = &S->index1_data()[0];
            int* colind = &S->index2_data()[0];
            int info;
            mkl_scsrcoo(job, &n, acsr, ja, ia, &nnz, acoo, rowind, colind, &info);
            assert(info == 0);
        }
#else
        csr_matrix1 A(*S);
#endif

        int i = A.size1(),
            j = M->size2(),
            nnz = M->value_data().size() * 6; // XXX: shouldn't this be 4?
        csr_matrix1 *C = new csr_matrix1(i, j, nnz);

        char trans = 'N'; // no transpose A
        int request = 0; // output arrays pre allocated
        int sort = 7; // reordering of zeroes
        int m = A.size1(); // rows of A
        int n = A.size2(); // cols of A
        int k = M->size1(); // rows of B
        float* a = &A.value_data()[0]; // A values
        int* ja = &A.index2_data()[0]; // A col indices
        int* ia = &A.index1_data()[0]; // A row ptrs
        float* b = &M->value_data()[0]; // B values
        int* jb = &M->index2_data()[0]; // B col indices
        int* ib = &M->index1_data()[0]; // B row ptrs
        int nzmax = C->value_data().size(); // max number of nonzeroes
        float* c = &C->value_data()[0];
        int* jc = &C->index2_data()[0];
        int* ic = &C->index1_data()[0];
        int info; // output info flag

        /* perform SpM*SpM */
        printf("PushMatrix mul %d-%d = %d-%d * %d-%d\n",
                (int) C->size1(), (int) C->size2(),
                (int) A.size1(), (int) A.size2(),
                (int) M->size1(), (int) M->size2());
        mkl_scsrmultcsr(&trans, &request, &sort,
                &m, &n, &k, a, ja, ia, b, jb, ib,
                c, jc, ic, &nzmax, &info);

        if (info != 0) {
            printf("Error: info returned %d\n", info);
            assert(info == 0);
        }

        delete M;
        M = C;
    }

    /* remove staged matrix */
    delete S;
    S = NULL;
}

void
OsdMklKernelDispatcher::ApplyMatrix(int offset)
{
    int numElems = _currentVertexBuffer->GetNumElements();
    float* V_in = _currentVertexBuffer->GetCpuBuffer();
    float* V_out = _currentVertexBuffer->GetCpuBuffer()
                   + offset * numElems;

    char transa = 'N';
    int m = M->size1();
    float* a = &M->value_data()[0];
    int* ia = &M->index1_data()[0];
    int* ja = &M->index2_data()[0];
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
    return (M != NULL);
}

void
OsdMklKernelDispatcher::PrintReport()
{
    printf("Subdiv matrix is %d-by-%d with %f%% nonzeroes.\n",
        M->size1(), M->size2(),
        100.0 * M->value_data().size() / M->size1() / M->size2());
}

} // end namespace OPENSUBDIV_VERSION

} // end namespace OpenSubdiv

#include "../version.h"
#include "../osd/mutex.h"
#include "../osd/mklDispatcher.h"

#include <stdio.h>

using namespace boost::numeric::ublas;

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

OsdMklKernelDispatcher::OsdMklKernelDispatcher( int levels )
    : OsdSpMVKernelDispatcher(levels), S(NULL), M(NULL), M_big(NULL)
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

        //printf("PushMatrix set %d-%d\n", S->size1(), S->size2());
        M = new csr_matrix1(*S);

    } else {

        /* convert S from COO to CSR format efficiently */
        csr_matrix1 A(S->size1(), S->size2(), S->nnz());
        {
            int nnz = S->nnz();
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
            A.set_filled(n+1, A.index1_data()[n] - 1);
        }

        int i = A.size1(),
            j = M->size2(),
            nnz = std::min(i*j, (int) M->nnz() * 7); // XXX: shouldn't this be 4?
        csr_matrix1 *C = new csr_matrix1(i, j, nnz);

        char trans = 'N'; // no transpose A
        int request = 0; // output arrays pre allocated
        int sort = 8; // reorder nonzeroes in C
        int m = A.size1(); // rows of A
        int n = A.size2(); // cols of A
        int k = M->size2(); // cols of B
        assert(A.size2() == M->size1());

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
        int info = 0; // output info flag

        /* perform SpM*SpM */
        //printf("PushMatrix mul %d-%d = %d-%d * %d-%d\n",
        //        (int) C->size1(), (int) C->size2(),
        //        (int) A.size1(),  (int) A.size2(),
        //        (int) M->size1(), (int) M->size2());
        mkl_scsrmultcsr(&trans, &request, &sort,
                &m, &n, &k, a, ja, ia, b, jb, ib,
                c, jc, ic, &nzmax, &info);

        if (info != 0) {
            printf("Error: info returned %d\n", info);
            assert(info == 0);
        }

        /* update csr_mutrix1 state to reflect mkl writes */
        C->set_filled(i+1, C->index1_data()[i] - 1);

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
    int m = M_big->size1();
    float* a = &M_big->value_data()[0];
    int* ia = &M_big->index1_data()[0];
    int* ja = &M_big->index2_data()[0];
    float* x = V_in;
    float* y = V_out;

    mkl_scsrgemv(&transa, &m, a, ia, ja, x, y);
}

void
OsdMklKernelDispatcher::FinalizeMatrix()
{
    /* expand M to M_big if necessary */
    if (M_big == NULL) {
        int nve = _currentVertexBuffer->GetNumElements();
        coo_matrix1 M_big_coo(M->size1()*nve, M->size2()*nve, M->nnz()*nve);

        /* build M_big_coo matrix from M */
        for(int i = 0; i < M->size1(); i++) {
            for( int j = M->index1_data()[i]; j < M->index1_data()[i+1]; j++ ) {
                float factor = M->value_data()[ j-1 ];
                int ii = i;
                int jj = M->index2_data()[ j-1 ] - 1;
                for(int k = 0; k < nve; k++)
                    M_big_coo.append_element(ii*nve+k, jj*nve+k, factor);
            }
        }

        /* convert M_big_coo from COO to CSR format efficiently */
        M_big = new csr_matrix1(M_big_coo.size1(), M_big_coo.size2(), M_big_coo.nnz());
        {
            int nnz = M_big_coo.nnz();
            int job[] = {
                2, // job(1)=2 (coo->csr with sorting)
                1, // job(2)=1 (one-based indexing for csr matrix)
                1, // job(3)=1 (one-based indexing for coo matrix)
                0, // empty
                nnz, // job(5)=nnz (sets nnz for csr matrix)
                0  // job(6)=0 (all output arrays filled)
            };
            int n = M_big->size1();
            float* acsr = &M_big->value_data()[0];
            int* ja = &M_big->index2_data()[0];
            int* ia = &M_big->index1_data()[0];
            float* acoo = &M_big_coo.value_data()[0];
            int* rowind = &M_big_coo.index1_data()[0];
            int* colind = &M_big_coo.index2_data()[0];
            int info;
            mkl_scsrcoo(job, &n, acsr, ja, ia, &nnz, acoo, rowind, colind, &info);
            assert(info == 0);
            M_big->set_filled(n+1, M_big->index1_data()[n] - 1);
        }
    }

    this->PrintReport();
}

bool
OsdMklKernelDispatcher::MatrixReady()
{
    return (M_big != NULL);
}

void
OsdMklKernelDispatcher::PrintReport()
{
    int size_in_bytes =  (int) (M_big->index2_data().capacity() +
                                M_big->index1_data().capacity()) * sizeof(int)  +
                                M_big->value_data().capacity() * sizeof(float);
    double sparsity_factor = 100.0 * M_big->nnz() / M_big->size1() / M_big->size2();

#if BENCHMARKING
    printf(" nverts=%d", M->size1());
    printf(" size=%d", size_in_bytes);
    printf(" sparsity=%f", sparsity_factor);
#else
    printf("Subdiv matrix is %d-by-%d with %f%% nonzeroes, takes %d MB.\n",
        M_big->size1(), M_big->size2(), sparsity_factor, size_in_bytes / 1024 / 1024);
#endif
}

} // end namespace OPENSUBDIV_VERSION

} // end namespace OpenSubdiv

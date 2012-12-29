#include "../version.h"
#include "../osd/mutex.h"
#include "../osd/mklDispatcher.h"

#include <stdio.h>

using namespace boost::numeric::ublas;

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

Matrix::Matrix(int m, int n, int nnz, int nve, mode_t mode) :
    m(m), n(n), nnz(nnz), nve(nve), mode(mode) {
    this->rows = (int*) malloc((m+1) * sizeof(int));
    this->cols = (int*) malloc(n * sizeof(int));
    this->vals = (float*) malloc(m * sizeof(float));
    this->rows[m] = nnz;
}

Matrix::Matrix(const coo_matrix1* S, int nve, mode_t mode) :
    nve(nve), mode(mode) {
    m = S->size1();
    n = S->size2();
    nnz = S->nnz();

    this->rows = (int*) malloc((m+1) * sizeof(int));
    this->cols = (int*) malloc(n * sizeof(int));
    this->vals = (float*) malloc(m * sizeof(float));

    int job[] = {
        2, // job(1)=2 (coo->csr with sorting)
        1, // job(2)=1 (one-based indexing for csr matrix)
        1, // job(3)=1 (one-based indexing for coo matrix)
        0, // empty
        nnz, // job(5)=nnz (sets nnz for csr matrix)
        0  // job(6)=0 (all output arrays filled)
    };
    float* acoo = (float*) &S->value_data()[0];
    int* rowind = (int*) &S->index1_data()[0];
    int* colind = (int*) &S->index2_data()[0];
    int info;

    mkl_scsrcoo(job, &m, vals, cols, rows, &nnz, acoo, rowind, colind, &info);
    assert(info == 0);

    this->rows[m] = nnz;
}

int
Matrix::NumBytes() const {
    return nnz*sizeof(float) + nnz*sizeof(int) + (m+1)*sizeof(int);
}

double
Matrix::SparsityFactor() const {
    return (double) nnz / (double) (m * n);
}

void
Matrix::spmv(float* d_out, float* d_in) {
    if (mode == Matrix::VERTEX)
        expand();

    char transa = 'N';
    mkl_scsrgemv(&transa, &m, vals, rows, cols, d_in, d_out);
}

Matrix*
Matrix::gemm(Matrix* rhs) {
    if (rhs->mode != this->mode) {
        rhs->expand();
        this->expand();
    }

    Matrix* A = this;
    Matrix* B = rhs;

    int nnz = std::min(A->m*B->n, (int)nnz*7); // XXX: shouldn't this be 4, not 7?
    Matrix* C = new Matrix(A->m, B->n, C->nnz, nve, mode);

    char trans = 'N'; // no transpose A
    int request = 0; // output arrays pre allocated
    int sort = 8; // reorder nonzeroes in C
    int info = 0; // output info flag
    assert(A->n == B->m);

    /* perform SpM*SpM */
    //printf("PushMatrix mul %d-%d = %d-%d * %d-%d\n",
    //        (int) C->size1(), (int) C->size2(),
    //        (int) A.size1(),  (int) A.size2(),
    //        (int) M->size1(), (int) M->size2());
    mkl_scsrmultcsr(&trans, &request, &sort,
            &A->m, &A->n, &B->n, A->vals, A->cols, A->rows, B->vals, B->cols, B->rows,
            C->vals, C->cols, C->rows, &C->nnz, &info);

    if (info != 0) {
        printf("Error: info returned %d\n", info);
        assert(info == 0);
    }
}

Matrix*
Matrix::gemm(const coo_matrix1* rhs) {
    Matrix* rhs_csr = new Matrix(rhs);
    Matrix* answer = this->gemm(rhs_csr);
    delete rhs_csr;
    return answer;
}

void
Matrix::expand() {
    if (mode == Matrix::VERTEX) {
        // call expander with nve
        mode = Matrix::ELEMENT;
    }
}

void
Matrix::report(std::string name) {
}

Matrix::~Matrix() {
    free(rows);
    free(cols);
    free(vals);
}

OsdMklKernelDispatcher::OsdMklKernelDispatcher( int levels )
    : OsdSpMVKernelDispatcher(levels), S(NULL)
{ }

OsdMklKernelDispatcher::~OsdMklKernelDispatcher()
{
    if (S != NULL)
        delete S;
    if (subdiv_operator != NULL)
        delete subdiv_operator;
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
    /* if no subdiv_operator exists, create one from A */
    if (subdiv_operator == NULL) {

        //printf("PushMatrix set %d-%d\n", S->size1(), S->size2());
        subdiv_operator = new Matrix(S);

    } else {
#if 0
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
#endif
        Matrix* new_subdiv_operator = subdiv_operator->gemm(S);
        delete subdiv_operator;
        subdiv_operator = new_subdiv_operator;
    }

    /* remove staged matrix */
    delete S;
    S = NULL;
}

void
OsdMklKernelDispatcher::ApplyMatrix(int offset)
{
    int numElems = _currentVertexBuffer->GetNumElements();
    float* V_in = (float*) _currentVertexBuffer->Map();
    float* V_out = (float*) _currentVertexBuffer->Map()
                   + offset * numElems;

#if 0
    char transa = 'N';
    int m = M_big->size1();
    float* a = &M_big->value_data()[0];
    int* ia = &M_big->index1_data()[0];
    int* ja = &M_big->index2_data()[0];
    float* x = V_in;
    float* y = V_out;

    mkl_scsrgemv(&transa, &m, a, ia, ja, x, y);
#endif

    subdiv_operator->spmv(V_out, V_in);
}

void
OsdMklKernelDispatcher::FinalizeMatrix()
{
    /* expand M to M_big if necessary */
    subdiv_operator->expand();
#if 0
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
#endif

    this->PrintReport();

#if 0
    if (osdSpMVKernel_DumpSpy_FileName != NULL) {
        this->WriteMatrix(M, osdSpMVKernel_DumpSpy_FileName);
    }
#endif
}

bool
OsdMklKernelDispatcher::MatrixReady()
{
    return (subdiv_operator != NULL);
}

void
OsdMklKernelDispatcher::PrintReport()
{
    int size_in_bytes = subdiv_operator->NumBytes();
    double sparsity_factor = 100.0 * subdiv_operator->SparsityFactor();

#if BENCHMARKING
    printf(" nverts=%d", subdiv_operator->nnz);
    printf(" mem=%d", size_in_bytes);
    printf(" sparsity=%f", sparsity_factor);
#else
    printf("Subdiv matrix is %d-by-%d with %f%% nonzeroes, takes %d MB.\n",
        subdiv_operator->m, subdiv_operator->n, sparsity_factor, size_in_bytes / 1024 / 1024);
#endif
}

} // end namespace OPENSUBDIV_VERSION

} // end namespace OpenSubdiv

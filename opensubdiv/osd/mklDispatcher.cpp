#include "../version.h"
#include "../osd/mklDispatcher.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

Matrix::Matrix(int m, int n, int nnz, int nve, mode_t mode) :
    m(m), n(n), nnz(nnz), nve(nve), mode(mode) {
    rows = (int*) malloc((m+1) * sizeof(int));
    cols = (int*) malloc(nnz * sizeof(int));
    vals = (float*) malloc(nnz * sizeof(float));
    rows[m] = nnz+1;
}

Matrix::Matrix(const coo_matrix1* S, int nve, mode_t mode) :
    m(S->size1()), n(S->size2()), nnz(S->nnz()), nve(nve), mode(mode) {

    rows = (int*) malloc((m+1) * sizeof(int));
    cols = (int*) malloc(nnz * sizeof(int));
    vals = (float*) malloc(nnz * sizeof(float));

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

    mkl_scsrgemv("N", &m, vals, rows, cols, d_in, d_out);
}

Matrix*
Matrix::gemm(Matrix* rhs) {
    if (rhs->mode != this->mode) {
        rhs->expand();
        this->expand();
    }

    Matrix* A = this;
    Matrix* B = rhs;

    int c_nnz = std::min(A->m*B->n, (int) B->nnz*7); // XXX: shouldn't this be 4, not 7?
    Matrix* C = new Matrix(A->m, B->n, c_nnz, nve, mode);

    int request = 0; // output arrays pre allocated
    int sort = 8; // reorder nonzeroes in C
    int info = 0; // output info flag
    assert(A->n == B->m);

    /* perform SpM*SpM */
    mkl_scsrmultcsr("N", &request, &sort,
            &A->m, &A->n, &B->n,
            A->vals, A->cols, A->rows,
            B->vals, B->cols, B->rows,
            C->vals, C->cols, C->rows,
            &C->nnz, &info);

    if (info != 0) {
        printf("Error: info returned %d\n", info);
        assert(info == 0);
    }
    return C;
}

Matrix*
Matrix::gemm(const coo_matrix1* lhs) {
    Matrix* lhs_csr = new Matrix(lhs);
    Matrix* answer = lhs_csr->gemm(this);
    delete lhs_csr;
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
    : OsdSpMVKernelDispatcher(levels), S(NULL), subdiv_operator(NULL)
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
        subdiv_operator = new Matrix(S);
#if !BENCHMARKING
        printf("PushMatrix set %d-%d\n", subdiv_operator->m, subdiv_operator->n);
#endif
    } else {
        Matrix* new_subdiv_operator = subdiv_operator->gemm(S);
#if !BENCHMARKING
        printf("PushMatrix mul %d-%d = %d-%d * %d-%d\n",
                (int) new_subdiv_operator->m, (int) new_subdiv_operator->n,
                (int) S->size1(), (int) S->size2(),
                (int) subdiv_operator->m, (int) subdiv_operator->n);
#endif
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

    subdiv_operator->spmv(V_out, V_in);
}

void
OsdMklKernelDispatcher::FinalizeMatrix()
{
    /* expand M to M_big if necessary */
    subdiv_operator->expand();
    this->PrintReport();

#if 0
    if (osdSpMVKernel_DumpSpy_FileName != NULL) {
        this->WriteMatrix(subdiv_operator, osdSpMVKernel_DumpSpy_FileName);
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

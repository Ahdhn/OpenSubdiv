#include "../version.h"
#include "../osd/mklDispatcher.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

CooMatrix::CooMatrix(int m, int n) : m(m), n(n)
{ }

void
CooMatrix::append_element(int i, int j, float val) {
#ifdef DEBUG
    assert(0 <= i);
    assert(i < StagedOp->m);
    assert(0 <= j);
    assert(j < StagedOp->n);
#endif

    rows.push_back(i+1); // one-based indexing
    cols.push_back(j+1);
    vals.push_back(val);
}

int
CooMatrix::nnz() const {
    return vals.size();
}

CsrMatrix::CsrMatrix(int m, int n, int nnz, int nve, mode_t mode) :
    m(m), n(n), nve(nve), mode(mode) {
    rows = (int*) malloc((m+1) * sizeof(int));
    cols = (int*) malloc(nnz * sizeof(int));
    vals = (float*) malloc(nnz * sizeof(float));
    rows[m] = nnz+1;
}

CsrMatrix::CsrMatrix(const CooMatrix* StagedOp, int nve, mode_t mode) :
    nve(nve), mode(mode) {

    m = StagedOp->m;
    n = StagedOp->n;
    int numnz = StagedOp->nnz();
    rows = (int*) malloc((m+1) * sizeof(int));
    cols = (int*) malloc(numnz * sizeof(int));
    vals = (float*) malloc(numnz * sizeof(float));

    int job[] = {
        2, // job(1)=2 (coo->csr with sorting)
        1, // job(2)=1 (one-based indexing for csr matrix)
        1, // job(3)=1 (one-based indexing for coo matrix)
        0, // empty
        numnz, // job(5)=nnz (sets nnz for csr matrix)
        0  // job(6)=0 (all output arrays filled)
    };

    float* acoo = (float*) &StagedOp->vals[0];
    int* rowind = (int*) &StagedOp->rows[0];
    int* colind = (int*) &StagedOp->cols[0];
    int info;

    mkl_scsrcoo(job, &m, vals, cols, rows, &numnz, acoo, rowind, colind, &info);
    assert(info == 0);
}

int
CsrMatrix::nnz() {
    return this->rows[m];
}

int
CsrMatrix::NumBytes() {
    return nnz()*sizeof(float) + nnz()*sizeof(int) + (m+1)*sizeof(int);
}

double
CsrMatrix::SparsityFactor() {
    return (double) nnz() / (double) (m * n);
}

void
CsrMatrix::spmv(float* d_out, float* d_in) {
    assert(mode == CsrMatrix::ELEMENT);
    mkl_scsrgemv((char*)"N", &m, vals, rows, cols, d_in, d_out);
}

CsrMatrix*
CsrMatrix::gemm(CsrMatrix* rhs) {
    if (rhs->mode != this->mode) {
        rhs->expand();
        this->expand();
    }

    CsrMatrix* A = this;
    CsrMatrix* B = rhs;

    int c_nnz = std::min(A->m*B->n, (int) B->nnz()*7); // XXX: shouldn't this be 4, not 7?
    CsrMatrix* C = new CsrMatrix(A->m, B->n, c_nnz, B->nve, mode);

    int request = 0; // output arrays pre allocated
    int sort = 8; // reorder nonzeroes in C
    int info = 0; // output info flag
    assert(A->n == B->m);

    /* perform SpM*SpM */
    mkl_scsrmultcsr((char*)"N", &request, &sort,
            &A->m, &A->n, &B->n,
            A->vals, A->cols, A->rows,
            B->vals, B->cols, B->rows,
            C->vals, C->cols, C->rows,
            &c_nnz, &info);

    if (info != 0) {
        printf("Error: info returned %d\n", info);
        assert(info == 0);
    }
    return C;
}

CsrMatrix*
CsrMatrix::gemm(const CooMatrix* lhs) {
    CsrMatrix* lhs_csr = new CsrMatrix(lhs);
    CsrMatrix* answer = lhs_csr->gemm(this);
    delete lhs_csr;
    return answer;
}

void
CsrMatrix::expand() {
    if (mode == CsrMatrix::VERTEX) {
        int* new_rows = (int*) malloc((nve*m+1) * sizeof(int));
        int* new_cols = (int*) malloc(nve*nnz() * sizeof(int));
        float* new_vals = (float*) malloc(nve*nnz() * sizeof(float));

        int new_i = 0;
        for(int r = 0; r < m; r++) {
            for(int k = 0; k < nve; k++) {
                new_rows[r*nve + k] = new_i+1;
                for(int i = rows[r]; i < rows[r+1]; i++, new_i++) {
                    int col_one = cols[i-1];
                    float val = vals[i-1];
                    new_cols[new_i] = ((col_one-1)*nve + k) + 1;
                    new_vals[new_i] = val;
                }
            }
        }

        free(rows);
        free(cols);
        free(vals);

        m = m*nve;
        n = n*nve;
        rows = new_rows;
        cols = new_cols;
        vals = new_vals;
        mode = CsrMatrix::ELEMENT;
        new_rows[m] = new_i+1;
    }
}

void
CsrMatrix::dump(std::string ofilename) {
    FILE* ofile = fopen(ofilename.c_str(), "w");
    assert(ofile != NULL);

    fprintf(ofile, "%%%%MatrixMarket matrix coordinate real general\n");
    fprintf(ofile, "%d %d %d\n", m, n, nnz());

    for(int r = 0; r < m; r++) {
        for(int i = rows[r]; i < rows[r+1]; i++) {
            int col = cols[i-1];
            float val = vals[i-1];
            fprintf(ofile, "%d %d %10.3g\n", r+1, col, val);
        }
    }

    fclose(ofile);
}

CsrMatrix::~CsrMatrix() {
    free(rows);
    free(cols);
    free(vals);
}

OsdMklKernelDispatcher::OsdMklKernelDispatcher( int levels )
    : OsdSpMVKernelDispatcher(levels), StagedOp(NULL), subdiv_operator(NULL)
{ }

OsdMklKernelDispatcher::~OsdMklKernelDispatcher()
{
    if (StagedOp != NULL)
        delete StagedOp;
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
    StagedOp = new CooMatrix(i,j);
}

inline void
OsdMklKernelDispatcher::StageElem(int i, int j, float value)
{
    StagedOp->append_element(i, j, value);
}

void
OsdMklKernelDispatcher::PushMatrix()
{
    /* if no subdiv_operator exists, create one from A */
    if (subdiv_operator == NULL) {
        int nve = _currentVertexBuffer->GetNumElements();
        subdiv_operator = new CsrMatrix(StagedOp, nve);
#if !BENCHMARKING
        printf("PushMatrix set %d-%d\n", subdiv_operator->m, subdiv_operator->n);
#endif
    } else {
        CsrMatrix* new_subdiv_operator = subdiv_operator->gemm(StagedOp);
#if !BENCHMARKING
        printf("PushMatrix mul %d-%d = %d-%d * %d-%d\n",
                (int) new_subdiv_operator->m, (int) new_subdiv_operator->n,
                (int) StagedOp->m, (int) StagedOp->n,
                (int) subdiv_operator->m, (int) subdiv_operator->n);
#endif
        delete subdiv_operator;
        subdiv_operator = new_subdiv_operator;
    }

    /* remove staged matrix */
    delete StagedOp;
    StagedOp = NULL;
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
    if (osdSpMVKernel_DumpSpy_FileName != NULL) {
        subdiv_operator->dump(osdSpMVKernel_DumpSpy_FileName);
    }

    subdiv_operator->expand();
    this->PrintReport();
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
    printf(" nverts=%d", subdiv_operator->nnz());
    printf(" mem=%d", size_in_bytes);
    printf(" sparsity=%f", sparsity_factor);
#else
    printf("Subdiv matrix is %d-by-%d with %f%% nonzeroes, takes %d MB.\n",
        subdiv_operator->m, subdiv_operator->n, sparsity_factor, size_in_bytes / 1024 / 1024);
#endif
}

} // end namespace OPENSUBDIV_VERSION

} // end namespace OpenSubdiv

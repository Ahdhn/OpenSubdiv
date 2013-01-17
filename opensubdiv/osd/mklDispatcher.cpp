#include "../version.h"
#include "../osd/mklDispatcher.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

CpuCooMatrix::CpuCooMatrix(int m, int n) :
    CooMatrix(m, n)
{ }

void
CpuCooMatrix::append_element(int i, int j, float val) {
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
CpuCooMatrix::nnz() const {
    return vals.size();
}

CpuCsrMatrix*
CpuCooMatrix::gemm(CpuCsrMatrix* rhs) {
    CpuCsrMatrix* lhs = new CpuCsrMatrix(this);
    CpuCsrMatrix* answer = lhs->gemm(rhs);
    delete lhs;
    return answer;
}

CpuCsrMatrix::CpuCsrMatrix(int m, int n, int nnz, int nve, mode_t mode) :
    CsrMatrix(m, n, nnz, nve, mode) {
    rows = (int*) malloc((m+1) * sizeof(int));
    cols = (int*) malloc(nnz * sizeof(int));
    vals = (float*) malloc(nnz * sizeof(float));
    rows[m] = nnz+1;
}

CpuCsrMatrix::CpuCsrMatrix(const CpuCooMatrix* StagedOp, int nve, mode_t mode) :
    CsrMatrix(StagedOp, nve, mode) {

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
CpuCsrMatrix::nnz() {
    return this->rows[m];
}

int
CpuCsrMatrix::NumBytes() {
    return nnz()*sizeof(float) + nnz()*sizeof(int) + (m+1)*sizeof(int);
}

double
CpuCsrMatrix::SparsityFactor() {
    return (double) nnz() / (double) (m * n);
}

void
CpuCsrMatrix::spmv(float* d_out, float* d_in) {
    assert(mode == CpuCsrMatrix::ELEMENT);
    mkl_scsrgemv((char*)"N", &m, vals, rows, cols, d_in, d_out);
}

CpuCsrMatrix*
CpuCsrMatrix::gemm(CpuCsrMatrix* rhs) {
    if (rhs->mode != this->mode) {
        rhs->expand();
        this->expand();
    }

    CpuCsrMatrix* A = this;
    CpuCsrMatrix* B = rhs;

    int c_nnz = std::min(A->m*B->n, (int) B->nnz()*7); // XXX: shouldn't this be 4, not 7?
    CpuCsrMatrix* C = new CpuCsrMatrix(A->m, B->n, c_nnz, B->nve, mode);

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

void
CpuCsrMatrix::expand() {
    if (mode == CpuCsrMatrix::VERTEX) {
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
        mode = CpuCsrMatrix::ELEMENT;
        new_rows[m] = new_i+1;
    }
}

void
CpuCsrMatrix::dump(std::string ofilename) {
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

CpuCsrMatrix::~CpuCsrMatrix() {
    free(rows);
    free(cols);
    free(vals);
}


OsdMklKernelDispatcher::OsdMklKernelDispatcher(int levels) :
    OsdSpMVKernelDispatcher<CpuCooMatrix,CpuCsrMatrix,OsdCpuVertexBuffer>(levels)
{ }

static OsdMklKernelDispatcher::OsdKernelDispatcher *
Create(int levels) {
    return new OsdMklKernelDispatcher(levels);
}

void
OsdMklKernelDispatcher::Register() {
    Factory::GetInstance().Register(Create, kMKL);
}

} // end namespace OPENSUBDIV_VERSION

} // end namespace OpenSubdiv

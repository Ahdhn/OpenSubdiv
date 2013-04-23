#include "../version.h"
#include "../osd/mklDispatcher.h"

#include <omp.h>
#include <xmmintrin.h>

char* osdSpMVKernel_DumpSpy_FileName = NULL;

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

CpuCooMatrix::CpuCooMatrix(int m, int n) :
    CooMatrix(m, n)
{ }

void
CpuCooMatrix::append_element(int i, int j, float val) {
#ifdef DEBUG
    assert(0 <= i);
    assert(i < m);
    assert(0 <= j);
    assert(j < n);
#endif

    rows.push_back(i+1); // one-based indexing
    cols.push_back(j+1);
    vals.push_back(val);

    nnz = vals.size();
}

CpuCsrMatrix*
CpuCooMatrix::gemm(CpuCsrMatrix* rhs) {
    CpuCsrMatrix* lhs = new CpuCsrMatrix(this);
    CpuCsrMatrix* answer = lhs->gemm(rhs);
    delete lhs;
    return answer;
}

CpuCsrMatrix::CpuCsrMatrix(int m, int n, int nnz, int nve) :
    CsrMatrix(m, n, nnz, nve) {
    rows = (int*) malloc((m+1) * sizeof(int));
    cols = (int*) malloc(nnz * sizeof(int));
    vals = (float*) malloc(nnz * sizeof(float));
    rows[m] = nnz+1;
}

CpuCsrMatrix::CpuCsrMatrix(const CpuCooMatrix* StagedOp, int nve) :
    CsrMatrix(StagedOp, nve) {

    m = StagedOp->m;
    n = StagedOp->n;
    int numnz = StagedOp->nnz;
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

    nnz = rows[m]-1;
}

void
CpuCsrMatrix::logical_spmv(float* d_out, float* d_in) {
    omp_set_num_threads( omp_get_num_procs() );

    #pragma omp parallel for
    for(int i = 0; i < m; i++) {

        register __m128
            out03v = _mm_setzero_ps(),
            out45v = _mm_setzero_ps();
        int out_idx = 6*i;

        for (int k = rows[i]; k < rows[i+1]; k++) {

            int in_idx = 6*cols[k];

            __m128 ignore,
                   in03v = _mm_loadu_ps( &d_in[in_idx] ),
                   in45v = _mm_loadl_pi( ignore, (const __m64*) &d_in[in_idx+4] ),
                   weightv = _mm_load1_ps( &vals[k] );

            out03v = _mm_add_ps(out03v, _mm_mul_ps(weightv, in03v));
            out45v = _mm_add_ps(out45v, _mm_mul_ps(weightv, in45v));

            _mm_storeu_ps( &d_out[ out_idx ], out03v );
            _mm_storel_pi( (__m64*) &d_out[ out_idx+4 ], out45v );
        }
    }
}

void
CpuCsrMatrix::spmv(float* d_out, float* d_in) {
    char *mkl_transa = (char*) "N";
    int mkl_m = m,
        mkl_n = nve,
        mkl_k = n;
    float mkl_alpha = 1.0f;
    char *mkl_matdesrca = (char*) "G__C__";
    float *mkl_val = vals;
    int *mkl_indx = cols,
        *mkl_pntrb = rows,
        *mkl_pntre = rows+1;
    float *mkl_b = d_in;
    int mkl_ldb = nve;
    float mkl_beta = 0.0f;
    float *mkl_c = d_out;
    int mkl_ldc = nve;

    mkl_scsrmm(mkl_transa, &mkl_m, &mkl_n, &mkl_k, &mkl_alpha,
            mkl_matdesrca, mkl_val, mkl_indx, mkl_pntrb, mkl_pntre,
            mkl_b, &mkl_ldb, &mkl_beta, mkl_c, &mkl_ldc);
}

CpuCsrMatrix*
CpuCsrMatrix::gemm(CpuCsrMatrix* rhs) {

    CpuCsrMatrix* A = this;
    CpuCsrMatrix* B = rhs;
    assert(A->n == B->m);

    int request = 1; // count nonzeroes
    int sort = 7; // don't reorder nonzeroes
    int info = 0; // output info flag

    int c_rows[A->m+1];
    int c_nnz;

    /* count nonzeroes in C */
    mkl_scsrmultcsr((char*)"N", &request, &sort,
            &A->m, &A->n, &B->n,
            A->vals, A->cols, A->rows,
            B->vals, B->cols, B->rows,
            NULL, NULL, &c_rows[0],
            &c_nnz, &info);

    if (info != 0) {
        printf("Error: info returned %d\n", info);
        assert(info == 0);
    }

    c_nnz = c_rows[A->m]-1;
    CpuCsrMatrix* C = new CpuCsrMatrix(A->m, B->n, c_nnz, B->nve);
    memcpy(&C->rows[0], &c_rows[0], (A->m+1)*sizeof(int));

    /* do multiplication  */
    request = 2;
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
OsdMklKernelDispatcher::FinalizeMatrix() {
    this->super::FinalizeMatrix();

    for (int i = 0; i < SubdivOp->m+1; i++)
        SubdivOp->rows[i] -= 1;
    for (int i = 0; i < SubdivOp->nnz; i++)
        SubdivOp->cols[i] -= 1;
}

void
CpuCsrMatrix::dump(std::string ofilename) {
    FILE* ofile = fopen(ofilename.c_str(), "w");
    assert(ofile != NULL);

    fprintf(ofile, "%%%%MatrixMarket matrix coordinate real general\n");
    fprintf(ofile, "%d %d %d\n", m, n, nnz);

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


OsdMklKernelDispatcher::OsdMklKernelDispatcher(int levels, bool logical) :
    super(levels,logical)
{ }

static OsdMklKernelDispatcher::OsdKernelDispatcher *
Create(int levels) {
    return new OsdMklKernelDispatcher(levels, false);
}

static OsdMklKernelDispatcher::OsdKernelDispatcher *
CreateLogical(int levels) {
    return new OsdMklKernelDispatcher(levels, true);
}

void
OsdMklKernelDispatcher::Register() {
    Factory::GetInstance().Register(Create, kMKL);
    Factory::GetInstance().Register(CreateLogical, kCCPU);
}

} // end namespace OPENSUBDIV_VERSION

} // end namespace OpenSubdiv

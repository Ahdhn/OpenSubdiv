#ifndef OSD_CUSPARSE_DISPATCHER_H
#define OSD_CUSPARSE_DISPATCHER_H

#include "../version.h"
#include "../osd/spmvDispatcher.h"
#include "../osd/cudaDispatcher.h"
#include "../osd/mklDispatcher.h"

#include <cusparse_v2.h>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

class CudaCsrMatrix;

class CudaCooMatrix : public CpuCooMatrix {
public:
    CudaCooMatrix(int m, int n) :
        CpuCooMatrix(m, n) { }

    virtual CudaCsrMatrix* gemm(CudaCsrMatrix* rhs);
};

class CudaCsrMatrix : public CsrMatrix {
public:
    CudaCsrMatrix(int m, int n, int nnz=0, int nve=1, mode_t=VERTEX);
    CudaCsrMatrix(const CudaCooMatrix* StagedOp, int nve=1, mode_t=VERTEX);
    virtual ~CudaCsrMatrix();
    void spmv(float* d_out, float* d_in);
    void logical_spmv(float* d_out, float* d_in);
    virtual CudaCsrMatrix* gemm(CudaCsrMatrix* rhs);
    void ellize();
    void dump(std::string ofilename);

    cusparseMatDescr_t desc;
    int* rows;
    int* cols;
    float* vals;

    // ellpack data
    float* ell_vals;
    int* ell_cols;
    int ell_k;

    // COO data
    int coo_nnz;
    float *coo_vals, *coo_scratch;
    int *coo_rows, *coo_cols;

    // scratch space for tranposes of input and output vectors
    float *d_in_scratch, *d_out_scratch;
};

class OsdCusparseKernelDispatcher :
    public OsdSpMVKernelDispatcher<CudaCooMatrix,CudaCsrMatrix,OsdCudaVertexBuffer>
{
public:
    typedef OsdSpMVKernelDispatcher<CudaCooMatrix, CudaCsrMatrix, OsdCudaVertexBuffer> super;
    OsdCusparseKernelDispatcher(int levels, bool logical);
    ~OsdCusparseKernelDispatcher();
    virtual void FinalizeMatrix();
    static void Register();
    void Synchronize();
};

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* OSD_CUSPARSE_DISPATCHER_H */

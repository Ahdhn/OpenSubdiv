#ifndef OSD_HYBRID_DISPATCHER_H
#define OSD_HYBRID_DISPATCHER_H

#include "../version.h"
#include "../osd/spmvDispatcher.h"
#include "../osd/cudaDispatcher.h"
#include "../osd/mklDispatcher.h"
#include "../osd/hybridDispatcher.h"

#include <cusparse_v2.h>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

class HybridCsrMatrix;

class HybridCooMatrix : public CpuCooMatrix {
public:
    HybridCooMatrix(int m, int n) :
        CpuCooMatrix(m, n) { }

    virtual HybridCsrMatrix* gemm(HybridCsrMatrix* rhs);
};

class HybridCsrMatrix : public CsrMatrix {
public:
    HybridCsrMatrix(int m, int n, int nnz=0, int nve=1);
    HybridCsrMatrix(const HybridCooMatrix* StagedOp, int nve=1);
    virtual ~HybridCsrMatrix();
    void spmv(float* d_out, float* d_in);
    void logical_spmv(float* d_out, float* d_in);
    virtual HybridCsrMatrix* gemm(HybridCsrMatrix* rhs);
    virtual int NumBytes();
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

class OsdHybridKernelDispatcher :
    public OsdSpMVKernelDispatcher<HybridCooMatrix,HybridCsrMatrix,OsdCudaVertexBuffer>
{
public:
    typedef OsdSpMVKernelDispatcher<HybridCooMatrix, HybridCsrMatrix, OsdCudaVertexBuffer> super;
    OsdHybridKernelDispatcher(int levels);
    ~OsdHybridKernelDispatcher();
    virtual void FinalizeMatrix();
    static void Register();
    void Synchronize();
};

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* OSD_HYBRID_DISPATCHER_H */

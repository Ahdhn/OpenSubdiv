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

class CudaCooMatrix : public CooMatrix {
public:
    CudaCooMatrix(int m, int n);
    void append_element(int i, int j, float val);
    CudaCsrMatrix* gemm(CudaCsrMatrix* rhs);
    int nnz() const;
};

class CudaCsrMatrix : public CsrMatrix {
public:
    CudaCsrMatrix(int m, int n, int nnz, int nve=1, mode_t=VERTEX);
    CudaCsrMatrix(const CudaCooMatrix* StagedOp, int nve=1, mode_t=VERTEX);
    void spmv(float* d_out, float* d_in);
    CudaCsrMatrix* gemm(CudaCsrMatrix* rhs);
    virtual ~CudaCsrMatrix();
    void expand();
    int nnz();
    void dump(std::string ofilename);

    cusparseMatDescr_t desc;
};

class OsdCusparseKernelDispatcher :
    public OsdSpMVKernelDispatcher<CudaCooMatrix,CudaCsrMatrix,OsdCudaVertexBuffer>
{
public:
    OsdCusparseKernelDispatcher(int levels);
    static void Register();
    void Synchronize();
};

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* OSD_CUSPARSE_DISPATCHER_H */

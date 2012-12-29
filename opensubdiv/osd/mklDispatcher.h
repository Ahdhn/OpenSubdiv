#ifndef OSD_MKL_DISPATCHER_H
#define OSD_MKL_DISPATCHER_H

#include "../version.h"
#include "../osd/spmvDispatcher.h"

extern "C" {
#include <mkl_spblas.h>
}

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

class CooMatrix {
public:
    int m, n;
    std::vector<int> rows;
    std::vector<int> cols;
    std::vector<float> vals;

    CooMatrix(int m, int n);
    void append_element(int i, int j, float val);
    int nnz() const;
};

class CsrMatrix {
public:
    int m, n, nve;
    int* rows;
    int* cols;
    float* vals;

    typedef enum {
        VERTEX, // matrix indices refer to logical vertices
        ELEMENT // matrix indices refer to vertex elements
    } mode_t;

    mode_t mode;

    CsrMatrix(int m, int n, int nnz, int nve=1, mode_t=VERTEX);
    CsrMatrix(const CooMatrix* S, int nve=1, mode_t=VERTEX);
    void spmv(float* d_out, float* d_in);
    CsrMatrix* gemm(CsrMatrix* rhs);
    CsrMatrix* gemm(const CooMatrix* rhs);
    virtual ~CsrMatrix();
    void expand();
    int nnz();
    void dump(std::string ofilename);

    int NumBytes();
    double SparsityFactor();
};

class OsdMklKernelDispatcher : public OsdSpMVKernelDispatcher
{
public:
    OsdMklKernelDispatcher(int levels);
    virtual ~OsdMklKernelDispatcher();

    static void Register();

    virtual void StageMatrix(int i, int j);
    virtual void StageElem(int i, int j, float value);
    virtual void PushMatrix();
    virtual void FinalizeMatrix();
    virtual void ApplyMatrix(int offset);
    virtual bool MatrixReady();
    virtual void PrintReport();

    CooMatrix* S;
    CsrMatrix* subdiv_operator;
};

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* OSD_MKL_DISPATCHER_H */

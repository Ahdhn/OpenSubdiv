#ifndef OSD_MKL_DISPATCHER_H
#define OSD_MKL_DISPATCHER_H

#include "../version.h"
#include "../osd/spmvDispatcher.h"

extern "C" {
#include <mkl_spblas.h>
}

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

class Matrix {

public:
    int m, n, nnz, nve;
    int* rows;
    int* cols;
    float* vals;

    typedef enum {
        VERTEX, // matrix indices refer to logical vertices
        ELEMENT // matrix indices refer to vertex elements
    } mode_t;

    mode_t mode;

    Matrix(int m, int n, int nnz, int nve=1, Matrix::mode_t=VERTEX);
    Matrix(const coo_matrix1& S, int nve=1);
    void spmv(float* d_out, const float* d_in);
    Matrix* operator*(Matrix* rhs);
    Matrix* operator*(const coo_matrix1* rhs);
    virtual ~Matrix();
    void expand();
    void report(std::string name);

    int NumBytes() const;
    double SparsityFactor() const;
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

    coo_matrix1 *S;
    Matrix* subdiv_operator;
};

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* OSD_MKL_DISPATCHER_H */

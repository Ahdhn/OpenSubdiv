#ifndef OSD_CUSPARSE_DISPATCHER_H
#define OSD_CUSPARSE_DISPATCHER_H

#include "../version.h"
#include "../osd/mklDispatcher.h"
#include "../osd/cudaDispatcher.h"

#include <cusparse_v2.h>
#include "boost/numeric/ublas/experimental/sparse_view.hpp"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

class device_csr_matrix_view {
public:
    device_csr_matrix_view(int m, int n, int nnz=0);
    device_csr_matrix_view(csr_matrix1* M);
    void spmv(float* d_out, const float* d_in);
    device_csr_matrix_view* times(device_csr_matrix_view* rhs);
    virtual ~device_csr_matrix_view();
    void report(std::string name);

    int m, n, nnz;
    int* rows;
    int* cols;
    float* vals;

    cusparseMatDescr_t desc;
};

class OsdCusparseVertexBuffer : public OsdCudaVertexBuffer {
public:
    OsdCusparseVertexBuffer(int numElements, int numVertices) :
        OsdCudaVertexBuffer(numElements, numVertices) { }
};

class OsdCusparseKernelDispatcher : public OsdMklKernelDispatcher
{
public:
    OsdCusparseKernelDispatcher(int levels);
    virtual ~OsdCusparseKernelDispatcher();

    static void Register();
    virtual void BindVertexBuffer(OsdVertexBuffer *vertex, OsdVertexBuffer *varying);
    virtual OsdVertexBuffer *InitializeVertexBuffer(int numElements, int numVertices);

    virtual void ApplyMatrix(int offset);
    virtual void PushMatrix();
    virtual void FinalizeMatrix();
    virtual bool MatrixReady();
    virtual void WriteMatrix() { };
    virtual void PrintReport() { };

    device_csr_matrix_view *_deviceMatrix;
    device_csr_matrix_view *_deviceMatrixBig;
};

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* OSD_CUSPARSE_DISPATCHER_H */

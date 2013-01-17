#ifndef OSD_SPMV_DISPATCHER_H
#define OSD_SPMV_DISPATCHER_H

#include "../version.h"
#include "../osd/cpuDispatcher.h"
#include "../osd/spmvKernel.h"

#include <stdio.h>
#include <sstream>
#include <string>

static char* osdSpMVKernel_DumpSpy_FileName = NULL;

#define DEBUG_PRINTF(fmt, ...) \
  fprintf(stderr, "[info] "fmt, ##__VA_ARGS__);

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

template <class CooMatrix_t, class CsrMatrix_t, class VertexBuffer_t>
class OsdSpMVKernelDispatcher : public OsdCpuKernelDispatcher
{
public:
    OsdSpMVKernelDispatcher( int levels )
        : OsdCpuKernelDispatcher(levels), StagedOp(NULL), SubdivOp(NULL)
    { }

    virtual ~OsdSpMVKernelDispatcher() {
        if (_vdesc)
            delete _vdesc;
        if (StagedOp != NULL)
            delete StagedOp;
        if (SubdivOp!= NULL)
            delete SubdivOp;
    }

    virtual void BindVertexBuffer(OsdVertexBuffer *vertex, OsdVertexBuffer *varying) {
        if (vertex)
            _currentVertexBuffer = dynamic_cast<VertexBuffer_t *>(vertex);
        else
            _currentVertexBuffer = NULL;

        if (varying)
            _currentVaryingBuffer = dynamic_cast<VertexBuffer_t *>(varying);
        else
            _currentVaryingBuffer = NULL;

        _vdesc = new SpMVVertexDescriptor(this,
                _currentVertexBuffer  ? _currentVertexBuffer->GetNumElements()  : 0,
                _currentVaryingBuffer ? _currentVaryingBuffer->GetNumElements() : 0);
    }

    virtual VertexBuffer_t* InitializeVertexBuffer(int numElements, int numVertices) {
	    return new VertexBuffer_t(numElements, numVertices);
    }

    int GetElemsPerVertex() const {
        return _currentVertexBuffer ? _currentVertexBuffer->GetNumElements() : 0;
    }

    int GetElemsPerVarying() const {
        return _currentVaryingBuffer ? _currentVaryingBuffer->GetNumElements() : 0;
    }

    int CopyNVerts(int nVerts, int index) {
        for (int i = 0; i < nVerts; i++)
            _vdesc->AddWithWeight(NULL, index+i, index+i, 1.0);
        return nVerts;
    }


    /**
     * Stage an i-by-j matrix with the dispatcher. The matrix will
     * be populated by StageElem calls executed later by the
     * subdivision driver. In pseudocode:
     * S = new matrix(i,j)
     */
    virtual void StageMatrix(int i, int j) {
        StagedOp = new CooMatrix_t(i,j);
    }

    /**
     * Insert an element of the given value at location (i,j) in
     * the staged matrix. In pseudocode:
     * S[i,j] = value
     */
    virtual void StageElem(int i, int j, float value) {
        StagedOp->append_element(i, j, value);
    }

    /**
     * Multiplies the current subdivision matrix by the staged
     * matrix, and unstages it. If there is no current subdivision
     * matrix, the staged matrix becomes the subdivision matrix.
     * In pseudocode:
     * M = S * M
     */
    virtual void PushMatrix() {
        /* if no SubdivOp exists, create one from A */
        if (SubdivOp == NULL) {
            int nve = _currentVertexBuffer->GetNumElements();
            SubdivOp = new CsrMatrix_t(StagedOp, nve);
            DEBUG_PRINTF("PushMatrix set %d-%d\n", SubdivOp->m, SubdivOp->n);
        } else {
            CsrMatrix_t* new_SubdivOp = StagedOp->gemm(SubdivOp);
            DEBUG_PRINTF("PushMatrix mul %d-%d = %d-%d * %d-%d\n",
                    (int) new_SubdivOp->m, (int) new_SubdivOp->n,
                    (int) StagedOp->m, (int) StagedOp->n,
                    (int) SubdivOp->m, (int) SubdivOp->n);
            delete SubdivOp;
            SubdivOp = new_SubdivOp;
        }

        /* remove staged matrix */
        delete StagedOp;
        StagedOp = NULL;
    }

    /**
     * Called after all matrices have been pushed, and before
     * the matrix is applied to the vertices (ApplyMatrix).
     */
    virtual void FinalizeMatrix() {
        if (osdSpMVKernel_DumpSpy_FileName != NULL)
            SubdivOp->dump(osdSpMVKernel_DumpSpy_FileName);

        SubdivOp->expand();
        this->PrintReport();
    }

    /**
     * Apply the subdivison matrix on the vertices at index 0,
     * and store the result at the given offset. In pseudocode:
     * v[offset:...] = M * v[0:...]
     */
    virtual void ApplyMatrix(int offset) {
        int numElems = _currentVertexBuffer->GetNumElements();
        float* V_in = (float*) _currentVertexBuffer->Map();
        float* V_out = (float*) _currentVertexBuffer->Map()
            + offset * numElems;

        SubdivOp->spmv(V_out, V_in);
    }

    /**
     * True if the subdivision matrix has been constructed and is
     * ready to be applied to the vector of vertices.
     */
    virtual bool MatrixReady() {
        return (SubdivOp != NULL);
    }

    /**
     * Print a report on stdout. This is called after the subdivision
     * matrix is constructed and is useful for displaying stats like
     * matrix dimensions, number of nonzeroes, memory usage, etc.
     */
    virtual void PrintReport() {
        int size_in_bytes = SubdivOp->NumBytes();
        double sparsity_factor = 100.0 * SubdivOp->SparsityFactor();

        #if BENCHMARKING
            printf(" nverts=%d", SubdivOp->nnz());
            printf(" mem=%d", size_in_bytes);
            printf(" sparsity=%f", sparsity_factor);
        #endif

        DEBUG_PRINTF("Subdiv matrix is %d-by-%d with %f%% nonzeroes, takes %d MB.\n",
                SubdivOp->m, SubdivOp->n, sparsity_factor, size_in_bytes / 1024 / 1024);
    }

    CooMatrix_t* StagedOp;
    CsrMatrix_t* SubdivOp;
};


class CsrMatrix;

class CooMatrix {
public:
    int m, n;
    CooMatrix(int m, int n) : m(m), n(n) { };

    virtual void append_element(int i, int j, float val) = 0;
    virtual int nnz() const = 0;
};

class CsrMatrix {
public:
    int m, n, nve;

    typedef enum {
        VERTEX, // matrix indices refer to logical vertices
        ELEMENT // matrix indices refer to vertex elements
    } mode_t;

    mode_t mode;

    CsrMatrix(int m, int n, int nnz=1, int nve=1, mode_t mode=VERTEX) :
        m(m), n(n), nve(nve), mode(mode) { };
    CsrMatrix(const CooMatrix* StagedOp, int nve=1, mode_t mode=VERTEX) :
        m(StagedOp->m), n(StagedOp->n), nve(nve), mode(mode) { }

    virtual void spmv(float* d_out, float* d_in) = 0;
    virtual void expand() = 0;
    virtual int nnz() = 0;
    virtual void dump(std::string ofilename) = 0;

    virtual inline int NumBytes() {
        return nnz()*sizeof(float) + nnz()*sizeof(int) + (m+1)*sizeof(int);
    }

    virtual inline double SparsityFactor() {
        return (double) nnz() / (double) (m * n);
    }
};

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* OSD_SPMV_DISPATCHER_H */

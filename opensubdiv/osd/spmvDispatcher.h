#ifndef OSD_SPMV_DISPATCHER_H
#define OSD_SPMV_DISPATCHER_H

#include "../version.h"
#include "../osd/cpuDispatcher.h"
#include "../osd/spmvKernel.h"

#include <stdio.h>
#include <sstream>
#include <string>
#include <vector>

#include <boost/foreach.hpp>
#define foreach BOOST_FOREACH

static char* osdSpMVKernel_DumpSpy_FileName = NULL;
static int osdSpMVKernel_NMultiplications= 5;

#define DEBUG_PRINTF(fmt, ...) \
  fprintf(stderr, "[info] "fmt, ##__VA_ARGS__);

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

template <class CooMatrix_t, class CsrMatrix_t, class VertexBuffer_t>
class OsdSpMVKernelDispatcher : public OsdCpuKernelDispatcher
{
public:
    OsdSpMVKernelDispatcher( int levels )
        : OsdCpuKernelDispatcher(levels), StagedOp(NULL), multiplications(osdSpMVKernel_NMultiplications)
    { }

    virtual ~OsdSpMVKernelDispatcher() {
        if (_vdesc)
            delete _vdesc;
        if (StagedOp != NULL)
            delete StagedOp;
        foreach(CsrMatrix_t* op, SubdivOps)
            delete op;
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
        if (multiplications > 0 && SubdivOps.size() > 0) {
            CsrMatrix_t* new_SubdivOp = StagedOp->gemm(SubdivOps.back());
            DEBUG_PRINTF("PushMatrix mul %d-%d = %d-%d * %d-%d\n",
                    (int) new_SubdivOp->m, (int) new_SubdivOp->n,
                    (int) StagedOp->m, (int) StagedOp->n,
                    (int) SubdivOps.back()->m, (int) SubdivOps.back()->n);
            delete SubdivOps.back();
            SubdivOps.pop_back();
            SubdivOps.push_back(new_SubdivOp);
            multiplications--;
        } else {
            int nve = _currentVertexBuffer->GetNumElements();
            CsrMatrix_t* M = new CsrMatrix_t(StagedOp, nve);
            SubdivOps.push_back(M);
            DEBUG_PRINTF("PushMatrix set %d-%d\n", M->m, M->n);
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
            SubdivOps.back()->dump(osdSpMVKernel_DumpSpy_FileName);

        foreach(CsrMatrix_t* op, SubdivOps)
            op->expand();

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

        foreach(CsrMatrix_t* op, SubdivOps)
            op->spmv(V_out, V_in);
    }

    /**
     * True if the subdivision matrix has been constructed and is
     * ready to be applied to the vector of vertices.
     */
    virtual bool MatrixReady() {
        return (SubdivOps.size() > 0);
    }

    /**
     * Print a report on stdout. This is called after the subdivision
     * matrix is constructed and is useful for displaying stats like
     * matrix dimensions, number of nonzeroes, memory usage, etc.
     */
    virtual void PrintReport() {
        int size_in_bytes = 0;
        foreach(CsrMatrix_t* op, SubdivOps)
            size_in_bytes += op->NumBytes();
        double sparsity_factor = 100.0 * SubdivOps.back()->SparsityFactor();

        #if BENCHMARKING
            printf(" nverts=%d", SubdivOps.back()->nnz());
            printf(" mem=%d", size_in_bytes);
            printf(" sparsity=%f", sparsity_factor);
        #endif

        int opnum = 0;
        foreach(CsrMatrix_t* op, SubdivOps) {
            DEBUG_PRINTF("Subdiv matrix #%d is %d-by-%d with %f%% nonzeroes, takes %d MB.\n",
                    opnum, op->m, op->n, sparsity_factor, size_in_bytes / 1024 / 1024);
            opnum += 1;
        }
    }

    CooMatrix_t* StagedOp;
    std::vector<CsrMatrix_t*> SubdivOps;
    int multiplications;
};

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* OSD_SPMV_DISPATCHER_H */

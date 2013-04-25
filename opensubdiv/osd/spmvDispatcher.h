#ifndef OSD_SPMV_DISPATCHER_H
#define OSD_SPMV_DISPATCHER_H

#include "../version.h"
#include "../osd/cpuDispatcher.h"
#include "../osd/spmvKernel.h"
#include "../../examples/common/stopwatch.h"

#include <stdio.h>
#include <sstream>
#include <string>

extern char* osdSpMVKernel_DumpSpy_FileName;
extern Stopwatch g_matrixTimer;

#ifdef BENCHMARKING
  #define DEBUG_PRINTF(fmt, ...) {};
#else
  #define DEBUG_PRINTF(fmt, ...) \
    fprintf(stderr, "[info] "fmt, ##__VA_ARGS__);
#endif

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

template <class CooMatrix_t, class CsrMatrix_t, class VertexBuffer_t>
class OsdSpMVKernelDispatcher : public OsdCpuKernelDispatcher
{
public:
    OsdSpMVKernelDispatcher( int levels, bool logical=false )
        : OsdCpuKernelDispatcher(levels), logical(logical), StagedOp(NULL), SubdivOp(NULL)
    { }

    virtual ~OsdSpMVKernelDispatcher() {
        if (_vdesc)           delete _vdesc;
        if (StagedOp != NULL) delete StagedOp;
        if (SubdivOp != NULL)  delete SubdivOp;
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

    virtual OsdVertexBuffer* InitializeVertexBuffer(int numElements, int numVertices) {
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

        // NICK you could allocate storage here for the staged_subdiv_operator, or do it on-demand when the first StageEditAdd is called.
        // int nve = _currentVertexBuffer->GetNumElements();
        // staged_subdiv_operator = std::vector<float>(nve*StagedOp->m, 0.0f);
    }

    /**
     * Insert an element of the given value at location (i,j) in
     * the staged matrix. In pseudocode:
     * S[i,j] = value
     */
    virtual void StageElem(int i, int j, float value) {
        StagedOp->append_element(i, j, value);
    }

    // NICK add methods for staging vector and inserting additive edits
    // StageEditAdd(int vert_num, int elem_num, float weight) ->
    //    staged_vec[vert_num*numVertElements + elem_num] = weight;
    //
    // In case there are multiple edits to the same vertex, we might consider doing:
    //    staged_vec[vert_num*numVertElements + elem_num] += weight;

    /**
     * Multiplies the current subdivision matrix by the staged
     * matrix, and unstages it. If there is no current subdivision
     * matrix, the staged matrix becomes the subdivision matrix.
     * In pseudocode:
     * M = S * M
     */
    virtual void PushMatrix() {
        // NICK at end of one level, express edits at next level via mat-vec product
        // based on opensubdiv/far/dispatcher.h:169, edits are applied at the end of a subd level.
        //
        // global_edit_vector: all edits before the current level, but not edits on this level.
        // staged_edit_vector: all edits on the current level
        // SubdivOp is the current multi-level subdivision matrix, not including this level
        // StagedOp is the current single-level subdiv matrix
        //   this routine already combines the matrices into a multi-level matrix via GEMM
        // StagedOp is m-by-n in size (conceptually), but represented in COO format
        //  access with StagedOp->{m,n,rows,cols,vals,nnz}
        //  m: number of rows (aka output vertices)  =>  staged_edit_vector is numVertexElements*m long
        //  n: number of columns (aka input vertices) => global_edit_vector is numVertexElements*n long
        //
        // // pseudocode:
        // if SubdivOp == NULL: // first push
        //   global_edit_vector = staged_edit_vector
        // else:
        //   // express prev level edits at this level:
        //   // StagedOp is in COO representation using 1-based indexing. we need it to be in CSR representation
        //   CsrStagedOp = new CsrMatrix_t(StagedOp, numVertexElements); // uses mkl_scsrcoo under hood
        //   CsrStagedOp->spmv(global_edit_vector, copy(global_edit_vector)); // uses mkl_scsrmm under hood. Computes: g_e_vec = CsrStagedOp * g_e_vec
        //   global_edit_vector += staged_edit_vector // vector-vector add. Could be rolled into previous call with mkl_scsrmm beta param == 1.
        //
        // // CSR and COO formats are described in http://en.wikipedia.org/wiki/Sparse_matrix
        // // ideally we'd like this code to be library-independent, so MKL code would go in MklDispatcher
        // //   That'll make it easier to do a CUDA version too.
        // //   I don't think you'll have to call MKL to get this working! You can just reuse my abstractions is CpuCsrMatrix.
        //
        // clean up (delete CsrStagedOp, etc.)
        //
        // At the end of this routine, global_edit_vector will contain edits from the current and previous levels, expressed at the current level.

        /* if no SubdivOp exists, create one from A */
        if (SubdivOp == NULL) {
            DEBUG_PRINTF("PushMatrix set %d-%d\n", StagedOp->m, StagedOp->n);
            int nve = _currentVertexBuffer->GetNumElements();
            SubdivOp = new CsrMatrix_t(StagedOp, nve);
        } else {
            DEBUG_PRINTF("PushMatrix mul %d-%d = %d-%d * %d-%d\n",
                    (int) StagedOp->m, (int) SubdivOp->n,
                    (int) StagedOp->m, (int) StagedOp->n,
                    (int) SubdivOp->m, (int) SubdivOp->n);
            CsrMatrix_t* new_SubdivOp = StagedOp->gemm(SubdivOp);
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
        float* V_out = (float*) V_in + offset * numElems;

        if (logical)
            SubdivOp->logical_spmv(V_out, V_in);
        else
            SubdivOp->spmv(V_out, V_in);

        // NICK this is the routine that is called every frame
        // V_out += global_edit_vector // vector-vector add (write own or use cblas_saxpy in MKL)
        //
        // it would be cool to memcpy the global_edit_vector into V_out, and use beta=1 in mkl_csrmm to take care of the addition. I'd have to update logical_spmv to support that, but spmv() will work.

        _currentVertexBuffer->Unmap();
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
            printf(" nnz=%d", SubdivOp->nnz);
            printf(" mem=%d", size_in_bytes);
            printf(" sparsity=%f", sparsity_factor);
        #endif

	DEBUG_PRINTF("Subdiv matrix is %d-by-%d with %f%% nonzeroes, takes %d MB.\n",
            SubdivOp->m, SubdivOp->n, sparsity_factor, size_in_bytes / 1024 / 1024);
    }

    /**
     * True if this kernel supports limit surface evaluation. This method exists to
     * avoid no-op calls into the matrix manipulation methods for adhoc kernels.
     */
    virtual int SupportsExactEvaluation() {
        return 1;
    }

    CooMatrix_t* StagedOp;
    CsrMatrix_t* SubdivOp;
    bool logical;
};


class CsrMatrix;

class CooMatrix {
public:
    int m, n, nnz;
    CooMatrix(int m, int n, int nnz=0) : m(m), n(n), nnz(nnz) { };

    virtual void append_element(int i, int j, float val) = 0;
};

class CsrMatrix {
public:
    int m, n, nve, nnz;

    CsrMatrix(int m, int n, int nnz=1, int nve=1) :
        m(m), n(n), nnz(nnz), nve(nve) { };
    CsrMatrix(const CooMatrix* StagedOp, int nve=1) :
        m(StagedOp->m), n(StagedOp->n), nnz(StagedOp->nnz), nve(nve) { }

    virtual void spmv(float* d_out, float* d_in) = 0;
    virtual void logical_spmv(float* d_out, float* d_in) = 0;
    virtual void dump(std::string ofilename) = 0;

    virtual int NumBytes() {
        return nnz*sizeof(float) + nnz*sizeof(int) + (m+1)*sizeof(int);
    }

    virtual inline double SparsityFactor() {
        return (double) nnz / ((double) m * (double) n);
    }
};

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* OSD_SPMV_DISPATCHER_H */

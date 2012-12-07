#ifndef OSD_SPMV_DISPATCHER_H
#define OSD_SPMV_DISPATCHER_H

#include "../version.h"
#include "../osd/cpuDispatcher.h"

#include <stdio.h>
#include <sstream>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>

extern bool osdSpMVKernel_DumpSpy;

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

typedef boost::numeric::ublas::compressed_matrix<
    float,
    boost::numeric::ublas::basic_row_major<int,int>,
    0,
    std::vector<int>,
    std::vector<float>
> csr_matrix;

typedef boost::numeric::ublas::coordinate_matrix<
    float,
    boost::numeric::ublas::basic_row_major<int,int>,
    0,
    boost::numeric::ublas::unbounded_array<int>,
    boost::numeric::ublas::unbounded_array<float>
> coo_matrix;

typedef boost::numeric::ublas::compressed_matrix<
    float,
    boost::numeric::ublas::basic_row_major<int,int>,
    1,
    std::vector<int>,
    std::vector<float>
> csr_matrix1;

typedef boost::numeric::ublas::coordinate_matrix<
    float,
    boost::numeric::ublas::basic_row_major<int,int>,
    1,
    boost::numeric::ublas::unbounded_array<int>,
    boost::numeric::ublas::unbounded_array<float>
> coo_matrix1;

class OsdSpMVKernelDispatcher : public OsdCpuKernelDispatcher
{
public:
    OsdSpMVKernelDispatcher(int levels);
    virtual ~OsdSpMVKernelDispatcher();

    virtual FarMesh<OsdVertex>::Strategy GetStrategy() {
        return FarMesh<OsdVertex>::SpMV;
    }

    virtual void BindVertexBuffer(OsdVertexBuffer *vertex, OsdVertexBuffer *varying);
    int GetElemsPerVertex() const { return _currentVertexBuffer ? _currentVertexBuffer->GetNumElements() : 0; }
    int GetElemsPerVarying() const { return _currentVaryingBuffer ? _currentVaryingBuffer->GetNumElements() : 0; }
    int CopyNVerts(int nVerts, int dstIndex, int srcIndex);

    // static OsdSpMVKernelDispatcher* Create(int levels) = 0;
    // static void Register() = 0;

    /**
     * Stage an i-by-j matrix with the dispatcher. The matrix will
     * be populated by StageElem calls executed later by the
     * subdivision driver. In pseudocode:
     * S = new matrix(i,j)
     */
    virtual void StageMatrix(int i, int j) = 0;

    /**
     * Insert an element of the given value at location (i,j) in
     * the staged matrix. In pseudocode:
     * S[i,j] = value
     */
    virtual void StageElem(int i, int j, float value) = 0;

    /**
     * Multiplies the current subdivision matrix by the staged
     * matrix, and unstages it. If there is no current subdivision
     * matrix, the staged matrix becomes the subdivision matrix.
     * In pseudocode:
     * M = S * M
     */
    virtual void PushMatrix() = 0;

    /**
     * Called after all matrices have been pushed, and before
     * the matrix is applied to the vertices (ApplyMatrix).
     */
    virtual void FinalizeMatrix() = 0;

    /**
     * Apply the subdivison matrix on the vertices at index 0,
     * and store the result at the given offset. In pseudocode:
     * v[offset:...] = M * v[0:...]
     */
    virtual void ApplyMatrix(int offset) = 0;

    /**
     * Writes the subdivison matrix to a file. This step is helpful
     * for visualizing sparsity patterns.
     */
    void WriteMatrix(coo_matrix1* M, std::string file_basename, int id=0);
    void WriteMatrix(csr_matrix1* M, std::string file_basename, int id=0);

    /**
     * True if the subdivision matrix has been constructed and is
     * ready to be applied to the vector of vertices.
     */
    virtual bool MatrixReady() = 0;

    /**
     * Print a report on stdout. This is called after the subdivision
     * matrix is constructed and is useful for displaying stats like
     * matrix dimensions, number of nonzeroes, memory usage, etc.
     */
    virtual void PrintReport() = 0;

    /**
     * Unique ID for each subdivision matrix.
     */
    int matrix_id;
};

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* OSD_SPMV_DISPATCHER_H */

#ifndef OSD_UBLAS_DISPATCHER_H
#define OSD_UBLAS_DISPATCHER_H

#include "../version.h"
#include "../osd/spmvDispatcher.h"

#include <boost/numeric/ublas/operation.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

typedef boost::numeric::ublas::compressed_matrix<
    float,
    basic_row_major<int,int>,
    0,
    unbounded_array<int>,
    unbounded_array<float>
> csr_matrix;

typedef boost::numeric::ublas::coordinate_matrix<
    float,
    basic_row_major<int,int>,
    0,
    unbounded_array<int>,
    unbounded_array<float>
> coo_matrix;


namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

class OsdUBlasKernelDispatcher : public OsdSpMVKernelDispatcher
{
public:
    OsdUBlasKernelDispatcher(int levels);
    virtual ~OsdUBlasKernelDispatcher();

    static void Register();

    virtual void StageMatrix(int i, int j);
    virtual void StageElem(int i, int j, float value);
    virtual void PushMatrix();
    virtual void ApplyMatrix(int offset);
    virtual void WriteMatrix();
    virtual bool MatrixReady();
    virtual void PrintReport();

    coo_matrix *S;
    csr_matrix *M;
};

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* OSD_UBLAS_DISPATCHER_H */

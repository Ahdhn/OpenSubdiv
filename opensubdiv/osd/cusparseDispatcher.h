#ifndef OSD_CUSPARSE_DISPATCHER_H
#define OSD_CUSPARSE_DISPATCHER_H

#include "../version.h"
#include "../osd/spmvDispatcher.h"

#include <boost/numeric/ublas/matrix_sparse.hpp>

typedef boost::numeric::ublas::coordinate_matrix<
    float,
    boost::numeric::ublas::basic_row_major<int,int>,
    0,
    boost::numeric::ublas::unbounded_array<int>,
    boost::numeric::ublas::unbounded_array<float>
> coo_matrix;

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

class OsdCusparseKernelDispatcher : public OsdSpMVKernelDispatcher
{
public:
    OsdCusparseKernelDispatcher(int levels);
    virtual ~OsdCusparseKernelDispatcher();

    static void Register();

    virtual void StageMatrix(int i, int j);
    virtual void StageElem(int i, int j, float value);
    virtual void PushMatrix();
    virtual void ApplyMatrix(int offset);
    virtual void WriteMatrix();
    virtual void FinalizeMatrix();
    virtual bool MatrixReady();
    virtual void PrintReport();

    coo_matrix *S;
};

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* OSD_CUSPARSE_DISPATCHER_H */

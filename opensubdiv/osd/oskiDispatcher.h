#ifndef OSD_OSKI_DISPATCHER_H
#define OSD_OSKI_DISPATCHER_H

#include "../version.h"
#include "../osd/spmvDispatcher.h"

extern "C" {
    #include <oski/oski_Tis.h>
    #include "mmio.h"
}

#include <boost/numeric/ublas/operation.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

using namespace boost::numeric::ublas;

typedef compressed_matrix<
    float,
    basic_row_major<int,int>,
    0,
    unbounded_array<int>,
    unbounded_array<float>
> csr_matrix;

typedef coordinate_matrix<
    float,
    basic_row_major<int,int>,
    0,
    unbounded_array<int>,
    unbounded_array<float>
> coo_matrix;


namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

struct OskiVertexDescriptor;

class OsdOskiKernelDispatcher : public OsdSpMVKernelDispatcher
{
public:
    OsdOskiKernelDispatcher(int levels);
    virtual ~OsdOskiKernelDispatcher();

    virtual FarMesh<OsdVertex>::Strategy GetStrategy() {
        return FarMesh<OsdVertex>::SpMV;
    }

    static void Register();

    virtual void StageMatrix(int i, int j);
    virtual void StageElem(int i, int j, float value);
    virtual void PushMatrix();
    virtual void ApplyM(int offset);
    virtual void WriteM();
    virtual bool MReady();
    virtual void PrintReport();

    csr_matrix *M;
    coo_matrix *S;

    oski_matrix_t A_tunable;
    oski_vecview_t x_view, y_view;
};

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* OSD_OSKI_DISPATCHER_H */

#ifndef OSD_OSKI_DISPATCHER_H
#define OSD_OSKI_DISPATCHER_H

#include "../version.h"
#include "../osd/cpuDispatcher.h"

extern "C" {
    #include <oski/oski_Tis.h>
    #include "mmio.h"
}

#include <boost/numeric/ublas/operation.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

using namespace boost::numeric::ublas;

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

struct OskiVertexDescriptor;

class OsdOskiKernelDispatcher : public OsdCpuKernelDispatcher
{
public:
    OsdOskiKernelDispatcher(int levels);
    virtual ~OsdOskiKernelDispatcher();

    virtual FarMesh<OsdVertex>::Strategy GetStrategy() {
        return FarMesh<OsdVertex>::SpMV;
    }

    static void Register();
    virtual void BindVertexBuffer(OsdVertexBuffer *vertex, OsdVertexBuffer *varying);

    virtual int CopyNVerts(int nVerts, int index = 0, int offset = 0);
    virtual void StageMatrix(int i, int j);
    virtual void StageElem(int i, int j, float value);
    virtual void PushMatrix();
    virtual void ApplyM(int offset);

    virtual void WriteM();

    virtual int GetElemsPerVertex() const { return _currentVertexBuffer ? _currentVertexBuffer->GetNumElements() : NULL; }
    virtual int GetElemsPerVarying() const { return _currentVaryingBuffer ? _currentVaryingBuffer->GetNumElements() : NULL; }

    oski_matrix_t A_tunable;
    oski_vecview_t x_view, y_view;
};

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* OSD_OSKI_DISPATCHER_H */

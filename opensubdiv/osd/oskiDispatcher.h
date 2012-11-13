#ifndef OSD_OSKI_DISPATCHER_H
#define OSD_OSKI_DISPATCHER_H

#include "../version.h"
#include "../osd/ublasDispatcher.h"

extern "C" {
    #include <oski/oski_Tis.h>
}

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

class OsdOskiKernelDispatcher : public OsdUBlasKernelDispatcher
{
public:
    OsdOskiKernelDispatcher(int levels);
    virtual ~OsdOskiKernelDispatcher();

    static void Register();

    virtual void ApplyMatrix(int offset);

    oski_matrix_t A_tunable;
    oski_vecview_t x_view, y_view;
};

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* OSD_OSKI_DISPATCHER_H */

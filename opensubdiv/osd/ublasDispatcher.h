#ifndef OSD_UBLAS_DISPATCHER_H
#define OSD_UBLAS_DISPATCHER_H

#include "../version.h"
#include "../osd/spmvDispatcher.h"

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
    virtual void FinalizeMatrix();
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

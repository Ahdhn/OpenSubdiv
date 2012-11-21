#ifndef OSD_CUSPARSE_DISPATCHER_H
#define OSD_CUSPARSE_DISPATCHER_H

#include "../version.h"
#include "../osd/spmvDispatcher.h"
#include "../osd/cudaDispatcher.h"

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
    virtual void BindVertexBuffer(OsdVertexBuffer *vertex, OsdVertexBuffer *varying);
    virtual void UnbindVertexBuffer();


    virtual void StageMatrix(int i, int j);
    virtual void StageElem(int i, int j, float value);
    virtual void PushMatrix();
    virtual void ApplyMatrix(int offset);
    virtual void WriteMatrix();
    virtual void FinalizeMatrix();
    virtual bool MatrixReady();
    virtual void PrintReport();

    coo_matrix *S;

protected:
    struct DeviceTable
    {
        DeviceTable() : devicePtr(NULL) {}
       ~DeviceTable();

        void Copy(int size, const void *ptr);

        void *devicePtr;
    };

    std::vector<DeviceTable> _tables;
    std::vector<DeviceTable> _editTables;

    OsdCudaVertexBuffer *_currentVertexBuffer;
    float *_deviceVertices;
    int _numVertexElements;
};

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* OSD_CUSPARSE_DISPATCHER_H */

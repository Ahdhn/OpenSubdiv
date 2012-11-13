#include "../version.h"
#include "../osd/mutex.h"
#include "../osd/oskiDispatcher.h"

#include <stdio.h>

using namespace boost::numeric::ublas;

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

OsdOskiKernelDispatcher::OsdOskiKernelDispatcher( int levels )
    : OsdUBlasKernelDispatcher(levels), A_tunable(NULL) {
    oski_Init();
}

OsdOskiKernelDispatcher::~OsdOskiKernelDispatcher() {
    oski_Close();
}

static OsdOskiKernelDispatcher::OsdKernelDispatcher *
Create(int levels) {
    return new OsdOskiKernelDispatcher(levels);
}

void
OsdOskiKernelDispatcher::Register() {
    Factory::GetInstance().Register(Create, kOSKI);
}

void
OsdOskiKernelDispatcher::ApplyMatrix(int offset)
{
    if (A_tunable == NULL) {
        assert(M != NULL);
        assert(_currentVertexBuffer != NULL);

        int numElems = _currentVertexBuffer->GetNumElements();
        float* V_in = _currentVertexBuffer->GetCpuBuffer();
        float* V_out = _currentVertexBuffer->GetCpuBuffer() + offset * numElems;

        x_view = oski_CreateVecView( V_in, M->size2(), STRIDE_UNIT );
        y_view = oski_CreateVecView( V_out, M->size1(), STRIDE_UNIT );

        A_tunable = oski_CreateMatCSR(
                &M->index1_data()[0], // row ptrs
                &M->index2_data()[0], // idx ptrs
                &M->value_data()[0],  // values
                M->size1(),           // num rows
                M->size2(),           // num cols
                COPY_INPUTMAT,        // both use and oski share array
                1,                    // number of args to follow
                INDEX_ZERO_BASED      // zero based indexing
        );

        oski_SetHintMatMult( A_tunable, OP_NORMAL,
                1.0, x_view, 0.0, y_view, ALWAYS_TUNE_AGGRESSIVELY );
        oski_SetHint( A_tunable, HINT_NO_BLOCKS, ARGS_NONE );
        oski_TuneMat( A_tunable );

        //WriteMatrix();
    }

    oski_MatMult( A_tunable, OP_NORMAL, 1.0, x_view, 0.0, y_view );
}

} // end namespace OPENSUBDIV_VERSION

} // end namespace OpenSubdiv

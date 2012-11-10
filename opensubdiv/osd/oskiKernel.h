#ifndef OSD_OSKI_KERNEL_H
#define OSD_OSKI_KERNEL_H

#include "../version.h"
#include "osd/oskiDispatcher.h"
#include "osd/cpuKernel.h"

extern "C" {
    #include <oski/oski_Tis.h>
}

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

class OskiVertexDescriptor : public VertexDescriptor {
public:
    OskiVertexDescriptor(OsdKernelDispatcher* dispatcher, int numVertexElem, int numVaryingElem)
        : VertexDescriptor(numVertexElem, numVaryingElem), _dispatcher(dispatcher) { }

    virtual void Clear(float *vertex, float *varying, int index) const { }

    virtual void AddWithWeight(float *vertex, int dstIndex, int srcIndex, float weight) const {
        //printf("oski AddWithWeight array[%d] 0x%x [%d] <- %f * [%d]\n", numVertexElements, vertex, dstIndex, weight, srcIndex);
        int srcOffset = _dispatcher->srcOffset;
        int d = dstIndex * numVertexElements;
        int s = (srcIndex-srcOffset) * numVertexElements;
        for (int i = 0; i < numVertexElements; ++i)
            _dispatcher->StageElem(d+i,s+i,weight);
    }

    virtual void AddVaryingWithWeight(float *varying, int dstIndex, int srcIndex, float weight) const {
        assert(numVaryingElements == 0);
#if 0
        printf("oski AddVaryingWithWeight array[%d] 0x%x [%d] <- %f * [%d]\n", numVaryingElements, varying, dstIndex, weight, srcIndex);
        int d = dstIndex * numVaryingElements;
        int s = srcIndex * numVaryingElements;
        for (int i = 0; i < numVaryingElements; ++i)
            _dispatcher->StageElem(d+i,s+i,weight);
#endif
    }

    virtual void ApplyVertexEditAdd(float *vertex, int primVarOffset, int primVarWidth, int editIndex, const float *editValues) const {
        printf("Warning: spmv kernels don't support vertex add yet.\n");
        return;
        int d = editIndex * numVertexElements + primVarOffset;
        for (int i = 0; i < primVarWidth; ++i) {
            vertex[d++] += editValues[i];
        }
    }

    virtual void ApplyVertexEditSet(float *vertex, int primVarOffset, int primVarWidth, int editIndex, const float *editValues) const {
        printf("Warning: spmv kernels don't support vertex set.\n");
        return;
        int d = editIndex * numVertexElements + primVarOffset;
        for (int i = 0; i < primVarWidth; ++i) {
            vertex[d++] = editValues[i];
        }
    }

    OsdKernelDispatcher* _dispatcher;
};

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* OSD_OSKI_KERNEL_H */

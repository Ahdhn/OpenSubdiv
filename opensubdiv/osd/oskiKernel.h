#ifndef OSD_OSKI_KERNEL_H
#define OSD_OSKI_KERNEL_H

#include "../version.h"
#include "osd/oskiDispatcher.h"

#include <oski/oski_Tls.h>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

struct OskiVertexDescriptor {

    OskiVertexDescriptor(OsdKernelDispatcher* dispatcher, int numVertexElem, int numVaryingElem)
        : _dispatcher(dispatcher),  numVertexElements(numVertexElem), numVaryingElements(numVaryingElem) { }

    void Clear(float *vertex, float *varying, int index) const {
#if 0
        if (vertex) {
            for (int i = 0; i < numVertexElements; ++i)
                vertex[index*numVertexElements+i] = 0.0f;
        }

        if (varying) {
            for (int i = 0; i < numVaryingElements; ++i)
                varying[index*numVaryingElements+i] = 0.0f;
        }
#endif
    }

    void AddWithWeight(float *vertex, int dstIndex, int srcIndex, float weight) const {
        //printf("oski AddWithWeight array[%d] 0x%x [%d] <- %f * [%d]\n", numVertexElements, vertex, dstIndex, weight, srcIndex);
        int d = dstIndex * numVertexElements;
        int s = srcIndex * numVertexElements;
        for (int i = 0; i < numVertexElements; ++i)
            _dispatcher->StageElem(d+i,s+i,weight);
    }

    void AddVaryingWithWeight(float *varying, int dstIndex, int srcIndex, float weight) const {
        assert(numVaryingElements == 0);
#if 0
        printf("oski AddVaryingWithWeight array[%d] 0x%x [%d] <- %f * [%d]\n", numVaryingElements, varying, dstIndex, weight, srcIndex);
        int d = dstIndex * numVaryingElements;
        int s = srcIndex * numVaryingElements;
        for (int i = 0; i < numVaryingElements; ++i)
            _dispatcher->StageElem(d+i,s+i,weight);
#endif
    }

    void ApplyVertexEditAdd(float *vertex, int primVarOffset, int primVarWidth, int editIndex, const float *editValues) const {
        int d = editIndex * numVertexElements + primVarOffset;
        for (int i = 0; i < primVarWidth; ++i) {
            vertex[d++] += editValues[i];
        }
    }

    void ApplyVertexEditSet(float *vertex, int primVarOffset, int primVarWidth, int editIndex, const float *editValues) const {
        int d = editIndex * numVertexElements + primVarOffset;
        for (int i = 0; i < primVarWidth; ++i) {
            vertex[d++] = editValues[i];
        }
    }

    int numVertexElements;
    int numVaryingElements;
    OsdKernelDispatcher* _dispatcher;
};

extern "C" {

void oskiComputeFace(const OskiVertexDescriptor *vdesc, float * vertex, float * varying, const int *F_IT, const int *F_ITa, int dstOffset, int start, int end);

void oskiComputeEdge(const OskiVertexDescriptor *vdesc, float *vertex, float * varying, const int *E_IT, const float *E_W, int dstOffset, int start, int end);

void oskiComputeVertexA(const OskiVertexDescriptor *vdesc, float *vertex, float * varying, const int *V_ITa, const float *V_W, int dstOffset, int start, int end, int pass);

void oskiComputeVertexB(const OskiVertexDescriptor *vdesc, float *vertex, float * varying, const int *V_ITa, const int *V_IT, const float *V_W, int dstOffset, int start, int end);

void oskiComputeLoopVertexB(const OskiVertexDescriptor *vdesc, float *vertex, float * varying, const int *V_ITa, const int *V_IT, const float *V_W, int dstOffset, int start, int end);

void oskiComputeBilinearEdge(const OskiVertexDescriptor *vdesc, float *vertex, float * varying, const int *E_IT, int dstOffset, int start, int end);

void oskiComputeBilinearVertex(const OskiVertexDescriptor *vdesc, float *vertex, float * varying, const int *V_ITa, int dstOffset, int start, int end);

void oskiEditVertexAdd(const OskiVertexDescriptor *vdesc, float *vertex, int primVarOffset, int primVarWidth, int count, const int *editIndices, const float *editValues);

void oskiEditVertexSet(const OskiVertexDescriptor *vdesc, float *vertex, int primVarOffset, int primVarWidth, int count, const int *editIndices, const float *editValues);

}

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* OSD_OSKI_KERNEL_H */

#ifndef OSD_POSKI_KERNEL_H
#define OSD_POSKI_KERNEL_H

#include "../version.h"

#include <oski/oski.h>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

struct VertexDescriptor {

    VertexDescriptor(int numVertexElem, int numVaryingElem)
        : numVertexElements(numVertexElem), numVaryingElements(numVaryingElem) { }

    void Clear(float *vertex, float *varying, int index) const {
        if (vertex) {
            for (int i = 0; i < numVertexElements; ++i)
                vertex[index*numVertexElements+i] = 0.0f;
        }

        if (varying) {
            for (int i = 0; i < numVaryingElements; ++i)
                varying[index*numVaryingElements+i] = 0.0f;
        }
    }
    void AddWithWeight(float *vertex, int dstIndex, int srcIndex, float weight) const {
        int d = dstIndex * numVertexElements;
        int s = srcIndex * numVertexElements;
        for (int i = 0; i < numVertexElements; ++i)
            vertex[d++] += vertex[s++] * weight;
    }
    void AddVaryingWithWeight(float *varying, int dstIndex, int srcIndex, float weight) const {
        int d = dstIndex * numVaryingElements;
        int s = srcIndex * numVaryingElements;
        for (int i = 0; i < numVaryingElements; ++i)
            varying[d++] += varying[s++] * weight;
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
};

extern "C" {

void computeFace(const VertexDescriptor *vdesc, float * vertex, float * varying, const int *F_IT, const int *F_ITa, int offset, int start, int end);

void computeEdge(const VertexDescriptor *vdesc, float *vertex, float * varying, const int *E_IT, const float *E_W, int offset, int start, int end);

void computeVertexA(const VertexDescriptor *vdesc, float *vertex, float * varying, const int *V_ITa, const float *V_W, int offset, int start, int end, int pass);

void computeVertexB(const VertexDescriptor *vdesc, float *vertex, float * varying, const int *V_ITa, const int *V_IT, const float *V_W, int offset, int start, int end);

void computeLoopVertexB(const VertexDescriptor *vdesc, float *vertex, float * varying, const int *V_ITa, const int *V_IT, const float *V_W, int offset, int start, int end);

void computeBilinearEdge(const VertexDescriptor *vdesc, float *vertex, float * varying, const int *E_IT, int offset, int start, int end);

void computeBilinearVertex(const VertexDescriptor *vdesc, float *vertex, float * varying, const int *V_ITa, int offset, int start, int end);

void editVertexAdd(const VertexDescriptor *vdesc, float *vertex, int primVarOffset, int primVarWidth, int count, const int *editIndices, const float *editValues);

void editVertexSet(const VertexDescriptor *vdesc, float *vertex, int primVarOffset, int primVarWidth, int count, const int *editIndices, const float *editValues);

}

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* OSD_POSKI_KERNEL_H */

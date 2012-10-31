#include <math.h>

#include "../version.h"
#include "../osd/oskiKernel.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

void oskiComputeFace( const OskiVertexDescriptor *vdesc, float * vertex, float * varying, const int *F_IT, const int *F_ITa, int offset, int start, int end) {

    for (int i = start; i < end; i++) {
        int h = F_ITa[2*i];
        int n = F_ITa[2*i+1];

        float weight = 1.0f/n;

        // XXX: should use local vertex struct variable instead of accumulating directly into global memory.
        int dstIndex = offset + i;
        vdesc->Clear(vertex, varying, dstIndex);

        for (int j=0; j<n; ++j) {
            int index = F_IT[h+j];
            vdesc->AddWithWeight(vertex, dstIndex, index, weight);
            vdesc->AddVaryingWithWeight(varying, dstIndex, index, weight);
        }
    }
}

void oskiComputeEdge( const OskiVertexDescriptor *vdesc, float *vertex, float *varying, const int *E_IT, const float *E_W, int offset, int start, int end) {

    for (int i = start; i < end; i++) {
        int eidx0 = E_IT[4*i+0];
        int eidx1 = E_IT[4*i+1];
        int eidx2 = E_IT[4*i+2];
        int eidx3 = E_IT[4*i+3];

        float vertWeight = E_W[i*2+0];

        int dstIndex = offset + i;
        vdesc->Clear(vertex, varying, dstIndex);

        vdesc->AddWithWeight(vertex, dstIndex, eidx0, vertWeight);
        vdesc->AddWithWeight(vertex, dstIndex, eidx1, vertWeight);

        if (eidx2 != -1) {
            float faceWeight = E_W[i*2+1];

            vdesc->AddWithWeight(vertex, dstIndex, eidx2, faceWeight);
            vdesc->AddWithWeight(vertex, dstIndex, eidx3, faceWeight);
        }

        vdesc->AddVaryingWithWeight(varying, dstIndex, eidx0, 0.5f);
        vdesc->AddVaryingWithWeight(varying, dstIndex, eidx1, 0.5f);
    }
}

void oskiComputeVertexA(const OskiVertexDescriptor *vdesc, float *vertex, float *varying, const int *V_ITa, const float *V_W, int offset, int start, int end, int pass) {

    for (int i = start; i < end; i++) {
        int n     = V_ITa[5*i+1];
        int p     = V_ITa[5*i+2];
        int eidx0 = V_ITa[5*i+3];
        int eidx1 = V_ITa[5*i+4];

        float weight = (pass==1) ? V_W[i] : 1.0f - V_W[i];

        // In the case of fractional weight, the weight must be inverted since
        // the value is shared with the k_Smooth kernel (statistically the
        // k_Smooth kernel runs much more often than this one)
        if (weight>0.0f && weight<1.0f && n > 0)
            weight=1.0f-weight;

        int dstIndex = offset + i;
        if(not pass)
            vdesc->Clear(vertex, varying, dstIndex);

        if (eidx0==-1 || (pass==0 && (n==-1)) ) {
            vdesc->AddWithWeight(vertex, dstIndex, p, weight);
        } else {
            vdesc->AddWithWeight(vertex, dstIndex, p, weight * 0.75f);
            vdesc->AddWithWeight(vertex, dstIndex, eidx0, weight * 0.125f);
            vdesc->AddWithWeight(vertex, dstIndex, eidx1, weight * 0.125f);
        }

        if (not pass)
            vdesc->AddVaryingWithWeight(varying, dstIndex, p, 1.0f);
    }
}

void oskiComputeVertexB(const OskiVertexDescriptor *vdesc, float *vertex, float *varying, const int *V_ITa, const int *V_IT, const float *V_W, int offset, int start, int end) {

    for (int i = start; i < end; i++) {
        int h = V_ITa[5*i];
        int n = V_ITa[5*i+1];
        int p = V_ITa[5*i+2];

        float weight = V_W[i];
        float wp = 1.0f/float(n*n);
        float wv = (n-2.0f) * n * wp;

        int dstIndex = offset + i;
        vdesc->Clear(vertex, varying, dstIndex);

        vdesc->AddWithWeight(vertex, dstIndex, p, weight * wv);

        for (int j = 0; j < n; ++j) {
            vdesc->AddWithWeight(vertex, dstIndex, V_IT[h+j*2], weight * wp);
            vdesc->AddWithWeight(vertex, dstIndex, V_IT[h+j*2+1], weight * wp);
        }
        vdesc->AddVaryingWithWeight(varying, dstIndex, p, 1.0f);
    }
}

void oskiComputeLoopVertexB(const OskiVertexDescriptor *vdesc, float *vertex, float *varying, const int *V_ITa, const int *V_IT, const float *V_W, int offset, int start, int end) {

    for (int i = start; i < end; i++) {
        int h = V_ITa[5*i];
        int n = V_ITa[5*i+1];
        int p = V_ITa[5*i+2];

        float weight = V_W[i];
        float wp = 1.0f/float(n);
        float beta = 0.25f * cosf(float(M_PI) * 2.0f * wp) + 0.375f;
        beta = beta * beta;
        beta = (0.625f - beta) * wp;

        int dstIndex = offset + i;
        vdesc->Clear(vertex, varying, dstIndex);

        vdesc->AddWithWeight(vertex, dstIndex, p, weight * (1.0f - (beta * n)));

        for (int j = 0; j < n; ++j)
            vdesc->AddWithWeight(vertex, dstIndex, V_IT[h+j], weight * beta);

        vdesc->AddVaryingWithWeight(varying, dstIndex, p, 1.0f);
    }
}

void oskiComputeBilinearEdge(const OskiVertexDescriptor *vdesc, float *vertex, float *varying, const int *E_IT, int offset, int start, int end) {

    for (int i = start; i < end; i++) {
        int eidx0 = E_IT[2*i+0];
        int eidx1 = E_IT[2*i+1];

        int dstIndex = offset + i;
        vdesc->Clear(vertex, varying, dstIndex);

        vdesc->AddWithWeight(vertex, dstIndex, eidx0, 0.5f);
        vdesc->AddWithWeight(vertex, dstIndex, eidx1, 0.5f);

        vdesc->AddVaryingWithWeight(varying, dstIndex, eidx0, 0.5f);
        vdesc->AddVaryingWithWeight(varying, dstIndex, eidx1, 0.5f);
    }
}

void oskiComputeBilinearVertex(const OskiVertexDescriptor *vdesc, float *vertex, float *varying, const int *V_ITa, int offset, int start, int end) {

    for (int i = start; i < end; i++) {
        int p = V_ITa[i];

        int dstIndex = offset + i;
        vdesc->Clear(vertex, varying, dstIndex);

        vdesc->AddWithWeight(vertex, dstIndex, p, 1.0f);
        vdesc->AddVaryingWithWeight(varying, dstIndex, p, 1.0f);
    }
}

void oskiEditVertexAdd(const OskiVertexDescriptor *vdesc, float *vertex, int primVarOffset, int primVarWidth, int vertexCount, const int *editIndices, const float *editValues) {

    for (int i = 0; i < vertexCount; i++) {
        vdesc->ApplyVertexEditAdd(vertex, primVarOffset, primVarWidth, editIndices[i], &editValues[i*primVarWidth]);
    }
}

void oskiEditVertexSet(const OskiVertexDescriptor *vdesc, float *vertex, int primVarOffset, int primVarWidth, int vertexCount, const int *editIndices, const float *editValues) {

    for (int i = 0; i < vertexCount; i++) {
        vdesc->ApplyVertexEditSet(vertex, primVarOffset, primVarWidth, editIndices[i], &editValues[i*primVarWidth]);
    }
}

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv

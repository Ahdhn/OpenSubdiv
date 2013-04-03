//
//     Copyright (C) Pixar. All rights reserved.
//
//     This license governs use of the accompanying software. If you
//     use the software, you accept this license. If you do not accept
//     the license, do not use the software.
//
//     1. Definitions
//     The terms "reproduce," "reproduction," "derivative works," and
//     "distribution" have the same meaning here as under U.S.
//     copyright law.  A "contribution" is the original software, or
//     any additions or changes to the software.
//     A "contributor" is any person or entity that distributes its
//     contribution under this license.
//     "Licensed patents" are a contributor's patent claims that read
//     directly on its contribution.
//
//     2. Grant of Rights
//     (A) Copyright Grant- Subject to the terms of this license,
//     including the license conditions and limitations in section 3,
//     each contributor grants you a non-exclusive, worldwide,
//     royalty-free copyright license to reproduce its contribution,
//     prepare derivative works of its contribution, and distribute
//     its contribution or any derivative works that you create.
//     (B) Patent Grant- Subject to the terms of this license,
//     including the license conditions and limitations in section 3,
//     each contributor grants you a non-exclusive, worldwide,
//     royalty-free license under its licensed patents to make, have
//     made, use, sell, offer for sale, import, and/or otherwise
//     dispose of its contribution in the software or derivative works
//     of the contribution in the software.
//
//     3. Conditions and Limitations
//     (A) No Trademark License- This license does not grant you
//     rights to use any contributor's name, logo, or trademarks.
//     (B) If you bring a patent claim against any contributor over
//     patents that you claim are infringed by the software, your
//     patent license from such contributor to the software ends
//     automatically.
//     (C) If you distribute any portion of the software, you must
//     retain all copyright, patent, trademark, and attribution
//     notices that are present in the software.
//     (D) If you distribute any portion of the software in source
//     code form, you may do so only under this license by including a
//     complete copy of this license with your distribution. If you
//     distribute any portion of the software in compiled or object
//     code form, you may only do so under a license that complies
//     with this license.
//     (E) The software is licensed "as-is." You bear the risk of
//     using it. The contributors give no express warranties,
//     guarantees or conditions. You may have additional consumer
//     rights under your local laws which this license cannot change.
//     To the extent permitted under your local laws, the contributors
//     exclude the implied warranties of merchantability, fitness for
//     a particular purpose and non-infringement.
//
#ifndef FAR_SUBDIVISION_TABLES_H
#define FAR_SUBDIVISION_TABLES_H

#include <cassert>
#include <utility>
#include <vector>

#include "../version.h"
#include "../far/table.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

struct EigenStruct { const double *L, *iV, *x[3]; };


template <class U> class FarMesh;
template <class U> class FarDispatcher;

/// \brief FarSubdivisionTables are a serialized topological data representation.
///
/// Subdivision tables store the indexing tables required in order to compute
/// the refined positions of a mesh without the help of a hierarchical data
/// structure. The advantage of this representation is its ability to be executed
/// in a massively parallel environment without data dependencies.
///
/// The vertex indexing tables require the vertex buffer to be sorted based on the
/// nature of the parent of a given vertex : either a face, an edge, or a vertex.
/// (note : the Loop subdivision scheme does not create vertices as a child of a
/// face).
///
/// Each type of vertex in the buffer is associated the following tables :
/// - _<T>_IT : indices of all the adjacent vertices required by the compute kernels
/// - _<T>_W : fractional weight of the vertex (based on sharpness & topology)
/// - _<T>_ITa : codex for the two previous tables
/// (where T denotes a face-vertex / edge-vertex / vertex-vertex)
///
///
/// Because each subdivision scheme (Catmark / Loop / Bilinear) introduces variations
/// in the subdivision rules, a derived class specialization is associated with
/// each scheme.
///
/// For more details see : "Feature Adaptive GPU Rendering of Catmull-Clark
/// Subdivision Surfaces"  (p.3 - par. 3.2)
///
template <class U> class FarSubdivisionTables {
public:

    /// Destructor
    FarSubdivisionTables<U>() : eigen(NULL) {}
    virtual ~FarSubdivisionTables<U>();

    /// Return the highest level of subdivision possible with these tables
    int GetMaxLevel() const { return (int)(_vertsOffsets.size()); }

    /// Memory required to store the indexing tables
    virtual int GetMemoryUsed() const;

    /// Compute the positions of refined vertices using the specified kernels
    virtual void Apply( int level, void * clientdata=0 ) const=0;
    virtual void PushToLimitSurface( int level, void * clientdata=0 );
    virtual void PushLimitMatrix(int nverts, int offset) = 0;

    /// Pointer back to the mesh owning the table
    FarMesh<U> * GetMesh() { return _mesh; }

    /// The index of the first vertex that belongs to the level of subdivision
    /// represented by this set of FarCatmarkSubdivisionTables
    int GetFirstVertexOffset( int level ) const;

    /// Number of vertices children of a face at a given level (always 0 for Loop)
    int GetNumFaceVertices( int level ) const;

    /// Number of vertices children of an edge at a given level
    int GetNumEdgeVertices( int level ) const;

    /// Number of vertices children of a vertex at a given level
    int GetNumVertexVertices( int level ) const;

    // Total number of vertices at a given level
    int GetNumVertices( int level ) const;

    /// Indexing tables accessors

    /// Returns the edge vertices indexing table
    FarTable<int> const &          Get_E_IT() const { return _E_IT; }

    /// Returns the edge vertices weights table
    FarTable<float> const &        Get_E_W() const { return _E_W; }

    /// Returns the vertex vertices codex table
    FarTable<int> const &          Get_V_ITa() const { return _V_ITa; }

    /// Returns the vertex vertices indexing table
    FarTable<unsigned int> const & Get_V_IT() const { return _V_IT; }

    /// Returns the vertex vertices weights table
    FarTable<float> const &        Get_V_W() const { return _V_W; }

    /// Returns the number of indexing tables needed to represent this particular
    /// subdivision scheme.
    virtual int GetNumTables() const { return 5; }

public:
    template <class X, class Y> friend class FarMeshFactory;

    FarSubdivisionTables<U>( FarMesh<U> * mesh, int maxlevel );

    // Returns an integer based on the order in which the kernels are applied
    static int getMaskRanking( unsigned char mask0, unsigned char mask1 );

    struct VertexKernelBatch {
        int kernelF; // number of face vertices
        int kernelE; // number of edge vertices

        std::pair<int,int> kernelB;  // first / last vertex vertex batch (kernel B)
        std::pair<int,int> kernelA1; // first / last vertex vertex batch (kernel A pass 1)
        std::pair<int,int> kernelA2; // first / last vertex vertex batch (kernel A pass 2)

        VertexKernelBatch() : kernelF(0), kernelE(0) { }

        void InitVertexKernels(int a, int b) {
            kernelB.first = kernelA1.first = kernelA2.first = a;
            kernelB.second = kernelA1.second = kernelA2.second = b;
        }

        void AddVertex( int index, int rank ) {
            // expand the range of kernel batches based on vertex index and rank
            if (rank<7) {
                if (index < kernelB.first)
                    kernelB.first=index;
                if (index > kernelB.second)
                    kernelB.second=index;
            }
            if ((rank>2) and (rank<8)) {
                if (index < kernelA2.first)
                    kernelA2.first=index;
                if (index > kernelA2.second)
                    kernelA2.second=index;
            }
            if (rank>6) {
                if (index < kernelA1.first)
                    kernelA1.first=index;
                if (index > kernelA1.second)
                    kernelA1.second=index;
            }
        }
    };

    // Returns the range of vertex indices of each of the 3 batches of VertexPoint
    // compute Kernels (kernel application order is : B / A / A)
    std::vector<VertexKernelBatch> & getKernelBatches() const { return _batches; }

    /* for limit surface evaluation */
    void read_eval ( char* filename );
    double *ccdata;
    EigenStruct *eigen;

protected:
    // mesh that owns this subdivisionTable
    FarMesh<U> * _mesh;

    FarTable<int>          _E_IT;  // vertices from edge refinement
    FarTable<float>        _E_W;   // weigths

    FarTable<int>          _V_ITa; // vertices from vertex refinement
    FarTable<unsigned int> _V_IT;  // indices of adjacent vertices
    FarTable<float>        _V_W;   // weights

    std::vector<VertexKernelBatch> _batches; // batches of vertices for kernel execution

    std::vector<int> _vertsOffsets; // offset to the first vertex of each level
};

template <class U>
FarSubdivisionTables<U>::FarSubdivisionTables( FarMesh<U> * mesh, int maxlevel ) :
    _mesh(mesh),
    _E_IT(maxlevel+1),
    _E_W(maxlevel+1),
    _V_ITa(maxlevel+1),
    _V_IT(maxlevel+1),
    _V_W(maxlevel+1),
    _batches(maxlevel),
    _vertsOffsets(maxlevel+1,0),
    ccdata(NULL),
    eigen(NULL)
{
    assert( maxlevel > 0 );
}

// The ranking matrix defines the order of execution for the various combinations
// of Corner, Crease, Dart and Smooth topological configurations. This matrix is
// somewhat arbitrary as it is possible to perform some permutations in the
// ordering without adverse effects, but it does try to minimize kernel switching
// during the exececution of Apply(). This table is identical for both the Loop
// and Catmull-Clark schemes.
//
// The matrix is derived from this table :
// Rules     +----+----+----+----+----+----+----+----+----+----+
//   Pass 0  | Dt | Sm | Sm | Dt | Sm | Dt | Sm | Cr | Co | Cr |
//   Pass 1  |    |    |    | Co | Co | Cr | Cr | Co |    |    |
// Kernel    +----+----+----+----+----+----+----+----+----+----+
//   Pass 0  | B  | B  | B  | B  | B  | B  | B  | A  | A  | A  |
//   Pass 1  |    |    |    | A  | A  | A  | A  | A  |    |    |
//           +----+----+----+----+----+----+----+----+----+----+
// Rank      | 0  | 1  | 2  | 3  | 4  | 5  | 6  | 7  | 8  | 9  |
//           +----+----+----+----+----+----+----+----+----+----+
// with :
//     - A : compute kernel applying k_Crease / k_Corner rules
//     - B : compute kernel applying k_Smooth / k_Dart rules
template <class U> int
FarSubdivisionTables<U>::getMaskRanking( unsigned char mask0, unsigned char mask1 ) {
    static short masks[4][4] = { {    0,    1,    6,    4 },
                                 { 0xFF,    2,    5,    3 },
                                 { 0xFF, 0xFF,    9,    7 },
                                 { 0xFF, 0xFF, 0xFF,    8 } };
    return masks[mask0][mask1];
}

template <class U> int
FarSubdivisionTables<U>::GetFirstVertexOffset( int level ) const {
    assert(level>=0 and level<=(int)_vertsOffsets.size());
    return _vertsOffsets[level];
}

template <class U> int
FarSubdivisionTables<U>::GetNumFaceVertices( int level ) const {
    assert(level>=0 and level<=(int)_batches.size());
    return _batches[level-1].kernelF;
}

template <class U> int
FarSubdivisionTables<U>::GetNumEdgeVertices( int level ) const {
    assert(level>=0 and level<=(int)_batches.size());
    return _batches[level-1].kernelE;
}

template <class U> int
FarSubdivisionTables<U>::GetNumVertexVertices( int level ) const {
    assert(level>=0 and level<=(int)_batches.size());
    if (level==0)
        return _mesh->GetNumCoarseVertices();
    else
        return std::max( _batches[level-1].kernelB.second,
                   std::max(_batches[level-1].kernelA1.second,
                       _batches[level-1].kernelA2.second));
}

template <class U> int
FarSubdivisionTables<U>::GetNumVertices( int level ) const {
    assert(level>=0 and level<=(int)_batches.size());
    if (level==0)
        return GetNumVertexVertices(0);
    else
        return GetNumFaceVertices(level)+
               GetNumEdgeVertices(level)+
               GetNumVertexVertices(level);
}

template <class U> int
FarSubdivisionTables<U>::GetMemoryUsed() const {
    return _E_IT.GetMemoryUsed()+
           _E_W.GetMemoryUsed()+
           _V_ITa.GetMemoryUsed()+
           _V_IT.GetMemoryUsed()+
           _V_W.GetMemoryUsed();
}

template <class U> void
FarSubdivisionTables<U>::PushToLimitSurface( int level, void * clientdata ) {

    int nverts = this->GetNumVertices( level );
    int offset = this->GetFirstVertexOffset( level );

    /* Build and push projection matrix */
    this->PushLimitMatrix(nverts, offset);
}

template <class U>
void
FarSubdivisionTables<U>::read_eval ( char * filename )
{
    // code from https://svn.blender.org/svnroot/bf-blender/branches/ndof/extern/qdune/primitives/CCSubdivision.cpp
    FILE * f = fopen(filename, "rb");
    if (f == NULL) {
        printf("Could not load subdivision data!\n");
        exit(1);
    }

    int Nmax;
    fread (&Nmax, sizeof(int), 1, f);
    // expecting Nmax==50
    if (Nmax != 50) { // should never happen
        printf("[ERROR] -> JS_SDPatch::getCCData(): Unexpected value for Nmax in subdivision data -> %d\n", Nmax);
        exit(1);
    }
    int totdoubles = 0;
    for (int i=0; i<Nmax-2; i++) {
        const int N = i+3;
        const int K = 2*N + 8;
        totdoubles += K + K*K + 3*K*16;
    }
    ccdata = new double[totdoubles];
    fread(ccdata, sizeof(double), totdoubles, f);
    fclose(f);

    // now set the actual EigenStructs as pointers to data in array
    eigen = new EigenStruct[48];
    int ofs1 = 0;
    for (int i=0; i<48; i++) {
        const int K = 2*(i + 3) + 8;
        const int ofs2 = ofs1 + K;
        const int ofs3 = ofs2 + K*K;
        const int ofs4 = ofs3 + K*16;
        const int ofs5 = ofs4 + K*16;
        eigen[i].L = ccdata + ofs1;
        eigen[i].iV = ccdata + ofs2;
        eigen[i].x[0] = ccdata + ofs3;
        eigen[i].x[1] = ccdata + ofs4;
        eigen[i].x[2] = ccdata + ofs5;
        ofs1 = ofs5 + K*16;
    }

    // make bspline evaluation basis
    double buv[16][16];
    memset(buv, 0, sizeof(double)*16*16);
    // bspline basis (could use RiBSplineBasis, but want double prec by default)
    double bsp[4][4] = {{-1.0/6.0,     0.5,    -0.5, 1.0/6.0},
        {     0.5,    -1.0,     0.5,     0.0},
        {    -0.5,     0.0,     0.5,     0.0},
        { 1.0/6.0, 4.0/6.0, 1.0/6.0,     0.0}};
    for (int i=0; i<16; i++) {
        const int d = i >> 2, r = i & 3;
        for (int v=0; v<4; v++)
            for (int u=0; u<4; u++)
                buv[i][v*4 + u] = bsp[u][d]*bsp[v][r];
    }
    double tmp[1728]; // max size needed for N==50
    for (int rn=0; rn<Nmax-2; rn++) {
        const int K = 2*(rn + 3) + 8;
        for (int k=0; k<3; k++) {
            memset(tmp, 0, sizeof(double)*K*16);
            int idx = 0;
            for (int i=0; i<K; i++) {
                for (int j=0; j<16; j++) {
                    double sc = eigen[rn].x[k][i + j*K]; // x==Phi here
                    for (int y4=0; y4<16; y4+=4)
                        for (int x=0; x<4; x++)
                            tmp[idx + y4 + x] += sc*buv[j][y4 + x];
                }
                idx += 16;
            }
            // now replace 'Phi' by tmp array
            memcpy(const_cast<double*>(&eigen[rn].x[k][0]), tmp, sizeof(double)*K*16);
        }
    }
}

template <class U>
FarSubdivisionTables<U>::~FarSubdivisionTables() {
}


} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* FAR_SUBDIVISION_TABLES_H */

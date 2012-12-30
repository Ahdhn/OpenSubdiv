#if defined(__APPLE__)
    #include <OpenGL/gl3.h>
    #include <GLUT/glut.h>
#else
    #include <stdlib.h>
    #include <GL/glew.h>
    #include <GL/glut.h>
#endif

#include <osd/mutex.h>

#include <hbr/mesh.h>
#include <hbr/face.h>

#include <osd/vertex.h>
#include <osd/mesh.h>
#include <osd/cpuDispatcher.h>

#ifdef OPENSUBDIV_HAS_GLSL
    #include <osd/glslDispatcher.h>
#endif

#ifdef OPENSUBDIV_HAS_OPENCL
    #include <osd/clDispatcher.h>
    #include <osd/clspmvDispatcher.h>
#endif

#ifdef OPENSUBDIV_HAS_MKL
    #include <osd/mklDispatcher.h>
#endif

#ifdef OPENSUBDIV_HAS_BOOST
    #include <osd/ublasDispatcher.h>
#endif

#ifdef OPENSUBDIV_HAS_CUDA
    #include <osd/cudaDispatcher.h>
    #include <osd/cusparseDispatcher.h>

    #include <cuda_runtime_api.h>
    #include <cuda_gl_interop.h>

    #include "../common/cudaInit.h"
#endif

#include <common/shape_utils.h>

#include "../common/stopwatch.h"

#include <float.h>
#include <vector>
#include <fstream>
#include <sstream>

struct SimpleShape {
    std::string  name;
    Scheme       scheme;
    char const * data;

    SimpleShape() { }
    SimpleShape( char const * idata, char const * iname, Scheme ischeme )
        : name(iname), scheme(ischeme), data(idata) { }
};

std::vector<SimpleShape> g_defaultShapes;
int g_kernel = OpenSubdiv::OsdKernelDispatcher::kCPU;
int   g_frame = 100,
      g_repeatCount = 0;
int g_currentShape = 0;
int g_level = 2;
float g_cpuTime = 0;
float g_gpuTime = 0;
float g_moveScale = 1.0f;
std::vector<float> g_orgPositions,
                   g_positions,
                   g_normals;
Scheme             g_scheme;
bool g_regression = false;

OpenSubdiv::OsdMesh * g_osdmesh = 0;
OpenSubdiv::OsdVertexBuffer * g_vertexBuffer = 0;

/* for regression check */
OpenSubdiv::OsdMesh * g_cpu_osdmesh = 0;
OpenSubdiv::OsdVertexBuffer * g_cpu_vertexBuffer = 0;
#define PRECISION 1e-6

const char *getKernelName(int kernel) {

         if (kernel == OpenSubdiv::OsdKernelDispatcher::kCPU)
        return "CPU";
    else if (kernel == OpenSubdiv::OsdKernelDispatcher::kOPENMP)
        return "OpenMP";
    else if (kernel == OpenSubdiv::OsdKernelDispatcher::kCUDA)
        return "Cuda";
    else if (kernel == OpenSubdiv::OsdKernelDispatcher::kGLSL)
        return "GLSL";
    else if (kernel == OpenSubdiv::OsdKernelDispatcher::kCL)
        return "OpenCL";
    else if (kernel == OpenSubdiv::OsdKernelDispatcher::kMKL)
        return "MKL";
    else if (kernel == OpenSubdiv::OsdKernelDispatcher::kCLSPMV)
        return "ClSpMV";
    else if (kernel == OpenSubdiv::OsdKernelDispatcher::kCUSPARSE)
        return "CuSPARSE";
    else if (kernel == OpenSubdiv::OsdKernelDispatcher::kUBLAS)
        return "uBLAS";
    return "Unknown";
}

void
initializeShapes( ) {

#include <shapes/bigguy.h>
    g_defaultShapes.push_back(SimpleShape(bigguy, "BigGuy", kCatmark));

#include <shapes/al.h>
    g_defaultShapes.push_back(SimpleShape(al, "Al", kCatmark));

#include <shapes/bunny.h>
    g_defaultShapes.push_back(SimpleShape(bunny, "Bunny", kLoop));

#include <shapes/cupid.h>
    g_defaultShapes.push_back(SimpleShape(cupid, "Cupid", kCatmark));

#include <shapes/monsterfrog.h>
    g_defaultShapes.push_back(SimpleShape(monsterfrog, "MonsterFrog", kCatmark));

#include <shapes/torii.h>
    g_defaultShapes.push_back(SimpleShape(torii, "Torii", kCatmark));

#include <shapes/twist.h>
    g_defaultShapes.push_back(SimpleShape(twist, "Twist", kLoop));

#include <shapes/venus.h>
    g_defaultShapes.push_back(SimpleShape(venus, "Venus", kLoop));

#include <shapes/bilinear_cube.h>
    g_defaultShapes.push_back(SimpleShape(bilinear_cube, "bilinear_cube", kBilinear));

#include <shapes/catmark_cube_corner0.h>
    g_defaultShapes.push_back(SimpleShape(catmark_cube_corner0, "catmark_cube_corner0", kCatmark));

#include <shapes/catmark_cube_corner1.h>
    g_defaultShapes.push_back(SimpleShape(catmark_cube_corner1, "catmark_cube_corner1", kCatmark));

#include <shapes/catmark_cube_corner2.h>
    g_defaultShapes.push_back(SimpleShape(catmark_cube_corner2, "catmark_cube_corner2", kCatmark));

#include <shapes/catmark_cube_corner3.h>
    g_defaultShapes.push_back(SimpleShape(catmark_cube_corner3, "catmark_cube_corner3", kCatmark));

#include <shapes/catmark_cube_corner4.h>
    g_defaultShapes.push_back(SimpleShape(catmark_cube_corner4, "catmark_cube_corner4", kCatmark));

#include <shapes/catmark_cube_creases0.h>
    g_defaultShapes.push_back(SimpleShape(catmark_cube_creases0, "catmark_cube_creases0", kCatmark));

#include <shapes/catmark_cube_creases1.h>
    g_defaultShapes.push_back(SimpleShape(catmark_cube_creases1, "catmark_cube_creases1", kCatmark));

#include <shapes/catmark_cube.h>
    g_defaultShapes.push_back(SimpleShape(catmark_cube, "catmark_cube", kCatmark));

#include <shapes/catmark_dart_edgecorner.h>
    g_defaultShapes.push_back(SimpleShape(catmark_dart_edgecorner, "catmark_dart_edgecorner", kCatmark));

#include <shapes/catmark_dart_edgeonly.h>
    g_defaultShapes.push_back(SimpleShape(catmark_dart_edgeonly, "catmark_dart_edgeonly", kCatmark));

#include <shapes/catmark_edgecorner.h>
    g_defaultShapes.push_back(SimpleShape(catmark_edgecorner ,"catmark_edgecorner", kCatmark));

#include <shapes/catmark_edgeonly.h>
    g_defaultShapes.push_back(SimpleShape(catmark_edgeonly, "catmark_edgeonly", kCatmark));

#include <shapes/catmark_pyramid_creases0.h>
    g_defaultShapes.push_back(SimpleShape(catmark_pyramid_creases0, "catmark_pyramid_creases0", kCatmark));

#include <shapes/catmark_pyramid_creases1.h>
    g_defaultShapes.push_back(SimpleShape(catmark_pyramid_creases1, "catmark_pyramid_creases1", kCatmark));

#include <shapes/catmark_pyramid.h>
    g_defaultShapes.push_back(SimpleShape(catmark_pyramid, "catmark_pyramid", kCatmark));

#include <shapes/catmark_tent_creases0.h>
    g_defaultShapes.push_back(SimpleShape(catmark_tent_creases0, "catmark_tent_creases0", kCatmark));

#include <shapes/catmark_tent_creases1.h>
    g_defaultShapes.push_back(SimpleShape(catmark_tent_creases1, "catmark_tent_creases1", kCatmark));

#include <shapes/catmark_tent.h>
    g_defaultShapes.push_back(SimpleShape(catmark_tent, "catmark_tent", kCatmark));

#include <shapes/catmark_square_hedit0.h>
    g_defaultShapes.push_back(SimpleShape(catmark_square_hedit0, "catmark_square_hedit0", kCatmark));

#include <shapes/catmark_square_hedit1.h>
    g_defaultShapes.push_back(SimpleShape(catmark_square_hedit1, "catmark_square_hedit1", kCatmark));

#include <shapes/catmark_square_hedit2.h>
    g_defaultShapes.push_back(SimpleShape(catmark_square_hedit2, "catmark_square_hedit2", kCatmark));

#include <shapes/catmark_square_hedit3.h>
    g_defaultShapes.push_back(SimpleShape(catmark_square_hedit3, "catmark_square_hedit3", kCatmark));



#include <shapes/loop_cube_creases0.h>
    g_defaultShapes.push_back(SimpleShape(loop_cube_creases0, "loop_cube_creases0", kLoop));

#include <shapes/loop_cube_creases1.h>
    g_defaultShapes.push_back(SimpleShape(loop_cube_creases1, "loop_cube_creases1", kLoop));

#include <shapes/loop_cube.h>
    g_defaultShapes.push_back(SimpleShape(loop_cube, "loop_cube", kLoop));

#include <shapes/loop_icosahedron.h>
    g_defaultShapes.push_back(SimpleShape(loop_icosahedron, "loop_icosahedron", kLoop));

#include <shapes/loop_saddle_edgecorner.h>
    g_defaultShapes.push_back(SimpleShape(loop_saddle_edgecorner, "loop_saddle_edgecorner", kLoop));

#include <shapes/loop_saddle_edgeonly.h>
    g_defaultShapes.push_back(SimpleShape(loop_saddle_edgeonly, "loop_saddle_edgeonly", kLoop));

#include <shapes/loop_triangle_edgecorner.h>
    g_defaultShapes.push_back(SimpleShape(loop_triangle_edgecorner, "loop_triangle_edgecorner", kLoop));

#include <shapes/loop_triangle_edgeonly.h>
    g_defaultShapes.push_back(SimpleShape(loop_triangle_edgeonly, "loop_triangle_edgeonly", kLoop));
}

void
updateGeom(bool reportMaxError=false) {
    int nverts = (int)g_orgPositions.size() / 3;

    std::vector<float> vertex;
    vertex.reserve(nverts*6);

    const float *p = &g_orgPositions[0];
    const float *n = &g_normals[0];

    float r = sin(g_frame*0.001f) * g_moveScale;
    for (int i = 0; i < nverts; ++i) {
        float ct = cos(p[2] * r);
        float st = sin(p[2] * r);
        g_positions[i*3+0] = p[0]*ct + p[1]*st;
        g_positions[i*3+1] = -p[0]*st + p[1]*ct;
        g_positions[i*3+2] = p[2];

        p += 3;
    }

    p = &g_positions[0];
    for (int i = 0; i < nverts; ++i) {
        vertex.push_back(p[0]);
        vertex.push_back(p[1]);
        vertex.push_back(p[2]);
        vertex.push_back(n[0]);
        vertex.push_back(n[1]);
        vertex.push_back(n[2]);

        p += 3;
        n += 3;
    }

    if (!g_vertexBuffer)
        g_vertexBuffer = g_osdmesh->InitializeVertexBuffer(6);
    g_vertexBuffer->UpdateData(&vertex[0], nverts);

    g_cpuTime = (float) g_osdmesh->Subdivide(g_vertexBuffer, NULL) * 1000.0f;
    g_gpuTime = (float) g_osdmesh->Synchronize() * 1000.0f;

    double frameVertsPerMillisecond = g_osdmesh->GetFarMesh()->GetNumVertices() / (g_cpuTime+g_gpuTime);
    printf(" %f", frameVertsPerMillisecond);

    if (g_regression and reportMaxError) {
        if (!g_cpu_vertexBuffer)
            g_cpu_vertexBuffer = g_cpu_osdmesh->InitializeVertexBuffer(6);
        g_cpu_vertexBuffer->UpdateData(&vertex[0], nverts);
        g_cpu_osdmesh->Subdivide(g_cpu_vertexBuffer, NULL) * 1000.0f;

        float maxerror = 0.0;
        int level = g_osdmesh->GetLevel();
        int elemsPerVert = g_vertexBuffer->GetNumElements();
        int offset = g_osdmesh->GetFarMesh()->GetSubdivision()->GetFirstVertexOffset(level) * elemsPerVert;
        int nfineverts = g_osdmesh->GetFarMesh()->GetSubdivision()->GetNumVertices(level);

        float* expected = (float*) g_cpu_vertexBuffer->GetCpuBuffer() + offset;
        float* actual =   (float*) g_vertexBuffer->GetCpuBuffer()     + offset;

        for (int i = 0; i < nfineverts*elemsPerVert; i++)
            maxerror = fmaxf(maxerror, fabs(expected[i] - actual[i]));
        printf(" maxerror=%e", maxerror);
    }
}

//------------------------------------------------------------------------------
inline void
cross(float *n, const float *p0, const float *p1, const float *p2) {

    float a[3] = { p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2] };
    float b[3] = { p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2] };
    n[0] = a[1]*b[2]-a[2]*b[1];
    n[1] = a[2]*b[0]-a[0]*b[2];
    n[2] = a[0]*b[1]-a[1]*b[0];

    float rn = 1.0f/sqrtf(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
    n[0] *= rn;
    n[1] *= rn;
    n[2] *= rn;
}

//------------------------------------------------------------------------------
inline void
normalize(float * p) {

    float dist = sqrtf( p[0]*p[0] + p[1]*p[1]  + p[2]*p[2] );
    p[0]/=dist;
    p[1]/=dist;
    p[2]/=dist;
}


//------------------------------------------------------------------------------
static void
calcNormals(OpenSubdiv::OsdHbrMesh * mesh, std::vector<float> const & pos, std::vector<float> & result ) {

    // calc normal vectors
    int nverts = (int)pos.size()/3;

    int nfaces = mesh->GetNumCoarseFaces();
    for (int i = 0; i < nfaces; ++i) {

        OpenSubdiv::OsdHbrFace * f = mesh->GetFace(i);

        float const * p0 = &pos[f->GetVertex(0)->GetID()*3],
                    * p1 = &pos[f->GetVertex(1)->GetID()*3],
                    * p2 = &pos[f->GetVertex(2)->GetID()*3];

        float n[3];
        cross( n, p0, p1, p2 );

        for (int j = 0; j < f->GetNumVertices(); j++) {
            int idx = f->GetVertex(j)->GetID() * 3;
            result[idx  ] += n[0];
            result[idx+1] += n[1];
            result[idx+2] += n[2];
        }
    }
    for (int i = 0; i < nverts; ++i)
        normalize( &result[i*3] );
}

void
createOsdMesh( const char * shape, int level, int kernel, Scheme scheme=kCatmark ) {
    // start timer
    Stopwatch s;
    s.Start();

    // generate Hbr representation from "obj" description
    OpenSubdiv::OsdHbrMesh * hmesh = simpleHbr<OpenSubdiv::OsdVertex>(shape, scheme, g_orgPositions);

    g_normals.resize(g_orgPositions.size(),0.0f);
    g_positions.resize(g_orgPositions.size(),0.0f);
    calcNormals( hmesh, g_orgPositions, g_normals );


    // generate Osd mesh from Hbr mesh
    if (g_osdmesh) delete g_osdmesh;
    g_osdmesh = new OpenSubdiv::OsdMesh();
    g_osdmesh->Create(hmesh, level, kernel);
    if (g_vertexBuffer) {
        delete g_vertexBuffer;
        g_vertexBuffer = NULL;
    }

    if (g_regression) {
        if (g_cpu_osdmesh) delete g_cpu_osdmesh;
        g_cpu_osdmesh = new OpenSubdiv::OsdMesh();
        g_cpu_osdmesh->Create(hmesh, level, OpenSubdiv::OsdKernelDispatcher::kCPU);
        if (g_cpu_vertexBuffer) {
            delete g_cpu_vertexBuffer;
            g_cpu_vertexBuffer = NULL;
        }
    }

    // Hbr mesh can be deleted
    delete hmesh;

    updateGeom( /* reportMaxError = */ true );

    s.Stop();
    printf(" ttff=%f",  float(s.GetElapsed() * 1000.0f));
}

int main(int argc, char* argv[]) {

    std::string str;
    if (argc > 1) {
        std::ifstream ifs(argv[1]);
        if (ifs) {
            std::stringstream ss;
            ss << ifs.rdbuf();
            ifs.close();
            str = ss.str();

            g_defaultShapes.push_back(SimpleShape(str.c_str(), argv[1], kCatmark));
        }
    }

    initializeShapes();

    // Register Osd compute kernels
    OpenSubdiv::OsdCpuKernelDispatcher::Register();

#if OPENSUBDIV_HAS_GLSL
    OpenSubdiv::OsdGlslKernelDispatcher::Register();
#endif

#if OPENSUBDIV_HAS_OPENCL
    OpenSubdiv::OsdClKernelDispatcher::Register();
    OpenSubdiv::OsdClSpMVKernelDispatcher::Register();
#endif

#if OPENSUBDIV_HAS_MKL
    OpenSubdiv::OsdMklKernelDispatcher::Register();
#endif

#if OPENSUBDIV_HAS_BOOST
    OpenSubdiv::OsdUBlasKernelDispatcher::Register();
#endif

#if OPENSUBDIV_HAS_CUDA
    OpenSubdiv::OsdCudaKernelDispatcher::Register();
    OpenSubdiv::OsdCusparseKernelDispatcher::Register();

    // Note: This function randomly crashes with linux 5.0-dev driver.
    // cudaGetDeviceProperties overrun stack..?
    cudaGLSetGLDevice( cutGetMaxGflopsDeviceId() );
#endif

    int nKernels = OpenSubdiv::OsdKernelDispatcher::kMAX;
    for(int i = 0; i < nKernels; ++i)
        if(OpenSubdiv::OsdKernelDispatcher::HasKernelType(
               OpenSubdiv::OsdKernelDispatcher::KernelType(i)))
            printf(" has_%s=1", getKernelName(i));

    const char *filename = NULL;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-l") || !strcmp(argv[i], "--level"))
            g_level = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-c") || !strcmp(argv[i], "--count"))
            g_repeatCount = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-m") || !strcmp(argv[i], "--model"))
            g_currentShape = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-k") || !strcmp(argv[i], "--kernel"))
            g_kernel = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-r") || !strcmp(argv[i], "--regression"))
            g_regression = true;
        else if (!strcmp(argv[i], "-s") || !strcmp(argv[i], "--spy")) {
            osdSpMVKernel_DumpSpy_FileName = argv[++i];
        }
        else
            filename = argv[i];
    }
    printf(" level=%d", g_level);
    printf(" frames=%d", g_repeatCount);
    printf(" kernel=%s", getKernelName(g_kernel));
    printf(" model=%s", g_defaultShapes[ g_currentShape ].name.c_str());

    createOsdMesh( g_defaultShapes[ g_currentShape ].data,
                   g_level, g_kernel,
                   g_defaultShapes[ g_currentShape ].scheme );

    printf(" nverts=%d", g_osdmesh->GetFarMesh()->GetSubdivision()->GetNumVertices(g_level));

    for(int frame = 0; frame < g_repeatCount; frame++)
        updateGeom();

    printf("\n");
    return 0;
}

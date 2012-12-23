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

OpenSubdiv::OsdMesh * g_osdmesh = 0;
OpenSubdiv::OsdVertexBuffer * g_vertexBuffer = 0;

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

    //printf(" nverts=%d", g_osdmesh->GetFarMesh()->GetSubdivision()->GetNumVertices(g_level));

    printf("\n");
}

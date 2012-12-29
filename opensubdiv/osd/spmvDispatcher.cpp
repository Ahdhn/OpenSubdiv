#include "../version.h"
#include "../osd/mutex.h"
#include "../osd/spmvDispatcher.h"
#include "../osd/spmvKernel.h"

#include <stdio.h>

char* osdSpMVKernel_DumpSpy_FileName = NULL;

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

OsdSpMVKernelDispatcher::OsdSpMVKernelDispatcher( int levels )
    : OsdCpuKernelDispatcher(levels), matrix_id(0)
{
#if BENCHMARKING
    printf("\n");
#endif
}

OsdSpMVKernelDispatcher::~OsdSpMVKernelDispatcher() {
    if (_vdesc)
        delete _vdesc;
}

void
OsdSpMVKernelDispatcher::BindVertexBuffer(OsdVertexBuffer *vertex, OsdVertexBuffer *varying) {

    if (vertex)
        _currentVertexBuffer = dynamic_cast<OsdCpuVertexBuffer *>(vertex);
    else
        _currentVertexBuffer = NULL;

    if (varying)
        _currentVaryingBuffer = dynamic_cast<OsdCpuVertexBuffer *>(varying);
    else
        _currentVaryingBuffer = NULL;

    _vdesc = new SpMVVertexDescriptor(this,
            _currentVertexBuffer  ? _currentVertexBuffer->GetNumElements()  : 0,
            _currentVaryingBuffer ? _currentVaryingBuffer->GetNumElements() : 0);
}

int
OsdSpMVKernelDispatcher::CopyNVerts(int nVerts, int index) {
    for (int i = 0; i < nVerts; i++)
        _vdesc->AddWithWeight(NULL, index+i, index+i, 1.0);
    return nVerts;
}

void
OsdSpMVKernelDispatcher::WriteMatrix(coo_matrix1* M, std::string filename) {
    FILE* ofile = fopen(filename.c_str(), "w");
    assert(ofile != NULL);

    fprintf(ofile, "%%%%MatrixMarket matrix coordinate real general\n");
    fprintf(ofile, "%d %d %d\n", M->size1(), M->size2(), M->nnz());

    for(int i = 0; i < M->nnz(); i++)
        fprintf(ofile, "%d %d %10.3g\n",
                M->index1_data()[i],
                M->index2_data()[i],
                M->value_data()[i]);

    fclose(ofile);
}

void
OsdSpMVKernelDispatcher::WriteMatrix(csr_matrix1* M, std::string filename) {
    FILE* ofile = fopen(filename.c_str(), "w");
    assert(ofile != NULL);

    fprintf(ofile, "%%%%MatrixMarket matrix coordinate real general\n");
    fprintf(ofile, "%d %d %d\n", M->size1(), M->size2()+1, M->nnz());

    int* rows = &M->index1_data()[0];
    int* cols = &M->index2_data()[0];
    float* vals = &M->value_data()[0];

    for(int r = 0; r < M->size1(); r++) {
        for(int offset = rows[r]; offset < rows[r+1]; offset++) {
	    int c      = cols[offset];
            float val  = vals[offset];
            fprintf(ofile, "%d %d %10.3g\n", r+1, c+1, val);
        }
    }

    fclose(ofile);
}

} // end namespace OPENSUBDIV_VERSION

} // end namespace OpenSubdiv

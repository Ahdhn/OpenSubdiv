#include "../version.h"

void
LogicalSpMV_csr_cpu(int m, int n, int nnz,
    float *val, int *colIndx, int *rowPtrs,
    float *__restrict__ d_in, float *__restrict__ d_out);

#include "../version.h"

void
logical_spmv_kernel(int m, int n, int nnz,
    float *val, int *colIndx, int *rowPtrs,
    float *__restrict__ d_in, float *__restrict__ d_out);

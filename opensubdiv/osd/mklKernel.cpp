#include "../version.h"

#include <math.h>
#include <omp.h>
#include <xmmintrin.h>

void
logical_spmv_kernel(int m, int n, int nnz,
    float *vals, int *colInds, int *rowPtrs,
    float *__restrict__ d_in, float *__restrict__ d_out)
{
    omp_set_num_threads( omp_get_num_procs() );

    #pragma omp parallel
    {
        int size = omp_get_num_threads(),
            rank = omp_get_thread_num(),
            rows_per_thread = (m + size - 1) / size,
            start_row = rank * rows_per_thread,
            end_row = fmin(m, (rank+1) * rows_per_thread);

        int row = start_row;
        int start_k = rowPtrs[start_row],
            end_k   = rowPtrs[end_row];

        register __m128
            out03v = _mm_setzero_ps(),
            out45v = _mm_setzero_ps();
        int out_idx, in_idx;


        int next_row_k = rowPtrs[row+1];

        for (int k = start_k; k < end_k; k++) {

            in_idx = 6*colInds[k];

            register __m128 ignore,
                   in03v = _mm_loadu_ps( &d_in[in_idx] ),
                   in45v = _mm_loadl_pi( ignore, (const __m64*) &d_in[in_idx+4] ),
                   weightv = _mm_load1_ps( &vals[k] );

            out03v = _mm_add_ps(out03v, _mm_mul_ps(weightv, in03v));
            out45v = _mm_add_ps(out45v, _mm_mul_ps(weightv, in45v));

            if (k+1 == next_row_k) {
                out_idx  = 6*row;
                _mm_storeu_ps( &d_out[ out_idx ], out03v );
                _mm_storel_pi( (__m64*) &d_out[ out_idx+4 ], out45v );
                out03v = _mm_setzero_ps();
                out45v = _mm_setzero_ps();
                row += 1;
                next_row_k = rowPtrs[row+1];
            }
        }
    }
}

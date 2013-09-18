#include <algorithm>
#include <cstdio>

#include <omp.h>
#include <xmmintrin.h>


void LogicalSpMV_csr1_cpu(int m, int *rowPtrs, int *colInds, float *vals, float *d_in, float *d_out) {
    omp_set_num_threads( omp_get_num_procs() );

    int *__restrict__ rrows = &rowPtrs[0];
    int *__restrict__ rcols = &colInds[0];
    float *__restrict__ rvals = &vals[0];

    #pragma omp parallel
    {

        int size = omp_get_num_threads(),
            rank = omp_get_thread_num(),
            rows_per_thread = (m + size - 1) / size,
            start_row = rank * rows_per_thread,
            end_row = std::min(m, (rank+1) * rows_per_thread);

        int row = start_row;
        int start_k = rrows[start_row],
            end_k   = rrows[end_row];

        register __m128
            out03v = _mm_setzero_ps(),
            out45v = _mm_setzero_ps();
        int out_idx, in_idx;


        int next_row_k = rrows[row+1];

        for (int k = start_k; k < end_k; k++) {

            in_idx = 6*rcols[k];

            register __m128 ignore,
                   in03v = _mm_loadu_ps( &d_in[in_idx] ),
                   in45v = _mm_loadl_pi( ignore, (const __m64*) &d_in[in_idx+4] ),
                   weightv = _mm_load1_ps( &rvals[k] );

            out03v = _mm_add_ps(out03v, _mm_mul_ps(weightv, in03v));
            out45v = _mm_add_ps(out45v, _mm_mul_ps(weightv, in45v));

            if (k+1 == next_row_k) {
                out_idx  = 6*row;
                _mm_storeu_ps( &d_out[ out_idx ], out03v );
                _mm_storel_pi( (__m64*) &d_out[ out_idx+4 ], out45v );
                out03v = _mm_setzero_ps();
                out45v = _mm_setzero_ps();
                row += 1;
                next_row_k = rrows[row+1];
            }
        }
    }
}

void LogicalSpMV_csr0_cpu(int m, int *rowPtrs, int *colInds, float *vals, float *d_in, float *d_out) {

    double start = omp_get_wtime();
    #pragma omp parallel for
    for (int i = 0; i < m; i++) {

        int base = rowPtrs[i];
        int nVals = rowPtrs[i+1] - base;

        if (nVals > 0) {

            register __m128
                out03v = _mm_setzero_ps(),
                       out45v = _mm_setzero_ps();

            for (int idx = base; idx < base+nVals; idx++) {

                int in_col = colInds[idx]*6;
                register __m128 ignore,
                         in03v = _mm_loadu_ps( &d_in[in_col] ),
                         in45v = _mm_loadl_pi( ignore, (const __m64*) &d_in[in_col+4] ),
                         weightv = _mm_load1_ps( &vals[idx] );

                out03v = _mm_add_ps(out03v, _mm_mul_ps(weightv, in03v));
                out45v = _mm_add_ps(out45v, _mm_mul_ps(weightv, in45v));

            }

            int out_row = 6*i;
            _mm_storeu_ps( &d_out[out_row], out03v );
            _mm_storel_pi( (__m64*) &d_out[out_row+4], out45v );

        }
    }

    double duration = omp_get_wtime() - start;
    int bytes = m*sizeof(int) + rowPtrs[m]*(sizeof(int)*sizeof(float)) + 6*m*sizeof(float);
    printf("\n LogicalSpMV_csr0_cpu ran at %f GB/s\n", bytes / 1024.0 / 1024.0 / 1024.0 / duration);
}

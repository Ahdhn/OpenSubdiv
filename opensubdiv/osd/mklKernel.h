#ifndef OSD_MKL_KERNEL_H
#define OSD_MKL_KERNEL_H

void LogicalSpMV_csr1_cpu(int m, int *rowPtrs, int *colInds, float *vals, float *d_in, float *d_out);
void LogicalSpMV_csr0_cpu(int m, int *rowPtrs, int *colInds, float *vals, float *d_in, float *d_out);
void LogicalSpMV_coo0_cpu(int *schedule, int *rowInds, int *colInds, float *vals, float *d_in, float *d_out);

#endif // define OSD_MKL_KERNEL_H
